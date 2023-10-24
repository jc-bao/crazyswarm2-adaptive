from crazyflie_py import Crazyswarm
import numpy as np
from icecream import ic
import tf2_ros
import transforms3d as tf3d
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
import os
import pickle
from .pid_controller import PIDController, PIDParam, PIDState
from copy import deepcopy

import jax
import chex
from jax import numpy as jnp

import quadjax
from quadjax.envs.quad3d_free import Quad3D, get_controller
from quadjax.train import ActorCritic, Compressor, Adaptor
from quadjax import controllers
from quadjax.dynamics import utils
from quadjax.dynamics.dataclass import EnvParams3D, EnvState3D, Action3D

class Crazyflie:

    def __init__(self, task = 'tracking', controller_name = 'pid', controller_params = None) -> None:
        # create jax environment for reference
        self.env = Quad3D(task=task, dynamics='bodyrate', obs_type='quad', lower_controller='base', enable_randomizer=False, disturb_type='none')
        self.rng = jax.random.PRNGKey(0)
        rng_params, self.rng = jax.random.split(self.rng)
        self.env_params = self.env.sample_params(rng_params)
        rng_state, self.rng = jax.random.split(self.rng)
        obs, info, self.state_sim = self.env.reset(rng_state)
        # deep copy flax.struct.dataclass 
        self.state_real = deepcopy(self.state_sim)

        # real-world parameters
        self.world_center = jnp.array([0.0, 0.0, 1.5])
        self.xyz_min = jnp.array([-2.0, -3.0, -2.0])
        self.xyz_max = jnp.array([2.0, 2.0, 1.5])

        # crazyswarm related initialization
        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.cf = self.swarm.allcfs.crazyflies[0]
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(buffer=self.tf_buffer, node=self.swarm.allcfs)

        # base controller: PID
        self.base_controller, self.base_control_params = get_controller(self.env, 'pid', None)
        self.default_control_params = deepcopy(self.base_control_params)
        # controller to test out
        self.controller, self.control_params = get_controller(self.env, controller_name, controller_params)
        self.default_control_params = deepcopy(self.control_params)

        # ROS related initialization
        self.pos_kf = jnp.zeros(3)
        self.quat_kf = jnp.array([0.0, 0.0, 0.0, 1.0])
        self.pos_hist = jnp.zeros((self.env.default_params.adapt_horizon+3, 3), dtype=jnp.float32)
        self.quat_hist = jnp.zeros((self.env.default_params.adapt_horizon+3, 4), dtype=jnp.float32)
        func = partial(self.state_callback_cf, cfid=1)
        self.swarm.allcfs.create_subscription(PoseStamped, 'cf1/pose', func, int(1/self.env_params.dt))
        # state publisher
        # initialize publisher
        rate = int(1.0 / self.env_params.dt)
        self.world_center_pub = self.swarm.allcfs.create_publisher(PoseStamped, 'world_center', rate)
        self.pos_sim_pub = self.swarm.allcfs.create_publisher(PoseStamped, 'pos_sim', rate)
        self.pos_real_pub = self.swarm.allcfs.create_publisher(PoseStamped, 'pos_real', rate)
        self.pos_tar_pub = self.swarm.allcfs.create_publisher(PoseStamped, 'pos_tar', rate)
        self.traj_pub = self.swarm.allcfs.create_publisher(Path, 'traj', rate)

    def state_callback_cf(self, data, cfid):
        pos = data.pose.position
        quat = data.pose.orientation
        self.pos_kf = np.array([pos.x, pos.y, pos.z])
        self.quat_kf = np.array([quat.x, quat.y, quat.z, quat.w])

    def get_real_state(self):
        dt = self.env.default_params.dt

        vel_hist = jnp.diff(self.pos_hist, axis=0) / dt
        dquat_hist = jnp.diff(self.quat_hist, axis=0) # NOTE diff here will make the length of dquat_hist 1 less than the others
        omega_hist = 2 * dquat_hist[:, :-1] / (jnp.linalg.norm(self.quat_hist[:-1,:-1], axis=-1, keepdims=True)+1e-3) / dt

        return EnvState3D(
            # drone
            pos = self.pos_hist[-1], 
            vel = vel_hist[-1], 
            omega = omega_hist[-1],
            omega_tar = jnp.zeros(3),
            quat = self.quat_hist[-1],
            # obj
            pos_obj = jnp.zeros(3),
            vel_obj = jnp.zeros(3),
            # hook
            pos_hook = jnp.zeros(3),
            vel_hook = jnp.zeros(3),
            # rope
            l_rope = 0.0,
            zeta = jnp.zeros(3),
            zeta_dot = jnp.zeros(3),
            f_rope = jnp.zeros(3),
            f_rope_norm = 0.0,
            # trajectory
            pos_tar = self.state_sim.pos_tar,
            vel_tar = self.state_sim.vel_tar,
            acc_tar = self.state_sim.acc_tar,
            pos_traj = self.state_sim.pos_traj,
            vel_traj = self.state_sim.vel_traj,
            acc_traj = self.state_sim.acc_traj,
            # debug value
            last_thrust = self.state_sim.last_thrust,
            last_torque = self.state_sim.last_torque,
            # step
            time = self.state_sim.time,
            # disturbance
            f_disturb = jnp.zeros(3),
            # trajectory information for adaptation
            vel_hist = vel_hist, 
            omega_hist = omega_hist,
            action_hist = self.state_real.action_hist,
            # control parameters
            control_params = self.control_params,
        )

    def get_drone_state(self):
        trans_mocap = self.tf_buffer.lookup_transform('world', 'cf1', rclpy.time.Time())
        pos = trans_mocap.transform.translation
        quat = trans_mocap.transform.rotation
        return jnp.array([pos.x, pos.y, pos.z]) - self.world_center, jnp.array([quat.x, quat.y, quat.z, quat.w])

    def set_attirate(self, omega_target, thrust_target):
        # convert to degree
        omega_target = np.array(omega_target, dtype=np.float64) / np.pi * 180.0 # NOTE: make sure the type is float64
        acc_z_target = thrust_target / self.env.default_params.m
        self.cf.cmdFullState(
            np.zeros(3),np.zeros(3),np.array([0,0,acc_z_target]), 0.0, omega_target)

    def goto(self, pos, timelimit=3.0):
        dt = self.env.default_params.dt
        stablize_time = 1.0
        # linear interpolation
        pos_start_real = self.get_drone_state()[0]
        pos_start_sim = self.state_sim.pos
        pos_end = pos
        N = int(timelimit / dt)
        N_move = int((timelimit - stablize_time) / dt)
        for i in range(N):
            if i > N_move:
                pos_tar_real = pos_end
                vel_tar_real = np.zeros(3)
                pos_tar_sim = pos_end
                vel_tar_sim = np.zeros(3)
            else:
                pos_tar_real = pos_start_real + (pos_end - pos_start_real) * i / N_move
                vel_tar_real = (pos_end - pos_start_real) / timelimit
                pos_tar_sim = pos_start_sim + (pos_end - pos_start_sim) * i / N_move
                vel_tar_sim = (pos_end - pos_start_sim) / timelimit
            if self.state_real.pos[2] < (-self.world_center[2]+0.1):
                pos_tar_real = jnp.array([0.0, 0.0, -self.world_center[2]+0.15])
                vel_tar_real = jnp.zeros(3)
            state_real_replaced = self.state_real.replace(pos_tar=pos_tar_real, vel_tar=vel_tar_real, acc_tar = jnp.zeros(3))
            action, _, _ = self.base_controller(None, state_real_replaced, self.env_params, None, self.base_control_params, None)
            # thrust = 0.027*(9.81+1.0)
            # action = jnp.array([thrust/0.8*2.0-1.0, 0.01/10.0, 0.0, 0.0]) 
            state_sim_replaced = self.state_sim.replace(pos_tar=pos_tar_sim, vel_tar=vel_tar_sim, acc_tar = jnp.zeros(3))
            action_sim, _, _ = self.controller(None, state_sim_replaced, self.env_params, None, self.control_params, None)
            next_state_dict = self.step(action, action_sim)
            self.state_real = self.state_real.replace(time=0)
            self.state_sim = self.state_sim.replace(time=0)
        return next_state_dict
    
    def reset(self):
        # NOTE use this to estabilish connection
        for _ in range(2):
            self.set_attirate(np.zeros(3), 0.0)
            self.timeHelper.sleepForRate(10.0)
        reset_rng, self.rng = jax.random.split(self.rng)
        # reset controller
        self.base_control_params = self.default_control_params
        self.control_params = self.default_control_params
        # fly to initial point
        next_state_dict = self.goto(self.state_real.pos_tar, timelimit=3.0)
        self.state_real = self.get_real_state()
        # reset simulator
        # obs_sim, info_sim, self.state_sim = self.env.reset(reset_rng, self.env_params)
        return next_state_dict
    
    def step(self, action: jnp.ndarray, action_sim = None):
        if action_sim is None:
            action_sim = action
        rng_step, self.rng = jax.random.split(self.rng)
        # step simulator state
        obs_sim, self.state_sim, reward_sim, done_sim, info_sim = self.env.step(rng_step, self.state_sim, action_sim, self.env_params) 
        
        # step real-world state
        thrust_tar = (action[0]+1.0)/2.0*self.env.default_params.max_thrust
        omega_tar = action[1:4] * self.env.default_params.max_omega
        self.set_attirate(omega_tar, thrust_tar)
        self.timeHelper.sleepForRate(1.0/self.env.default_params.dt)
        # update real-world state
        pos, quat = self.get_drone_state()
        self.pos_hist = jnp.concatenate([self.pos_hist[1:], pos.reshape(1,3)], axis=0)
        self.quat_hist = jnp.concatenate([self.quat_hist[1:], quat.reshape(1,4)], axis=0)
        self.state_real = self.get_real_state()
        obs_real = self.env.get_obs(self.state_real, self.env_params)
        reward_real = self.env.reward_fn(self.state_real, self.env_params)
        done_real = self.env.is_terminal(self.state_real, self.env_params)
        info_real = self.env.get_info(self.state_real, self.env_params)

        self.pub_state()

        return {
            'real': [obs_real, self.state_real, reward_real, done_real, info_real],
            'sim': [obs_sim, self.state_sim, reward_sim, done_sim, info_sim],
        }
    
    def get_pose_msg(self, pos, quat):
        msg = PoseStamped()
        pos = np.array(pos + self.world_center, dtype=np.float64)
        quat = np.array(quat, dtype=np.float64)
        msg.header.frame_id = 'world'
        msg.header.stamp = rclpy.time.Time().to_msg()
        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1]
        msg.pose.position.z = pos[2]
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        return msg
    
    def get_path_msg(self, pos_traj):
        path_msg = Path()
        path_msg.header.frame_id = 'world'
        for pos in pos_traj:
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = 'world'
            pos = np.array(pos + self.world_center, dtype=np.float64)
            pose_msg.pose.position.x = pos[0]
            pose_msg.pose.position.y = pos[1]
            pose_msg.pose.position.z = pos[2]
            path_msg.poses.append(pose_msg)
        return path_msg

    def pub_state(self):
        self.world_center_pub.publish(self.get_pose_msg(self.world_center, np.array([0.0, 0.0, 0.0, 1,0])))
        self.pos_sim_pub.publish(self.get_pose_msg(self.state_sim.pos, self.state_sim.quat))
        self.pos_real_pub.publish(self.get_pose_msg(self.state_real.pos, self.state_real.quat))
        self.pos_tar_pub.publish(self.get_pose_msg(self.state_real.pos_tar, np.array([0.0, 0.0, 0.0, 1,0])))
        self.traj_pub.publish(self.get_path_msg(self.state_real.pos_traj))

    def emergency(self, obs):
        self.cf.emergency()
        self.reset()
        raise ValueError
    
def main(repeat_times = 1, filename = ''):
    env = Crazyflie(task='hovering', controller_name='pid')

    print('reset...')
    next_state_dict = env.reset()
    obs_real, state_real, reward_real, done_real, info_real = next_state_dict['real']
    obs_sim, state_sim, reward_sim, done_sim, info_sim = next_state_dict['sim']

    '''
    print('main task...')
    rng = jax.random.PRNGKey(1)
    state_real_seq, obs_real_seq, reward_real_seq = [], [], []
    state_sim_seq, obs_sim_seq, reward_sim_seq = [], [], []
    control_seq = []
    n_dones = 0
    while n_dones < repeat_times:
        state_real_seq.append(state_real)
        state_sim_seq.append(state_sim)

        rng, rng_act = jax.random.split(rng)
        action, env.control_params, control_info = env.controller(obs_real, state_real, env.env_params, rng_act, env.control_params, info_real)

        # manually record certain control parameters into state_seq
        control_params = env.control_params
        if hasattr(control_params, 'd_hat') and hasattr(control_params, 'vel_hat'):
            control_seq.append({'d_hat': control_params.d_hat, 'vel_hat': control_params.vel_hat})
        if hasattr(control_params, 'a_hat'):
            control_seq.append({'a_hat': control_params.a_hat})
        if hasattr(control_params, 'quat_desired'):
            control_seq.append({'quat_desired': control_params.quat_desired})
        next_state_dict = env.step(action)
        obs_real, state_real, reward_real, done_real, info_real = next_state_dict['real']
        obs_sim, state_sim, reward_sim, done_sim, info_sim = next_state_dict['sim']

        if done_real:
            control_params = env.controller.update_params(env.env_params, control_params)
            n_dones += 1

        reward_real_seq.append(reward_real)
        reward_sim_seq.append(reward_sim)
        obs_real_seq.append(obs_real)
        obs_sim_seq.append(obs_sim)

    print('landing...')
    env.goto(jnp.array([0.0, 0.0, -env.world_center[2]+0.1]), timelimit=3.0)

    print('plotting...')
    # convert state into dict
    state_real_seq_dict = [s.__dict__ for s in state_real_seq]
    state_sim_seq_dict = [s.__dict__ for s in state_sim_seq]
    if len(control_seq) > 0:
        # merge control_seq into state_seq with dict
        for i in range(len(state_real_seq)):
            state_real_seq_dict[i] = {**state_real_seq_dict[i], **control_seq[i]}
            state_sim_seq_dict[i] = {**state_sim_seq_dict[i], **control_seq[i]}
    with open(f"{quadjax.get_package_path()}/../results/real_state_seq_{filename}.pkl", "wb") as f:
        pickle.dump(state_real_seq_dict, f)
    with open(f"{quadjax.get_package_path()}/../results/sim_state_seq_{filename}.pkl", "wb") as f:
        pickle.dump(state_sim_seq_dict, f)
    utils.plot_states(state_real_seq_dict, obs_real_seq, reward_real_seq, env.env_params, 'real'+filename)
    utils.plot_states(state_sim_seq_dict, obs_sim_seq, reward_sim_seq, env.env_params, 'sim'+filename)
    '''

    rclpy.shutdown()

if __name__ == "__main__":
    main()