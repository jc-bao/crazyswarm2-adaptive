from crazyflie_py import Crazyswarm
import numpy as np
from icecream import ic
import tf2_ros
import transforms3d as tf3d
import rclpy
from geometry_msgs.msg import PoseStamped
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

    def __init__(self, task = 'hovering', controller_name = 'pid', controller_params = None) -> None:
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
        self.world_center = jnp.array([0.5, 0.0, 1.5])
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

    def state_callback_cf(self, data, cfid):
        pos = data.pose.position
        quat = data.pose.orientation
        self.pos_kf = np.array([pos.x, pos.y, pos.z])
        self.quat_kf = np.array([quat.x, quat.y, quat.z, quat.w])

    def get_real_state(self):
        dt = self.env.default_params.dt

        vel_hist = jnp.diff(self.pos_hist, axis=0) / dt
        dquat_hist = jnp.diff(self.quat_hist, axis=0)
        omega_hist = 2 * dquat_hist[:, :-1] / (jnp.linalg.norm(self.quat_hist[:,:-1], axis=-1, keepdims=True)+1e-3) / dt

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
        omega_target = omega_target / np.pi * 180.0
        acc_z_target = thrust_target / self.env.default_params.mass
        self.cf.cmdFullState(
            np.zeros(3),np.zeros(3),np.array([0,0,acc_z_target]), 0.0, omega_target)
        

    def goto(self, pos, timelimit=3.0):
        dt = self.env.default_params.dt
        stablize_time = 1.0
        # linear interpolation
        pos_start = self.get_drone_state()[0]
        pos_end = pos
        N = int(timelimit / dt)
        for i in range(N):
            if i > int(stablize_time / dt):
                pos_tar = pos_end
                vel_tar = np.zeros(3)
            else:
                pos_tar = pos_start + (pos_end - pos_start) * i / N
                vel_tar = (pos_end - pos_start) / timelimit
            state_real_replaced = self.state_real.replace(pos_tar=pos_tar, vel_tar=vel_tar, acc_tar = jnp.zeros(3))
            action, self.base_control_params, _ = self.base_controller(None, state_real_replaced, self.env_params, None, self.base_control_params, None)
            step_rng, self.rng = jax.random.split(self.rng)
            next_state_dict = self.step(step_rng, action)
        return next_state_dict
    
    def reset(self):
        reset_rng, self.rng = jax.random.split(self.rng)
        # reset controller
        self.base_control_params = self.default_control_params
        self.control_params = self.default_control_params
        # fly to initial point
        next_state_dict = self.goto(jnp.zeros(3), timelimit=3.0)
        self.state_real = self.get_real_state()
        # reset simulator
        obs_sim, info_sim, self.state_sim = self.env.reset(reset_rng, self.env_params)
        return next_state_dict
    
    def step(self, rng:chex.PRNGKey, actions: jnp.ndarray):
        # step simulator state
        obs_sim, self.state_sim, reward_sim, done_sim, info_sim = self.env.step(actions, self.state_sim) 
        
        # step real-world state
        thrust_tar = (actions[0]+1.0)/2.0*self.env.default_params.max_thrust
        omega_tar = actions[1:4] * self.env.default_params.max_omega
        self.set_attirate(omega_tar, thrust_tar)
        self.timeHelper.sleepForRate(self.rate)
        # update real-world state
        pos, quat = self.get_drone_state()
        self.pos_hist = jnp.concatenate([self.pos_hist[1:], pos.reshape(1,3)], axis=0)
        self.quat_hist = jnp.concatenate([self.quat_hist[1:], quat.reshape(1,4)], axis=0)
        self.state_real = self.get_real_state()
        obs_real = self.env.get_obs(self.state_real)
        reward_real = self.env.reward_fn(self.state_real, self.env_params)
        done_real = self.env.is_terminal(self.state_real, self.env_params)
        info_real = self.env.get_info(self.state_real, self.env_params)

        return {
            'real': [obs_real, self.state_real, reward_real, done_real, info_real],
            'sim': [obs_sim, self.state_sim, reward_sim, done_sim, info_sim],
        }

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

    rclpy.shutdown()

if __name__ == "__main__":
    main()