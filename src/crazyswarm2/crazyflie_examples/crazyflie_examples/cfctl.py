from crazyflie_py import Crazyswarm
import numpy as np
import tf2_ros
import transforms3d as tf3d
from std_msgs.msg import Float32MultiArray
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
import pickle
from copy import deepcopy
from line_profiler import LineProfiler

import jax
import chex
from jax import numpy as jnp

import quadjax
from quadjax.envs.quad3d_free import Quad3D, get_controller
from quadjax.train import ActorCritic, Compressor, Adaptor
from quadjax import controllers
from quadjax.dynamics import utils, geom
from quadjax.dynamics.dataclass import EnvParams3D, EnvState3D, Action3D
import time

def do_profile(follow=[]):
    def inner(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(func)
                for f in follow:
                    profiler.add_function(f)
                profiler.enable_by_count()
                return func(*args, **kwargs)
            finally:
                profiler.print_stats()
        return profiled_func
    return inner


class Crazyflie:

    def __init__(self, task = 'tracking', controller_name = 'pid', controller_params = None) -> None:
        # create jax environment for reference
        self.env = Quad3D(task=task, dynamics='bodyrate', obs_type='quad', lower_controller='base', enable_randomizer=False, disturb_type='none')
        # self.step_jit = jax.jit(self.env.step)
        # self.reset_jit = jax.jit(self.env.reset)
        # self.get_obs_jit = jax.jit(self.env.get_obs)
        # self.get_info_jit = jax.jit(self.env.get_info)
        # self.get_reward_jit = jax.jit(self.env.reward_fn)
        # self.is_terminal_jit = jax.jit(self.env.is_terminal)
        # self.sample_params_jit = jax.jit(self.env.sample_params)

        self.rng = jax.random.PRNGKey(0)
        rng_params, self.rng = jax.random.split(self.rng)
        self.env_params = self.env.sample_params(rng_params)
        # self.env_params.replace(dt=1/25.0)
        rng_state, self.rng = jax.random.split(self.rng)
        obs, info, self.state_sim = self.env.reset(rng_state)
        # deep copy flax.struct.dataclass 
        self.state_real = deepcopy(self.state_sim)

        # real-world parameters
        self.world_center = jnp.array([-0.5, -0.5, 1.5])
        self.xyz_min = jnp.array([-2.0, -3.0, -2.0])
        self.xyz_max = jnp.array([2.0, 2.0, 1.5])

        # base controller: PID
        self.base_controller, self.base_control_params = get_controller(self.env, 'pid', None)
        self.base_control_params = self.base_control_params.replace(Kp_att=5.0, Kp=18.0, Kd=4.0, Ki=0.0, Ki_att=1.0)
        # self.base_control_params = self.base_control_params.replace(Kp_att=5.0, Kp=6.0, Kd=4.0, Ki=3.0, Ki_att=1.0)
        # self.base_control_params = self.base_control_params.replace(Kp_att=5.0, Kp=6.0, Kd=4.0, Ki=1.2, Ki_att=1.0)
        # self.base_control_params = self.base_control_params.replace(Kp_att=6.0, Kp=2.0, Kd=2.0, Ki=0.6, Ki_att=0.2)
        # self.base_control_params = self.base_control_params.replace(Kp_att=10.0, Kp=6.0, Kd=4.0, Ki=3.0, Ki_att=0.06)
        self.default_base_control_params = deepcopy(self.base_control_params)
        # self.base_controller_jit = jax.jit(self.base_controller)
        # controller to test out
        self.controller, self.control_params = get_controller(self.env, controller_name, controller_params)
        self.default_control_params = deepcopy(self.control_params)
        # self.controller_jit = jax.jit(self.controller)

        # ROS related initialization
        self.pos = jnp.zeros(3)
        self.quat = jnp.array([0.0, 0.0, 0.0, 1.0])
        self.pos_kf = jnp.zeros(3)
        self.quat_kf = jnp.array([0.0, 0.0, 0.0, 1.0])
        self.pos_hist = jnp.zeros((self.env.default_params.adapt_horizon+3, 3), dtype=jnp.float32)
        self.quat_hist = jnp.zeros((self.env.default_params.adapt_horizon+3, 4), dtype=jnp.float32)
        self.action_hist = jnp.zeros((self.env.default_params.adapt_horizon+2, 4), dtype=jnp.float32)
        # Debug values
        self.rpm = np.zeros(4)
        self.omega = np.zeros(3)
        self.omega_hist = jnp.zeros((self.env.default_params.adapt_horizon+3, 3), dtype=jnp.float32)

        # crazyswarm related initialization
        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.cf = self.swarm.allcfs.crazyflies[0]
        # publisher
        rate = int(1.0 / self.env_params.dt)
        self.world_center_pub = self.swarm.allcfs.create_publisher(PoseStamped, 'world_center', rate)
        self.pos_sim_pub = self.swarm.allcfs.create_publisher(PoseStamped, 'pos_sim', rate)
        self.pos_real_pub = self.swarm.allcfs.create_publisher(PoseStamped, 'pos_real', rate)
        self.pos_tar_pub = self.swarm.allcfs.create_publisher(PoseStamped, 'pos_tar', rate)
        self.traj_pub = self.swarm.allcfs.create_publisher(Path, 'traj', 1)
        self.omega_pub = self.swarm.allcfs.create_publisher(Float32MultiArray, 'omega_diff', rate)
        # listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(buffer=self.tf_buffer, node=self.swarm.allcfs)
        self.rpm_listener = self.swarm.allcfs.create_subscription(Float32MultiArray, 'rpm', self.rpm_callback, 10)
        self.omega_listener = self.swarm.allcfs.create_subscription(Float32MultiArray, 'omega', self.omega_callback, 10)
        self.swarm.allcfs.create_subscription(PoseStamped, 'cf1/pose', self.state_callback_cf, rate)

        # ROS timer
        self.last_control_time = 0.0

        mmddhhmmss = time.strftime("%m%d%H%M%S", time.localtime())
        self.log_path = f"/home/pcy/Research/code/crazyswarm2-adaptive/cflog/cfctl_{mmddhhmmss}.txt"
        self.log = []
        
        no_action_param = self.swarm.allcfs.declare_parameter('no_action', False)
        self.no_action = no_action_param.get_parameter_value().bool_value
        if self.no_action:
            self.swarm.allcfs.get_logger().warn("no_action is set to True, no action will be sent to the drone")

    # def state_callback_cf_tf(self, data):
    #     pos = data.pose.position
    #     quat = data.pose.orientation
    #     self.pos = np.array([pos.x, pos.y, pos.z])
    #     self.quat = np.array([quat.x, quat.y, quat.z, quat.w])

    def rpm_callback(self, data):
        self.rpm = np.array(data.data)

    def omega_callback(self, data):
        self.omega = np.array(data.data)

    def state_callback_cf(self, data):
        pos = data.pose.position
        quat = data.pose.orientation
        self.pos_kf = np.array([pos.x, pos.y, pos.z])
        self.quat_kf = np.array([quat.x, quat.y, quat.z, quat.w])


    def get_real_state(self):
        dt = self.env.default_params.dt

        vel_hist = jnp.diff(self.pos_hist, axis=0) / dt
        # clip vel to -2 to 2
        vel_hist = jnp.clip(vel_hist, -2.0, 2.0)

        # calculate velocity with low-pass filter
        vel = 0.5 * vel_hist[-1] + 0.5 * vel_hist[-2]

        # update vel_hist
        vel_hist = jnp.concatenate([vel_hist[1:], vel.reshape(1,3)], axis=0)
        
        # dquat_hist = jnp.diff(self.quat_hist, axis=0) # NOTE diff here will make the length of dquat_hist 1 less than the others
        # quat_hist_conj = jnp.concatenate([-self.quat_hist[:, :-1], self.quat_hist[:, -1:]], axis=-1)
        # omega_hist = 2 * jax.vmap(geom.multiple_quat)(quat_hist_conj[:-1], dquat_hist/dt)[:, :-1]
        # DEBUG
        omega_hist = self.omega_hist

        action = self.action_hist[-1]
        last_thrust = (action[0]+1.0)/2.0*self.env.default_params.max_thrust
        last_torque = action[1:4] * self.env.default_params.max_torque

        return EnvState3D(
            # drone
            pos = self.pos_hist[-1], 
            vel = vel, 
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
            last_thrust = last_thrust,
            last_torque = last_torque,
            # step
            time = self.state_sim.time,
            # disturbance
            f_disturb = jnp.zeros(3),
            # trajectory information for adaptation
            vel_hist = vel_hist, 
            omega_hist = omega_hist,
            action_hist = self.action_hist,
            # control parameters
            control_params = None, # NOTE: disable control_params for now
        )

    def get_drone_state(self):
        # trans_mocap = self.tf_buffer.lookup_transform('world', 'cf1', rclpy.time.Time())
        # pos = trans_mocap.transform.translation
        # pos = jnp.array([pos.x, pos.y, pos.z])
        # quat = trans_mocap.transform.rotation
        # quat = jnp.array([quat.x, quat.y, quat.z, quat.w])
        
        pos = self.pos_kf
        quat = self.quat_kf
        # get timestamp
        # return jnp.array([pos.x, pos.y, pos.z]) - self.world_center, jnp.array([quat.x, quat.y, quat.z, quat.w])
        if self.log:
            self.log[-1]['pos_kf'] = pos
            self.log[-1]['quat_kf'] = quat

        return jnp.array(pos - self.world_center), jnp.array(quat)

    def set_attirate(self, omega_target, thrust_target):
        # convert to degree
        omega_target = np.array(omega_target, dtype=np.float64) # NOTE: make sure the type is float64
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
            # if self.state_real.pos[2] < (-self.world_center[2]+0.1):
            #     pos_tar_real = jnp.array([0.0, 0.0, -self.world_center[2]+0.15])
            #     vel_tar_real = jnp.zeros(3)
            print(f"goto: {i}/{N}, pos_tar_real: {pos_tar_real}, pos_tar_sim: {pos_tar_sim}, pos_start_real: {pos_start_real}, pos_start_sim: {pos_start_sim}, pos_end: {pos_end}")
            state_real_replaced = self.state_real.replace(pos_tar=pos_tar_real, vel_tar=vel_tar_real, acc_tar = jnp.zeros(3))
            
            action, _, info = jax.block_until_ready(self.base_controller(None, state_real_replaced, self.env_params, None, self.base_control_params, None))
            # convert info to dict
            # info = {k: np.array(v) for k, v in info}
            info['ros_time'] = self.timeHelper.time()
            info['sys_time'] = time.time()
            self.log.append(info)

            state_sim_replaced = self.state_sim.replace(pos_tar=pos_tar_sim, vel_tar=vel_tar_sim, acc_tar = jnp.zeros(3))
            action_sim, _, _ = jax.block_until_ready(self.base_controller(None, state_sim_replaced, self.env_params, None, self.base_control_params, None))

            next_state_dict = self.step(action, action_sim, self.no_action)

            self.state_real = self.state_real.replace(time=0)
            self.state_sim = self.state_sim.replace(time=0)
        return next_state_dict
    
    def empty_control(self):
        pos_real = self.get_drone_state()[0]
        pos_sim = self.state_sim.pos
        for i in range(10):
            self.state_real = self.state_real.replace(pos_tar=pos_real, vel_tar=jnp.zeros(3), acc_tar = jnp.zeros(3))
            self.state_sim = self.state_sim.replace(pos_tar=pos_sim, vel_tar=jnp.zeros(3), acc_tar = jnp.zeros(3))
            action, _, info = jax.block_until_ready(self.base_controller(None, self.state_real, self.env_params, None, self.base_control_params, None))
            action_sim, _, _ = jax.block_until_ready(self.base_controller(None, self.state_sim, self.env_params, None, self.base_control_params, None))
            next_state_dict = self.step(action, action_sim, no_action=True)
            self.state_real = self.state_real.replace(time=0)
            self.state_sim = self.state_sim.replace(time=0)

        return next_state_dict


    def reset(self):
        # NOTE use this to estabilish connection
        print("empty spin started")
        for _ in range(10):
            self.set_attirate(np.zeros(3), 0.0)
            rclpy.spin_once(self.swarm.allcfs, timeout_sec=0)
        print("empty spin finished")

        print("empty control started")
        self.empty_control()
        print("empty control finished")


        reset_rng, self.rng = jax.random.split(self.rng)
        # reset controller
        self.base_control_params = self.default_base_control_params
        self.control_params = self.default_control_params
        # take off
        # pos = self.get_drone_state()[0]
        # pos_tar_0 = pos.at[2].set(0.0)
        # next_state_dict = self.goto(pos_tar_0, timelimit=10.0)
        # fly to initial point
        next_state_dict = self.goto(self.state_real.pos_tar, timelimit=10.0)
        self.state_real = self.get_real_state()
        # publish trajectory
        self.traj_pub.publish(self.get_path_msg(self.state_real.pos_traj))

        self.last_control_time = self.timeHelper.time()

        return next_state_dict
    
    # @do_profile()
    def step(self, action: jnp.ndarray, action_sim = None, no_action = False):
        # time_star = time.time()
        if action_sim is None:
            action_sim = action
        rng_step, self.rng = jax.random.split(self.rng)

        # step simulator state
        obs_sim, self.state_sim, reward_sim, done_sim, info_sim = jax.block_until_ready(self.env.step(rng_step, self.state_sim, action_sim, self.env_params)) 
        
        # step real-world state
        thrust_tar = (action[0]+1.0)/2.0*self.env.default_params.max_thrust
        omega_tar = action[1:4] * self.env.default_params.max_omega

        if not no_action:
            self.set_attirate(omega_tar, thrust_tar)
        else:
            self.set_attirate(np.zeros(3), 0.0)

        # wait for next time step
        dt = self.env.default_params.dt
        next_time = (int((self.timeHelper.time())/ dt) + 1) * dt
        delta_time = next_time - self.last_control_time
        if delta_time > (dt+1e-3):
            # warning if the time difference is too large
            print(f"WARNING: time difference is too large: {delta_time:.2f} s")
        self.last_control_time = next_time
        while (self.timeHelper.time() <= next_time):
            rclpy.spin_once(self.swarm.allcfs, timeout_sec=0.0)

        # update real-world state
        pos, quat = self.get_drone_state()
        self.pos_hist = jnp.concatenate([self.pos_hist[1:], pos.reshape(1,3)], axis=0)
        self.quat_hist = jnp.concatenate([self.quat_hist[1:], quat.reshape(1,4)], axis=0)

        # DEBUG
        self.omega_hist = jnp.concatenate([self.omega_hist[1:], self.omega.reshape(1,3)], axis=0)

        self.action_hist = jnp.concatenate([self.action_hist[1:], action.reshape(1,4)], axis=0)
        self.state_real = self.get_real_state()
        obs_real = self.env.get_obs(self.state_real, self.env_params)
        reward_real = self.env.reward_fn(self.state_real, self.env_params)
        done_real = self.env.is_terminal(self.state_real, self.env_params)
        info_real = self.env.get_info(self.state_real, self.env_params)

        self.pub_state()

        # time_end = time.time()
        # running_freq = 1.0 / (time_end - time_star)
        # print(f"running frequency: {running_freq:.2f} Hz", "time cost: ", time_end - time_star, "s")

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
        path_msg.header.stamp = rclpy.time.Time().to_msg()
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
        self.omega_pub.publish(Float32MultiArray(data=self.state_real.omega))

    def emergency(self, obs):
        self.cf.emergency()
        self.reset()
        raise ValueError
    
def main(repeat_times = 1, filename = ''):

    # env = Crazyflie(task='tracking', controller_name='pid', controller_params='real/ppo_params_RMA-DR')
    env = Crazyflie(task='tracking', controller_name='pid', controller_params='real/ppo_params_RMA-DR')

    try:

        print('reset...')
        next_state_dict = env.reset()
        obs_real, state_real, reward_real, done_real, info_real = next_state_dict['real']
        obs_sim, state_sim, reward_sim, done_sim, info_sim = next_state_dict['sim']

        env.cf.setParam('usd.logging', 1)
        print('main task...')
        rng = jax.random.PRNGKey(1)
        state_real_seq, obs_real_seq, reward_real_seq = [], [], []
        state_sim_seq, obs_sim_seq, reward_sim_seq = [], [], []
        control_seq = []
        ros_info_seq = []
        n_dones = 0

        # import pickle
        # with open("/home/pcy/Research/quadjax/results/real_state_seq_.pkl", "rb") as f:
        #     state_seq_real = pickle.load(f)
        # omega_tar = np.array([state['last_torque'] for state in state_seq_real]) / np.array([9e-3, 9e-3, 2e-3]) * np.array([10.0, 10.0, 3.0])

        # for i in range(200):
        '''
        while n_dones < repeat_times:
            state_real_seq.append(state_real)
            state_sim_seq.append(state_sim)

            rng, rng_act = jax.random.split(rng)
            action, env.control_params, control_info = jax.block_until_ready(env.controller(obs_real, state_real, env.env_params, rng_act, env.control_params, info_real))
            action_sim, _, _ = jax.block_until_ready(env.controller(obs_sim, state_sim, env.env_params, rng_act, env.control_params, info_sim))
            
            # action = np.array([1.0, omega_tar[i, 0]/10.0, omega_tar[i, 1]/10.0, omega_tar[i, 2]/3.0])
            # if t % 20 < 10:
            # action = np.array([1.0, 0.3*np.sin(t/20), 0.3*np.sin(t/15+np.pi/2), 0.3*np.sin(t/10+np.pi)])
            # action_sim = action
            # else:
            #     action = np.array([0.0, -0.3, -0.3, -0.3])
            #     action_sim = np.array([0.0, -0.3, -0.3, -0.3])

            # manually record certain control parameters into state_seq
            control_params = env.control_params
            if hasattr(control_params, 'd_hat') and hasattr(control_params, 'vel_hat'):
                control_seq.append({'d_hat': control_params.d_hat, 'vel_hat': control_params.vel_hat})
            if hasattr(control_params, 'a_hat'):
                control_seq.append({'a_hat': control_params.a_hat})
            if hasattr(control_params, 'quat_desired'):
                control_seq.append({'quat_desired': control_params.quat_desired})
            next_state_dict = env.step(action, action_sim)
            obs_real, state_real, reward_real, done_real, info_real = next_state_dict['real']
            obs_sim, state_sim, reward_sim, done_sim, info_sim = next_state_dict['sim']

            # DEBUG: should be done sim
            if done_real:
                control_params = env.controller.update_params(env.env_params, control_params)
                n_dones += 1

            reward_real_seq.append(reward_real)
            reward_sim_seq.append(reward_sim)
            obs_real_seq.append(obs_real)
            obs_sim_seq.append(obs_sim)
            ros_info_seq.append({'rpm': env.rpm})
        '''

        # print('landing...')
        env.set_attirate(np.zeros(3), 0.0)
        # env.goto(jnp.array([0.0, 0.0, -env.world_center[2]]), timelimit=5.0)

        print('plotting...')
        # convert state into dict
        state_real_seq_dict = [s.__dict__ for s in state_real_seq]
        state_sim_seq_dict = [s.__dict__ for s in state_sim_seq]
        if len(control_seq) > 0:
            # merge control_seq into state_seq with dict
            for i in range(len(state_real_seq)):
                state_real_seq_dict[i] = {**state_real_seq_dict[i], **control_seq[i], **ros_info_seq[i]}
                state_sim_seq_dict[i] = {**state_sim_seq_dict[i], **control_seq[i]}
        with open(f"{quadjax.get_package_path()}/../results/real_state_seq_{filename}.pkl", "wb") as f:
            pickle.dump(state_real_seq_dict, f)
        with open(f"{quadjax.get_package_path()}/../results/sim_state_seq_{filename}.pkl", "wb") as f:
            pickle.dump(state_sim_seq_dict, f)
        utils.plot_states(state_real_seq_dict, obs_real_seq, reward_real_seq, env.env_params, 'real'+filename)
        utils.plot_states(state_sim_seq_dict, obs_sim_seq, reward_sim_seq, env.env_params, 'sim'+filename)

    except KeyboardInterrupt:
        pass

    finally:
        env.cf.setParam('usd.logging', 0)

        with open(env.log_path, "wb") as f:
            pickle.dump(env.log, f)
        print("log saved to", env.log_path)
        
        rclpy.shutdown()


if __name__ == "__main__":
    # with jax.disable_jit():
    main(repeat_times=-1)