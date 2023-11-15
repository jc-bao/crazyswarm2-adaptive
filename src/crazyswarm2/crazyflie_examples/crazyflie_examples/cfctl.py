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
import time


def generate_smooth_traj(init_pos: np.array, dt: float) -> np.ndarray:
    """
    generate a smooth trajectory with max_steps steps
    """
    # generate take off trajectory
    target_pos = np.array([0.0, 0.0, 0.0])
    t_takeoff = 3.0
    N_takeoff = int(t_takeoff / dt)
    pos_takeoff = np.linspace(init_pos, target_pos, N_takeoff)
    vel_takeoff = (target_pos - init_pos) / t_takeoff
    acc_takeoff = np.zeros(3)

    # stablize for 1 second
    t_stablize = 1.0
    N_stablize = int(t_stablize / dt)
    pos_stablize = np.ones((N_stablize, 3)) * target_pos
    vel_stablize = np.zeros(3)
    acc_stablize = np.zeros(3)

    # generate main task trajectory
    scale = 1.0
    T = 5
    w0 = 2 * np.pi / T
    w1 = w0 * 2

    t = np.arange(0, T, dt)
    pos_main = np.zeros((len(t), 3))
    vel_main = np.zeros((len(t), 3))
    acc_main = np.zeros((len(t), 3))

    # generate figure 8 trajectory
    pos_main[:, 1] = scale * np.sin(w0 * t)
    vel_main[:, 1] = scale * w0 * np.cos(w0 * t)
    acc_main[:, 1] = -scale * w0**2 * np.sin(w0 * t)
    pos_main[:, 2] = scale * np.sin(w1 * t)
    vel_main[:, 2] = scale * w1 * np.cos(w1 * t)
    acc_main[:, 2] = -scale * w1**2 * np.sin(w1 * t)

    # generate landing trajectory by inverse the takeoff trajectory
    pos_landing = pos_takeoff[::-1]
    vel_landing = -vel_takeoff[::-1]
    acc_landing = np.zeros(3)

    # concatenate all trajectories
    pos = np.concatenate(
        [pos_takeoff, pos_stablize, pos_main, pos_stablize, pos_landing], axis=0
    )
    vel = np.concatenate(
        [vel_takeoff, vel_stablize, vel_main, vel_stablize, vel_landing], axis=0
    )
    acc = np.concatenate(
        [acc_takeoff, acc_stablize, acc_main, acc_stablize, acc_landing], axis=0
    )

    return pos, vel, acc


def multiple_quat(quat1: np.ndarray, quat2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (x, y, z, w)."""
    v1 = quat1[..., :3]
    w1 = quat1[..., 3:]
    v2 = quat2[..., :3]
    w2 = quat2[..., 3:]
    w = w1 * w2 - np.sum(v1 * v2, axis=-1, keepdims=True)
    v = w1 * v1 + w2 * v2 + np.cross(v1, v2, axis=-1)
    return np.concatenate([v, w], axis=-1)


def hat(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def L(q: np.ndarray) -> np.ndarray:
    """
    L(q) = [sI + hat(v), v; -v^T, s]
    left multiplication matrix of a quaternion
    """
    s = q[3]
    v = q[:3]
    right = np.hstack((v, s)).reshape(-1, 1)
    left_up = s * np.eye(3) + hat(v)
    left_down = -v
    left = np.vstack((left_up, left_down))
    return np.hstack((left, right))


def vee(R: np.ndarray):
    return np.array([R[2, 1], R[0, 2], R[1, 0]])


def qtoQ(q: np.ndarray) -> np.ndarray:
    """
    covert a quaternion to a 3x3 rotation matrix
    """
    T = np.diag(np.array([-1, -1, -1, 1]))
    H = np.vstack((np.eye(3), np.zeros((1, 3))))
    Lq = L(q)
    return H.T @ T @ Lq @ T @ Lq @ H


def Qtoq(Q: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    q[3] = 0.5 * np.sqrt(1 + Q[0, 0] + Q[1, 1] + Q[2, 2])
    q[:3] = (
        0.5
        / np.sqrt(1 + Q[0, 0] + Q[1, 1] + Q[2, 2])
        * np.array([Q[2, 1] - Q[1, 2], Q[0, 2] - Q[2, 0], Q[1, 0] - Q[0, 1]])
    )
    return q


def axisangletoR(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return (
        np.eye(3)
        + np.sin(angle) * hat(axis)
        + (1 - np.cos(angle)) * hat(axis) @ hat(axis)
    )


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


class EnvState3D:
    # meta state variable for taut state
    pos: np.ndarray  # (x,y,z)
    vel: np.ndarray  # (x,y,z)
    quat: np.ndarray  # quaternion (x,y,z,w)
    omega: np.ndarray  # angular velocity (x,y,z)
    omega_tar: np.ndarray  # angular velocity (x,y,z)
    zeta: np.ndarray  # S^2 unit vector (x,y,z)
    zeta_dot: np.ndarray  # S^2 (x,y,z)
    # target trajectory
    pos_traj: np.ndarray
    vel_traj: np.ndarray
    acc_traj: np.ndarray
    pos_tar: np.ndarray
    vel_tar: np.ndarray
    acc_tar: np.ndarray
    # hook state
    pos_hook: np.ndarray
    vel_hook: np.ndarray
    # object state
    pos_obj: np.ndarray
    vel_obj: np.ndarray
    # rope state
    f_rope_norm: float
    f_rope: np.ndarray
    l_rope: float
    # other variables
    last_thrust: float
    last_torque: np.ndarray  # torque in the local frame
    time: int
    f_disturb: np.ndarray

    # trajectory information for adaptation
    vel_hist: np.ndarray
    omega_hist: np.ndarray
    action_hist: np.ndarray


class EnvParams3D:
    max_speed: float = 8.0
    max_torque: np.ndarray = np.array([9e-3, 9e-3, 2e-3])
    max_omega: np.ndarray = np.array([10.0, 10.0, 3.0])
    max_thrust: float = 0.8
    dt: float = 0.02
    g: float = 9.81  # gravity

    m: float = 0.027  # mass
    m_mean: float = 0.027  # mass
    m_std: float = 0.003  # mass

    I: np.ndarray = np.array(
        [[1.7e-5, 0.0, 0.00], [0.0, 1.7e-5, 0.0], [0.0, 0.0, 3.0e-5]]
    )  # moment of inertia
    I_diag_mean: np.ndarray = np.array([1.7e-5, 1.7e-5, 3.0e-5])  # moment of inertia
    I_diag_std: np.ndarray = np.array([0.2e-5, 0.2e-5, 0.3e-5])  # moment of inertia

    mo: float = 0.01  # mass of the object attached to the rod
    mo_mean: float = 0.01
    mo_std: float = 0.003

    l: float = 0.3  # length of the rod
    l_mean: float = 0.3
    l_std: float = 0.1

    hook_offset: np.ndarray = np.array([0.0, 0.0, -0.01])
    hook_offset_mean: np.ndarray = np.array([0.0, 0.0, -0.02])
    hook_offset_std: np.ndarray = np.array([0.01, 0.01, 0.01])

    action_scale: float = 1.0
    action_scale_mean: float = 1.0
    action_scale_std: float = 0.1

    # 1st order dynamics
    alpha_bodyrate: float = 0.5
    alpha_bodyrate_mean: float = 0.5
    alpha_bodyrate_std: float = 0.1

    max_steps_in_episode: int = 300
    rope_taut_therehold: float = 1e-4
    traj_obs_len: int = 5
    traj_obs_gap: int = 5

    # disturbance related parameters
    d_offset: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    disturb_period: int = 50
    disturb_scale: float = 0.2
    disturb_params: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # curriculum related parameters
    curri_params: float = 1.0

    # RMA related parameters
    adapt_horizon: int = 4

    # noise related parameters
    dyn_noise_scale: float = 0.05


class PIDParams:
    Kp: float = 6.0
    Kd: float = 4.0
    Ki: float = 3.0
    Kp_att: float = 0.004

    integral: np.ndarray = np.array([0.0, 0.0, 0.0])
    quat_desired: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])


class PIDController:
    def __init__(self, env_params: EnvParams3D, control_params: PIDParams) -> None:
        self.param = env_params

    def update_params(self, env_param, control_params):
        return control_params

    def __call__(
        self,
        obs,
        state,
        env_param: EnvParams3D,
        rng_act,
        control_params: PIDParams,
        info=None,
    ) -> np.ndarray:
        # position control
        Q = qtoQ(state.quat)
        f_d = self.param.m * (
            np.array([0.0, 0.0, self.param.g])
            - control_params.Kp * (state.pos - state.pos_tar)
            - control_params.Kd * (state.vel - state.vel_tar)
            - control_params.Ki * control_params.integral
            + state.acc_tar
        )
        thrust = (Q.T @ f_d)[2]
        thrust = np.clip(thrust, 0.0, self.param.max_thrust)

        # attitude control
        z_d = f_d / np.linalg.norm(f_d)
        axis_angle = np.cross(np.array([0.0, 0.0, 1.0]), z_d)
        angle = np.linalg.norm(axis_angle)
        small_angle = np.abs(angle) < 1e-4
        axis = np.where(small_angle, np.array([0.0, 0.0, 1.0]), axis_angle / angle)
        R_d = axisangletoR(axis, angle)
        quat_desired = Qtoq(R_d)
        R_e = R_d.T @ Q
        angle_err = vee(R_e - R_e.T)

        # generate desired angular velocity
        omega_d = -control_params.Kp_att * angle_err

        # generate action
        action = np.concatenate(
            [
                np.array([(thrust / self.param.max_thrust) * 2.0 - 1.0]),
                (omega_d / self.param.max_omega),
            ]
        )

        # update control_params
        integral = control_params.integral + (state.pos - state.pos_tar) * env_param.dt

        control_params.integral = integral
        control_params.quat_desired = quat_desired

        return action, control_params, None


class Crazyflie:
    def __init__(
        self,
        task="tracking",
        controller_name="pid",
        controller_params=None,
        enable_logging=True,
    ) -> None:
        # control parameters
        self.timestep = 0
        self.dt = 0.02
        self.adapt_horizon = 10

        # real-world parameters
        self.world_center = np.array([0.0, 0.0, 2.0])
        self.xyz_min = np.array([-3.0, -3.0, -3.0])
        self.xyz_max = np.array([3.0, 3.0, 2.0])

        # environment parameters
        self.env_params = EnvParams3D()

        # base controller: PID
        self.control_params = PIDParams()
        self.controller = PIDController(self.env_params, self.base_control_params)

        # ROS related initialization
        self.pos = np.zeros(3)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.pos_kf = np.zeros(3)
        self.quat_kf = np.array([0.0, 0.0, 0.0, 1.0])
        self.pos_hist = np.zeros((self.adapt_horizon + 3, 3), dtype=np.float32)
        self.quat_hist = np.zeros((self.adapt_horizon + 3, 4), dtype=np.float32)
        self.action_hist = np.zeros((self.adapt_horizon + 2, 4), dtype=np.float32)
        # Debug values
        self.rpm = np.zeros(4)
        self.omega = np.zeros(3)
        self.omega_hist = np.zeros((self.adapt_horizon + 3, 3), dtype=np.float32)

        # crazyswarm related initialization
        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.cf = self.swarm.allcfs.crazyflies[0]
        # publisher
        rate = int(1.0 / self.env_params.dt)
        self.pos_real_pub = self.swarm.allcfs.create_publisher(
            PoseStamped, "pos_real", rate
        )
        self.pos_tar_pub = self.swarm.allcfs.create_publisher(
            PoseStamped, "pos_tar", rate
        )
        self.traj_pub = self.swarm.allcfs.create_publisher(Path, "traj", 1)
        # listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            buffer=self.tf_buffer, node=self.swarm.allcfs
        )
        self.swarm.allcfs.create_subscription(
            PoseStamped, "cf1/pose", self.state_callback_cf, rate
        )

        # logging
        self.enable_logging = enable_logging
        if enable_logging:
            mmddhhmmss = time.strftime("%m%d%H%M%S", time.localtime())
            self.log_path = f"/home/pcy/Research/code/crazyswarm2-adaptive/cflog/cfctl_{mmddhhmmss}.txt"
            self.log = []

        # establish connection
        # NOTE use this to estabilish connection
        for _ in range(10):
            self.set_attirate(np.zeros(3), 0.0)
            rclpy.spin_once(self.swarm.allcfs, timeout_sec=0)

        # initialize state
        self.pos_traj, self.vel_traj, self.acc_traj = generate_smooth_traj(
            self.pos_kf, self.dt
        )
        self.state_real = self.get_real_state()
        # publish trajectory
        self.traj_pub.publish(self.get_path_msg(self.state_real.pos_traj))
        self.last_control_time = self.timeHelper.time()

    def state_callback_cf(self, data):
        """
        read state from crazyflie, filtered with kalman filter
        """
        pos = data.pose.position
        quat = data.pose.orientation
        self.pos_kf = np.array([pos.x, pos.y, pos.z])
        self.quat_kf = np.array([quat.x, quat.y, quat.z, quat.w])

    def get_real_state(self):
        """
        state wrapper for real-world state
        """
        dt = self.dt

        vel_hist = np.diff(self.pos_hist, axis=0) / dt
        vel_hist = np.clip(vel_hist, -2.0, 2.0)

        # calculate velocity with low-pass filter
        vel = 0.5 * vel_hist[-1] + 0.5 * vel_hist[-2]
        vel_hist = np.concatenate([vel_hist[1:], vel.reshape(1, 3)], axis=0)

        dquat_hist = np.diff(
            self.quat_hist, axis=0
        )  # NOTE diff here will make the length of dquat_hist 1 less than the others
        quat_hist_conj = np.concatenate(
            [-self.quat_hist[:, :-1], self.quat_hist[:, -1:]], axis=-1
        )
        omega_hist = 2 * multiple_quat(quat_hist_conj[:-1], dquat_hist / dt)[:, :-1]

        action = self.action_hist[-1]
        last_thrust = (action[0] + 1.0) / 2.0 * self.env_params.max_thrust
        last_torque = action[1:4] * self.env_params.max_torque

        return EnvState3D(
            # drone
            pos=self.pos_hist[-1],
            vel=vel,
            omega=omega_hist[-1],
            omega_tar=np.zeros(3),
            quat=self.quat_hist[-1],
            # obj
            pos_obj=np.zeros(3),
            vel_obj=np.zeros(3),
            # hook
            pos_hook=np.zeros(3),
            vel_hook=np.zeros(3),
            # rope
            l_rope=0.0,
            zeta=np.zeros(3),
            zeta_dot=np.zeros(3),
            f_rope=np.zeros(3),
            f_rope_norm=0.0,
            # trajectory
            pos_tar=self.pos_traj[self.timestep],
            vel_tar=self.vel_traj[self.timestep],
            acc_tar=self.acc_traj[self.timestep],
            pos_traj=self.pos_traj,
            vel_traj=self.vel_traj,
            acc_traj=self.acc_traj,
            # debug value
            last_thrust=last_thrust,
            last_torque=last_torque,
            # step
            time=self.timestep,
            # disturbance
            f_disturb=np.zeros(3),
            # trajectory information for adaptation
            vel_hist=vel_hist,
            omega_hist=omega_hist,
            action_hist=self.action_hist,
        )

    def get_drone_state(self):
        """
        get state information from ros topic
        """
        # trans_mocap = self.tf_buffer.lookup_transform('world', 'cf1', rclpy.time.Time())
        # pos = trans_mocap.transform.translation
        # pos = np.array([pos.x, pos.y, pos.z])
        # quat = trans_mocap.transform.rotation
        # quat = np.array([quat.x, quat.y, quat.z, quat.w])

        pos = self.pos_kf
        quat = self.quat_kf
        # get timestamp
        # return np.array([pos.x, pos.y, pos.z]) - self.world_center, np.array([quat.x, quat.y, quat.z, quat.w])
        if self.enable_logging:
            self.log[-1]["pos_kf"] = pos
            self.log[-1]["quat_kf"] = quat

        return np.array(pos - self.world_center), np.array(quat)

    def set_attirate(self, omega_target, thrust_target):
        """
        set attitude rate and thrust through crazyflie lib
        """
        # convert to degree
        omega_target = (
            np.array(omega_target, dtype=np.float64) / np.pi * 180.0
        )  # NOTE: make sure the type is float64
        acc_z_target = thrust_target / self.env_params.m
        self.cf.cmdFullState(
            np.zeros(3), np.zeros(3), np.array([0, 0, acc_z_target]), 0.0, omega_target
        )

    # @do_profile()
    def step(self, action: np.ndarray):
        # step real-world state
        action = np.clip(action, -1.0, 1.0)
        thrust_tar = (action[0] + 1.0) / 2.0 * self.env_params.max_thrust
        omega_tar = action[1:4] * self.env_params.max_omega
        self.set_attirate(omega_tar, thrust_tar)

        # wait for next time step
        next_time = (int((self.timeHelper.time()) / self.dt) + 1) * self.dt
        delta_time = next_time - self.last_control_time
        if delta_time > (self.dt + 1e-3):
            print(f"WARNING: time difference is too large: {delta_time:.2f} s")
        self.last_control_time = next_time
        while self.timeHelper.time() <= next_time:
            rclpy.spin_once(self.swarm.allcfs, timeout_sec=0.0)

        # update real-world state
        pos, quat = self.get_drone_state()
        self.pos_hist = np.concatenate([self.pos_hist[1:], pos.reshape(1, 3)], axis=0)
        self.quat_hist = np.concatenate(
            [self.quat_hist[1:], quat.reshape(1, 4)], axis=0
        )
        self.action_hist = np.concatenate(
            [self.action_hist[1:], action.reshape(1, 4)], axis=0
        )
        self.pub_state()  # publish state for debug purpose

        self.state_real = self.get_real_state()
        obs_real = np.zeros(1)
        reward_real = 0.0
        done_real = False
        info_real = {}

        return obs_real, self.state_real, reward_real, done_real, info_real

    def get_pose_msg(self, pos, quat):
        msg = PoseStamped()
        pos = np.array(pos + self.world_center, dtype=np.float64)
        quat = np.array(quat, dtype=np.float64)
        msg.header.frame_id = "world"
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
        path_msg.header.frame_id = "world"
        path_msg.header.stamp = rclpy.time.Time().to_msg()
        for pos in pos_traj:
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "world"
            pos = np.array(pos + self.world_center, dtype=np.float64)
            pose_msg.pose.position.x = pos[0]
            pose_msg.pose.position.y = pos[1]
            pose_msg.pose.position.z = pos[2]
            path_msg.poses.append(pose_msg)
        return path_msg

    def pub_state(self):
        self.pos_real_pub.publish(
            self.get_pose_msg(self.state_real.pos, self.state_real.quat)
        )
        self.pos_tar_pub.publish(
            self.get_pose_msg(self.state_real.pos_tar, np.array([0.0, 0.0, 0.0, 1, 0]))
        )

    def emergency(self, obs):
        self.cf.emergency()
        self.reset()
        raise ValueError


def main(enable_logging=True):
    env = Crazyflie(enable_logging=enable_logging)

    try:
        env.cf.setParam("usd.logging", 1)
        state_real = env.state_real
        for _ in range(env.pos_traj.shape[0]):
            action, env.control_params, control_info = env.controller(
                None, state_real, env.env_params, None, env.control_params, None
            )
            obs_real, state_real, reward_real, done_real, info_real = env.step(action)
            env.log.append(control_info)
    except KeyboardInterrupt:
        pass
    finally:
        env.cf.setParam("usd.logging", 0)
        with open(env.log_path, "wb") as f:
            pickle.dump(env.log, f)
        print("log saved to", env.log_path)
        rclpy.shutdown()


if __name__ == "__main__":
    main(enable_logging=True)
