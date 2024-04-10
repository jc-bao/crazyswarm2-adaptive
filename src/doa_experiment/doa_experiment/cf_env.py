import numpy as np
import tf2_ros
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from dataclasses import dataclass
import rclpy
import os
import pickle
import json

from crazyflie_py import Crazyswarm
from .trajectory import generate_traj
from .pid import PIDController, PIDParams


@dataclass
class EnvState3D:
    # meta state variable for taut state
    pos: np.ndarray  # (x,y,z)
    vel: np.ndarray  # (x,y,z)
    quat: np.ndarray  # quaternion (x,y,z,w)
    omega: np.ndarray  # angular velocity (x,y,z)
    # target trajectory
    pos_tar: np.ndarray
    vel_tar: np.ndarray
    acc_tar: np.ndarray
    # err
    err: np.ndarray
    # other variables
    time: int
    f_disturb: np.ndarray

    # trajectory information for adaptation
    vel_hist: np.ndarray
    omega_hist: np.ndarray
    action_hist: np.ndarray


@dataclass
class EnvParams3D:
    max_speed: float = 8.0
    max_torque: np.ndarray = np.array([9e-3, 9e-3, 2e-3])
    max_omega: np.ndarray = np.array([10.0, 10.0, 2.0])
    max_thrust: float = 0.8
    dt: float = 0.02
    g: float = 9.81  # gravity

    m: float = 0.0411  # mass

    I: np.ndarray = np.array(
        [[1.7e-5, 0.0, 0.00], [0.0, 1.7e-5, 0.0], [0.0, 0.0, 3.0e-5]]
    )  # moment of inertia

    # RMA related parameters
    adapt_horizon: int = 2


def multiple_quat(quat1: np.ndarray, quat2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (x, y, z, w)."""
    v1 = quat1[..., :3]
    w1 = quat1[..., 3:]
    v2 = quat2[..., :3]
    w2 = quat2[..., 3:]
    w = w1 * w2 - np.sum(v1 * v2, axis=-1, keepdims=True)
    v = w1 * v1 + w2 * v2 + np.cross(v1, v2, axis=-1)
    return np.concatenate([v, w], axis=-1)


class Crazyflie:
    def __init__(
        self,
        T_takeoff,
        T_hover,
        T_task,
        mode,
        log_folder,
    ) -> None:

        self.T_takeoff = T_takeoff
        self.T_hover = T_hover
        self.T_task = T_task

        # control parameters
        self.timestep = 0
        self.dt = 0.02
        self.adapt_horizon = 10

        # real-world parameters
        self.world_center = np.array([0.0, 0.0, 1.5])
        self.xyz_min = np.array([-3.0, -3.0, -3.0])
        self.xyz_max = np.array([3.0, 3.0, 2.0])

        # environment parameters
        self.env_params = EnvParams3D()

        # base controller: PID
        self.control_params = PIDParams()
        self.controller = PIDController(
            self.control_params,
            self.env_params.m,
            self.env_params.g,
            self.env_params.max_thrust,
            self.env_params.max_omega,
        )

        # ROS related initialization
        self.pos_kf = np.zeros(3)
        self.quat_kf = np.array([0.0, 0.0, 0.0, 1.0])
        self.pos_hist = np.zeros((self.adapt_horizon + 3, 3), dtype=np.float32)
        self.quat_hist = np.zeros((self.adapt_horizon + 3, 4), dtype=np.float32)
        self.quat_hist[..., -1] = 1.0
        self.action_hist = np.zeros((self.adapt_horizon + 2, 4), dtype=np.float32)
        self.err = np.zeros(3)
        # Debug values
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
        self.log_folder = log_folder
        self.log = []

        # establish connection
        # NOTE use this to estabilish connection
        for _ in range(50):
            self.set_attirate(np.zeros(3), 0.0)
            rclpy.spin_once(self.swarm.allcfs, timeout_sec=0)

        # initialize state
        assert not np.allclose(
            self.pos_kf, np.zeros(3)
        ), "Drone initial position not updated"
        pos, quat = self.get_drone_state()
        self.pos_hist[-1] = pos
        self.quat_hist[-1] = quat
        self.pos_traj, self.vel_traj, self.acc_traj = generate_traj(
            pos, self.dt, T_takeoff, T_hover, T_task, mode=mode
        )
        self.state_real = self.get_real_state()
        # publish trajectory
        self.traj_pub.publish(self.get_path_msg(self.pos_traj))
        self.last_control_time = self.timeHelper.time()

        self.cf.setParam("usd.logging", 1)

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

        return EnvState3D(
            # drone
            pos=self.pos_hist[-1],
            vel=vel,
            omega=omega_hist[-1],
            quat=self.quat_hist[-1],
            # trajectory
            pos_tar=self.pos_traj[self.timestep],
            vel_tar=self.vel_traj[self.timestep],
            acc_tar=self.acc_traj[self.timestep],
            # err
            err=self.err,
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

        pos = self.pos_kf
        quat = self.quat_kf

        return np.array(pos - self.world_center), np.array(quat)

    def set_attirate(self, omega_target, thrust_target):
        """
        set attitude rate and thrust through crazyflie lib
        """
        omega_target = np.array(omega_target, dtype=np.float64)
        acc_z_target = thrust_target / self.env_params.m
        self.cf.cmdFullState(
            np.zeros(3), np.zeros(3), np.array([0, 0, acc_z_target]), 0.0, omega_target
        )

    def step(self, action_nn: np.ndarray):
        self.state_real.acc_tar -= action_nn / self.env_params.m
        err, action_pid, self.control_params, control_info = self.controller(
            self.state_real, self.control_params
        )

        action = action_pid

        self.err = err

        # step real-world state
        action = np.clip(action, -1.0, 1.0)
        thrust_tar = (action[0] + 1.0) / 2.0 * self.env_params.max_thrust
        omega_tar = action[1:4] * self.env_params.max_omega

        hovering_thrust_calib = 1.0
        thrust_tar = thrust_tar * hovering_thrust_calib
        self.set_attirate(omega_tar, thrust_tar)

        # wait for next time step
        last_discrete_time = int(self.last_control_time / self.dt)
        discrete_time = int(self.timeHelper.time() / self.dt)
        if discrete_time > (last_discrete_time + 1):
            next_time = (discrete_time + 1) * self.dt
        else:
            next_time = (last_discrete_time + 1) * self.dt
        delta_time = next_time - self.last_control_time + 1e-6
        frequncy = 1.0 / delta_time
        if frequncy < 49:
            print(f"frequncy: {frequncy:.2f} Hz")
        self.last_control_time = next_time
        while self.timeHelper.time() <= next_time:
            rclpy.spin_once(self.swarm.allcfs, timeout_sec=0.0)

        # update real-world state
        self.timestep += 1
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
        reward_real = 0.0
        done_real = False
        info_real = {
            "t": self.timestep * self.dt,
            "p": self.state_real.pos,
            "v": self.state_real.vel,
            "q": self.state_real.quat,
            "w": self.state_real.omega,
            "err": err,
            "ref_p": self.state_real.pos_tar,
            "ref_v": self.state_real.vel_tar,
            "ref_a": self.state_real.acc_tar,
            "d_est": action_nn,
            "act": action,
        }
        info_real |= control_info

        return self.state_real, reward_real, done_real, info_real

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

    def dump_log(self):
        begin = int((self.T_takeoff + self.T_hover) / self.dt)
        log = self.log[begin:-begin]
        self.cf.setParam("usd.logging", 0)
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        log_path = os.path.join(self.log_folder, "log.pkl")
        with open(self.log_path, "wb") as f:
            pickle.dump(log, f)
        print("log saved to", self.log_path)
        metrics_path = os.path.join(self.log_folder, "metrics.json")
        metrics = eval_tracking_performance(
            np.array([log[i]["p"] for i in range(len(log))]),
            np.array([log[i]["ref_p"] for i in range(len(log))]),
        )
        print("metrics:", metrics)
        print("metrics saved to", metrics_path)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        return log, metrics


def eval_tracking_performance(actual, reference):
    # calculate tracking performance
    # actual: np.array, shape (n, 2)
    # reference: np.array, shape (n, 2)
    # return: float, tracking performance
    whole_ade = np.mean(np.linalg.norm(actual - reference, axis=1))
    last_quarter_ade = np.mean(
        np.linalg.norm(
            actual[-int(len(actual) / 4) :] - reference[-int(len(reference) / 4) :],
            axis=1,
        )
    )
    max_error = np.max(np.linalg.norm(actual - reference, axis=1))
    last_quarter_max_error = np.max(
        np.linalg.norm(
            actual[-int(len(actual) / 4) :] - reference[-int(len(reference) / 4) :],
            axis=1,
        )
    )
    mse = np.mean(np.linalg.norm(actual - reference, axis=1) ** 2)
    rmse = np.sqrt(mse)
    return {
        "whole_ade": whole_ade,
        "last_quarter_ade": last_quarter_ade,
        "max_error": max_error,
        "last_quarter_max_error": last_quarter_max_error,
        "mse": mse,
        "rmse": rmse,
    }
