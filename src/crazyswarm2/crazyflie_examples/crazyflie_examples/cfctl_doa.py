from crazyflie_py import Crazyswarm
import numpy as np
import tf2_ros
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import pickle
import time
from dataclasses import dataclass

from quadjax.envs.quad3d_free import Quad3D
# import quadjax


def generate_traj(init_pos: np.array, dt: float, mode: str = "0") -> np.ndarray:
    """
    generate a trajectory with max_steps steps
    """

    # generate take off trajectory
    target_pos = np.array([0.0, 0.0, 0.0])
    t_takeoff = 3.0
    N_takeoff = int(t_takeoff / dt)
    pos_takeoff = np.linspace(init_pos, target_pos, N_takeoff)
    vel_takeoff = np.ones_like(pos_takeoff) * (target_pos - init_pos) / t_takeoff
    acc_takeoff = np.zeros_like(pos_takeoff)

    # stablize for 1.0 second
    t_stablize = 1.0
    N_stablize = int(t_stablize / dt)
    pos_stablize = np.ones((N_stablize, 3)) * target_pos
    vel_stablize = np.zeros_like(pos_stablize)
    acc_stablize = np.zeros_like(pos_stablize)

    # generate test trajectory
    t_task = 4.0

    target_pos = np.array([0.0, 0.0, 0.0])
    pos_task = np.ones((int(t_task / dt), 3)) * target_pos
    vel_task = np.zeros_like(pos_task)
    acc_task = np.zeros_like(pos_task)

    scale = 0.5
    pos_task = pos_task * scale
    vel_task = vel_task * scale
    acc_task = acc_task * scale

    # generate landing trajectory by inverse the takeoff trajectory
    pos_landing = pos_takeoff[::-1]
    vel_landing = -vel_takeoff[::-1]
    acc_landing = -acc_takeoff[::-1]

    # concatenate all trajectories
    pos = np.concatenate(
        [
            pos_takeoff,
            pos_stablize,
            pos_task,
            pos_stablize,
            pos_landing,
        ],
        axis=0,
    )
    vel = np.concatenate(
        [
            vel_takeoff,
            vel_stablize,
            vel_task,
            vel_stablize,
            vel_landing,
        ],
        axis=0,
    )
    acc = np.concatenate(
        [
            acc_takeoff,
            acc_stablize,
            acc_task,
            acc_stablize,
            acc_landing,
        ],
        axis=0,
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


@dataclass
class EnvState3D:
    # meta state variable for taut state
    pos: np.ndarray  # (x,y,z)
    vel: np.ndarray  # (x,y,z)
    quat: np.ndarray  # quaternion (x,y,z,w)
    omega: np.ndarray  # angular velocity (x,y,z)
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


class PIDController:
    def __init__(self, env_params: EnvParams3D) -> None:
        self.param = env_params
        self.integral = np.array([0.0, 0.0, 0.0])

    def __call__(
        self,
        state,
        ref,
        ref_dot,
        ref_ddot
    ) -> np.ndarray:

        Kp = 8.0
        Kd = 4.0
        Ki = 0.0
        Kp_att = 6.0
        
        

        ep = state.pos - ref
        ev = state.vel - ref_dot
        err = ep + ev

        # position control
        Q = qtoQ(state.quat)
        f_d = self.param.m * (
            np.array([0.0, 0.0, self.param.g])
            - Kp * (state.pos - ref)
            - Kd * (state.vel - ref_dot)
            - Ki * self.integral
            + ref_ddot
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
        omega_d = -Kp_att * angle_err

        # generate action
        action = np.concatenate(
            [
                np.array([(thrust / self.param.max_thrust) * 2.0 - 1.0]),
                (omega_d / self.param.max_omega),
            ]
        )

        # update control_params
        self.integral = self.integral + (state.pos - ref) * self.param.dt        

        return (
            err,
            action,
            {"ref_omega": omega_d, "ref_q": quat_desired, "thrust": thrust},
        )


class Crazyflie:
    def __init__(
        self,
        enable_logging=True,
    ) -> None:
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
        self.controller = PIDController(self.env_params)

        # ROS related initialization
        self.pos_kf = np.zeros(3)
        self.quat_kf = np.array([0.0, 0.0, 0.0, 1.0])
        self.pos_hist = np.zeros((self.adapt_horizon + 3, 3), dtype=np.float32)
        self.quat_hist = np.zeros((self.adapt_horizon + 3, 4), dtype=np.float32)
        self.quat_hist[..., -1] = 1.0
        self.action_hist = np.zeros((self.adapt_horizon + 2, 4), dtype=np.float32)
        self.err = np.zeros(3)
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
            self.log_path = (
                "/home/pcy/Research/code/crazyswarm2-adaptive/cflog/cfctl.txt"
            )
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
            pos, self.dt, mode="0"
        )
        self.state_real = self.get_real_state()
        # publish trajectory
        self.traj_pub.publish(self.get_path_msg(self.pos_traj))
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

        return EnvState3D(
            # drone
            pos=self.pos_hist[-1],
            vel=vel,
            omega=omega_hist[-1],
            quat=self.quat_hist[-1],
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

    def step(self, action: np.ndarray):
        err, action_pid, control_info = self.controller(
            self.state_real, self.pos_traj[self.timestep], self.vel_traj[self.timestep], self.acc_traj[self.timestep]
        )
        action += action_pid

        self.err = err

        # step real-world state
        action = np.clip(action, -1.0, 1.0)
        thrust_tar = (action[0] + 1.0) / 2.0 * self.env_params.max_thrust
        omega_tar = action[1:4] * self.env_params.max_omega
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
            "ref_p": self.pos_traj[self.timestep],
            "ref_v": self.vel_traj[self.timestep],
            "ref_a": self.acc_traj[self.timestep],
            "d_est": np.zeros(3),  # TODO
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


def main(enable_logging=True):  # mode  = mppi covo-online covo-offline nn
    env = Crazyflie(enable_logging=enable_logging)

    try:
        env.cf.setParam("usd.logging", 0)
        state_real = env.state_real

        total_steps = env.pos_traj.shape[0] - 1
        for timestep in range(total_steps):

            if timestep == int((4.0 - 0.1) * 50):
                env.cf.setParam("usd.logging", 1)
            elif timestep == int((10.0 + 0.1) * 50):
                env.cf.setParam("usd.logging", 0)

            action_applied = np.array([0.0, 0.0, 0.0, 0.0])
            state_real, reward_real, done_real, info_real = env.step(action_applied)
            err = state_real.err
            log_info = info_real
            env.log.append(log_info)
        for _ in range(50):
            env.set_attirate(np.zeros(3), 0.0)
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
