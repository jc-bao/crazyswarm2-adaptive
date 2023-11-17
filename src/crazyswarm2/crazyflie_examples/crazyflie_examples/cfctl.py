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
from dataclasses import dataclass
from typing import Any, Tuple

import jax, chex
from jax import lax
from jax import numpy as jnp
from quadjax import controllers
from quadjax.envs.quad3d_free import (
    Quad3D,
    EnvState3D as EnvState3DJax,
    EnvParams3D as EnvParams3DJax,
    Action3D as Action3DJax,
)
from quadjax.dynamics import utils
from quadjax import dynamics as quad_dyn

class Quad3DLite:
    """
    simplified version of quad3d
    """

    def __init__(self) -> None:
        self.default_params = EnvParams3DJax(alpha_bodyrate=0.2, alpha_thrust=1.0)
        self.sim_dt = 0.02
        self.action_dim = 4
        self.step_fn, self.dynamics_fn = quad_dyn.get_free_dynamics_3d_bodyrate(
            disturb_type="none"
        )
        self.step_env = self.step_env_wocontroller
        self.step_env_wocontroller_gradient = self.step_env_wocontroller
        self.reward_fn = utils.tracking_realworld_reward_fn
        self.get_obs = lambda s, p: 0.0

    @partial(jax.jit, static_argnums=(0,))
    def step_env_wocontroller(
        self,
        key: chex.PRNGKey,
        state: EnvState3DJax,
        action: jnp.ndarray,
        params: EnvParams3DJax,
        deterministic: bool = True,
    ) -> Tuple[chex.Array, EnvState3DJax, float, bool, dict]:
        action = jnp.clip(action, -1.0, 1.0)
        thrust = (action[0] + 1.0) / 2.0 * params.max_thrust
        torque = action[1:] * params.max_torque
        env_action = Action3DJax(thrust=thrust, torque=torque)
        next_state = self.step_fn(params, state, env_action, key, self.sim_dt)
        reward = self.reward_fn(next_state)
        return 0.0, next_state, reward, False, {}


class CoVOController(controllers.MPPIZejiController):
    def __init__(
        self, env, control_params, N: int, H: int, lam: float, mode: str = "offline"
    ) -> None:
        self.env_params = env.default_params
        self.key = jax.random.PRNGKey(0)
        if mode == "online":
            expansion_mode = "mean"
        elif mode == "offline":
            expansion_mode = "pid"
        else:
            raise NotImplementedError
        super().__init__(env, control_params, N, H, lam, expansion_mode)
        self.last_action = jnp.zeros(4)

    def __call__(
        self,
        obs,
        state,
        env_params,
        rng_act: chex.PRNGKey,
        control_params: controllers.MPPIParams,
        info=None,
    ) -> jnp.ndarray:
        # convert state to jax
        state_dict = state.__dict__
        state_dict_jax = {}
        for k, v in state_dict.items():
            state_dict_jax[k] = jnp.array(v)
        state_dict_jax["control_params"] = control_params
        state_jax = EnvState3DJax(**state_dict_jax)
        # get env_params_jax
        env_params_jax = self.env_params
        # get key
        self.key, rng_act = jax.random.split(self.key)
        action, control_params, control_info = super().__call__(
            obs, state_jax, env_params_jax, rng_act, control_params, info
        )
        action = jnp.clip(action, -1.0, 1.0)

        action_thrust = action[:1]
        last_action_thrust = self.last_action[:1]
        action_thrust_max = last_action_thrust + 0.2
        action_thrust_min = last_action_thrust - 0.2
        action_thrust = jnp.clip(action_thrust, action_thrust_min, action_thrust_max)

        action_omega = action[1:]
        last_action_omega = self.last_action[1:]
        action_omega_max = last_action_omega + 0.2
        action_omega_min = last_action_omega - 0.2
        action_omega = jnp.clip(action_omega, action_omega_min, action_omega_max)

        action = jnp.concatenate([action_thrust, action_omega], axis=0)

        # action = 1.0 * action + 0.0 * self.last_action
        self.last_action = action
        return action, control_params, control_info


class MPPIController(controllers.MPPIController):
    def __init__(self, env, control_params, N: int, H: int, lam: float) -> None:
        self.env_params = EnvParams3DJax(
            alpha_bodyrate=0.2,
            max_omega=jnp.array([4.0, 4.0, 2.0]),
            max_thrust=0.8,
            alpha_thrust=1.0,
        )
        self.key = jax.random.PRNGKey(0)
        self.last_action = jnp.zeros(4)
        super().__init__(env, control_params, N, H, lam)

    def __call__(
        self,
        obs,
        state,
        env_params,
        rng_act: chex.PRNGKey,
        control_params: controllers.MPPIParams,
        info=None,
    ) -> jnp.ndarray:
        # convert state to jax
        state_dict = state.__dict__
        state_dict_jax = {}
        for k, v in state_dict.items():
            state_dict_jax[k] = jnp.array(v)
        state_dict_jax["control_params"] = control_params
        state_jax = EnvState3DJax(**state_dict_jax)
        # get env_params_jax
        env_params_jax = self.env_params
        # get key
        self.key, rng_act = jax.random.split(self.key)
        action, control_params, control_info = super().__call__(
            obs, state_jax, env_params_jax, rng_act, control_params, info
        )
        action = jnp.clip(action, -1.0, 1.0)

        action_thrust = action[:1]
        last_action_thrust = self.last_action[:1]
        action_thrust_max = last_action_thrust + 0.2
        action_thrust_min = last_action_thrust - 0.2
        action_thrust = jnp.clip(action_thrust, action_thrust_min, action_thrust_max)

        action_omega = action[1:]
        last_action_omega = self.last_action[1:]
        action_omega_max = last_action_omega + 0.2
        action_omega_min = last_action_omega - 0.2
        action_omega = jnp.clip(action_omega, action_omega_min, action_omega_max)

        action = jnp.concatenate([action_thrust, action_omega], axis=0)

        # action = 1.0 * action + 0.0 * self.last_action
        self.last_action = action
        return action, control_params, control_info


def get_mppi_controller():
    sigma = 0.5
    N = 8192
    H = 32
    lam = 5e-2

    env = Quad3DLite()
    m = 0.027
    g = 9.81
    max_thrust = 0.8

    thrust_hover = m * g
    thrust_hover_normed = (thrust_hover / max_thrust) * 2.0 - 1.0
    a_mean_per_step = jnp.array([thrust_hover_normed, 0.0, 0.0, 0.0])
    a_mean = jnp.tile(a_mean_per_step, (H, 1))

    a_cov_per_step = jnp.diag(jnp.array([sigma**2] * env.action_dim))
    a_cov = jnp.tile(a_cov_per_step, (H, 1, 1))
    control_params = controllers.MPPIParams(
        gamma_mean=1.0,
        gamma_sigma=0.0,
        discount=1.0,
        sample_sigma=sigma,
        a_mean=a_mean,
        a_cov=a_cov,
        obs_noise_scale=0.05, 
    )

    # a_cov = jnp.diag(jnp.ones(H*env.action_dim)*sigma**2)
    # control_params = controllers.MPPIZejiParams(
    #     gamma_mean=1.0,
    #     gamma_sigma=0.0,
    #     discount=1.0,
    #     sample_sigma=sigma,
    #     a_mean=a_mean,
    #     a_cov=a_cov,
    #     a_cov_offline=jnp.zeros((1000, 128, 128)), 
    # )
    controller = MPPIController(
        env=env, control_params=control_params, N=N, H=H, lam=lam
    )
    return controller, control_params


def generate_smooth_traj(init_pos: np.array, dt: float) -> np.ndarray:
    """
    generate a smooth trajectory with max_steps steps
    """
    # generate still trajectory
    pos_still = np.ones((100, 3)) * init_pos
    pos_still[..., 2] -= 0.1 # NOTE: make sure the drone stay on the ground
    vel_still = np.zeros_like(pos_still)
    acc_still = np.zeros_like(pos_still)

    # generate take off trajectory
    # target_pos = np.array([0.0, 0.0, 0.0])
    target_pos = init_pos + np.array([0.0, 0.0, 1.0])
    t_takeoff = 2.0
    N_takeoff = int(t_takeoff / dt)
    pos_takeoff = np.linspace(init_pos, target_pos, N_takeoff)
    vel_takeoff = np.ones_like(pos_takeoff) * (target_pos - init_pos) / t_takeoff
    acc_takeoff = np.zeros_like(pos_takeoff)

    # stablize for 2.0 second
    t_stablize = 2.0
    N_stablize = int(t_stablize / dt)
    pos_stablize = np.ones((N_stablize, 3)) * target_pos
    vel_stablize = np.zeros_like(pos_stablize)
    acc_stablize = np.zeros_like(pos_stablize)

    # generate main task trajectory
    scale = 0.5
    T = 10.0
    w0 = 2 * np.pi / T
    w1 = w0 * 2

    t = np.arange(0, T, dt)
    pos_main = np.zeros((len(t), 3))
    vel_main = np.zeros((len(t), 3))
    acc_main = np.zeros((len(t), 3))

    # generate figure 8 trajectory
    pos_main[:, 1] = 2 * scale * np.sin(w0 * t)
    vel_main[:, 1] = 2 * scale * w0 * np.cos(w0 * t)
    acc_main[:, 1] = -2 * scale * w0**2 * np.sin(w0 * t)
    pos_main[:, 2] = scale * np.sin(w1 * t)
    vel_main[:, 2] = scale * w1 * np.cos(w1 * t)
    acc_main[:, 2] = -scale * w1**2 * np.sin(w1 * t)

    # generate landing trajectory by inverse the takeoff trajectory
    pos_landing = pos_takeoff[::-1]
    vel_landing = -vel_takeoff[::-1]
    acc_landing = -acc_takeoff[::-1]

    # concatenate all trajectories
    pos = np.concatenate(
        [pos_still, pos_takeoff, pos_stablize, pos_main, pos_stablize, pos_landing], axis=0
    )
    vel = np.concatenate(
        [vel_still, vel_takeoff, vel_stablize, vel_main, vel_stablize, vel_landing], axis=0
    )
    acc = np.concatenate(
        [acc_still, acc_takeoff, acc_stablize, acc_main, acc_stablize, acc_landing], axis=0
    )
    # pos = np.concatenate([pos_still, pos_takeoff, pos_stablize, pos_stablize, pos_landing], axis=0)
    # vel = np.concatenate([vel_still, vel_takeoff, vel_stablize, vel_stablize, vel_landing], axis=0)
    # acc = np.concatenate([acc_still, acc_takeoff, acc_stablize, acc_stablize, acc_landing], axis=0)
    # pos = np.ones((250, 3)) * init_pos
    # vel = np.zeros_like(pos)
    # acc = np.zeros_like(pos)

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


@dataclass
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


@dataclass
class EnvParams3D:
    max_speed: float = 8.0
    max_torque: np.ndarray = np.array([9e-3, 9e-3, 2e-3])
    max_omega: np.ndarray = np.array([4.0, 4.0, 2.0])
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

    max_steps_in_episode: int = 1000
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


@dataclass
class PIDParams:
    Kp: float = 8.0
    Kd: float = 4.0
    Ki: float = 3.0
    Kp_att: float = 6.0

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

        return action, control_params, {}


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
        self.world_center = np.array([0.0, 0.0, 1.5])
        self.xyz_min = np.array([-3.0, -3.0, -3.0])
        self.xyz_max = np.array([3.0, 3.0, 2.0])

        # environment parameters
        self.env_params = EnvParams3D()

        # base controller: PID
        self.mppi_controller, self.mppi_control_params = get_mppi_controller()
        self.control_params = PIDParams()
        self.controller = PIDController(self.env_params, self.control_params)

        # ROS related initialization
        self.pos = np.zeros(3)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.pos_kf = np.zeros(3)
        self.quat_kf = np.array([0.0, 0.0, 0.0, 1.0])
        self.pos_hist = np.zeros((self.adapt_horizon + 3, 3), dtype=np.float32)
        self.quat_hist = np.zeros((self.adapt_horizon + 3, 4), dtype=np.float32)
        self.quat_hist[..., -1] = 1.0
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
        self.pos_traj, self.vel_traj, self.acc_traj = generate_smooth_traj(pos, self.dt)
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

        return np.array(pos - self.world_center), np.array(quat)

    def set_attirate(self, omega_target, thrust_target):
        """
        set attitude rate and thrust through crazyflie lib
        """
        # convert to degree
        # omega_target = (
        #     np.array(omega_target, dtype=np.float64) / np.pi * 180.0 * 0.01
        # )  # NOTE: make sure the type is float64
        omega_target = np.array(omega_target, dtype=np.float64)
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
        last_discrete_time = int(self.last_control_time / self.dt)
        discrete_time = int(self.timeHelper.time() / self.dt)
        if discrete_time > (last_discrete_time + 1):
            next_time = (discrete_time+1) * self.dt
        else:
            next_time = (last_discrete_time + 1) * self.dt
        delta_time = next_time - self.last_control_time
        # if delta_time > (self.dt + 1e-3):
        #     print(f"WARNING: time difference is too large: {delta_time:.2f} s")
        print(f'frequncy: {(1.0 / delta_time):.2f} Hz')
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
        # env.cf.setParam("usd.logging", 1)
        state_real = env.state_real

        # update controller parameters for CoVO controller only
        # state_dict = state_real.__dict__
        # state_dict_jax = {}
        # for k, v in state_dict.items():
        #     state_dict_jax[k] = jnp.array(v)
        # state_dict_jax["control_params"] = env.mppi_control_params
        # state_jax = EnvState3DJax(**state_dict_jax)
        # env.mppi_control_params = env.mppi_controller.reset(state_jax, env.mppi_controller.env_params, env.mppi_control_params, jax.random.PRNGKey(0))

        total_steps = env.pos_traj.shape[0] - 1
        for timestep in range(total_steps):
            (
                action_mppi,
                env.mppi_control_params,
                mppi_control_info,
            ) = env.mppi_controller(
                None, state_real, env.env_params, None, env.mppi_control_params, None
            )
            action_pid, env.control_params, control_info = env.controller(
                None, state_real, env.env_params, None, env.control_params, None
            )
            # add noise to PID to test system robustness
            # action_pid[0] += 0.3*((timestep % 2) * 2.0 - 1.0)
            # action_pid[1:] += 0.1*((timestep % 2) * 2.0 - 1.0)
            if timestep < 6 * 50:
                k = 0.001
            elif timestep < 16 * 50:
                k = 1.0
            else:
                k = 0.001
            action_applied = action_mppi * k + action_pid * (1 - k)
            if timestep < 10:
                # not control at the beginning to warm up the controller
                action_applied = np.array([-1.0, 0.0, 0.0, 0.0]) + action_applied * 1e-4
            obs_real, state_real, reward_real, done_real, info_real = env.step(
                action_applied
            )
            log_info = {
                "pos": state_real.pos,
                "vel": state_real.vel,
                "quat": state_real.quat,
                "omega": state_real.omega,
                "action_pid": action_pid,
                "action_mppi": action_mppi,
                "pos_tar": state_real.pos_tar,
                "action_applied": action_applied,
            }
            env.log.append(log_info)
        for _ in range(50):
            env.set_attirate(np.zeros(3), 0.0)
    except KeyboardInterrupt:
        pass
    finally:
        # env.cf.setParam("usd.logging", 0)
        with open(env.log_path, "wb") as f:
            pickle.dump(env.log, f)
        print("log saved to", env.log_path)
        rclpy.shutdown()


if __name__ == "__main__":
    main(enable_logging=True)
