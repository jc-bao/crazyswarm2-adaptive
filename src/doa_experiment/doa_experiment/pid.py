import numpy as np
from dataclasses import dataclass


@dataclass
class PIDParams:
    Kp: float = 6.0
    Kd: float = 4.0
    Ki: float = 0.0
    Kp_att: float = 3.0

    integral: np.ndarray = np.array([0.0, 0.0, 0.0])
    quat_desired: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])


class PIDController:
    def __init__(self, control_params: PIDParams, m, g, max_thrust, max_omega) -> None:
        self.m = m
        self.g = g
        self.max_thrust = max_thrust
        self.max_omega = max_omega

    def __call__(
        self,
        state,
        control_params: PIDParams,
    ) -> np.ndarray:

        ep = state.pos - state.pos_tar
        ev = state.vel - state.vel_tar
        err = ep + ev

        # position control
        Q = qtoQ(state.quat)
        f_d = self.m * (
            np.array([0.0, 0.0, self.g])
            - control_params.Kp * (state.pos - state.pos_tar)
            - control_params.Kd * (state.vel - state.vel_tar)
            - control_params.Ki * control_params.integral
            + state.acc_tar
        )
        thrust = (Q.T @ f_d)[2]
        thrust = np.clip(thrust, 0.0, self.max_thrust)

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
                np.array([(thrust / self.max_thrust) * 2.0 - 1.0]),
                (omega_d / self.max_omega),
            ]
        )

        # update control_params
        # integral = control_params.integral + (state.pos - state.pos_tar) * self.param.dt

        # control_params.integral = integral
        control_params.quat_desired = quat_desired

        return (
            err,
            action,
            control_params,
            {"ref_omega": omega_d, "ref_q": quat_desired, "thrust": thrust},
        )


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
