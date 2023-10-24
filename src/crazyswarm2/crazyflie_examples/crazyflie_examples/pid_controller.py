from dataclasses import dataclass
import numpy as np
from . import geom
import transforms3d as tf3d
import rclpy
from geometry_msgs.msg import Vector3Stamped, Vector3
from std_msgs.msg import Header
from .util import np2vec3


@dataclass
class PIDParam:
    m: float
    g: float
    max_thrust: float
    max_omega: np.ndarray

    Kp: np.ndarray
    Kd: np.ndarray
    Kp_att: np.ndarray


@dataclass
class PIDState:
    pos: np.ndarray = np.zeros(3)
    vel: np.ndarray = np.zeros(3)
    acc: np.ndarray = np.zeros(3)
    quat: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])  # [x, y, z, w]
    omega: np.ndarray = np.zeros(3)

def print_arr(arr):
    return str(", ".join([f"{i/np.pi*180:.2f}" for i in arr]))

class PIDController(rclpy.node.Node):
    def __init__(self, params: PIDParam):
        super().__init__("pid_controller")
        self.attitude_desire_pub = self.create_publisher(Vector3Stamped, "attitude_desire", 10)
        self.attitude_pub = self.create_publisher(Vector3Stamped, "attitude", 10)
        self.angle_err_pub = self.create_publisher(Vector3Stamped, "angle_err", 10)
        self.omega_d_pub = self.create_publisher(Vector3Stamped, "omega_d", 10)
        self.omega_pub = self.create_publisher(Vector3Stamped, "omega", 10)
        
        self.params = params

    def __call__(self, state: PIDState, target: PIDState) -> np.ndarray:
        # position control
        Q = geom.qtoQ(state.quat)
                
        f_d = self.params.m * (
            np.array([0.0, 0.0, self.params.g])
            - self.params.Kp * (state.pos - target.pos)
            - self.params.Kd * (state.vel - target.vel)
            + target.acc
        )
        thrust = (Q.T @ f_d)[2]
        thrust = np.clip(thrust, 0.0, self.params.max_thrust)

        # attitude control
        z_d = f_d / np.linalg.norm(f_d)
        axis_angle = np.cross(np.array([0.0, 0.0, 1.0]), z_d)
        angle = np.linalg.norm(axis_angle)
        # when the rotation axis is zero, set it to [0.0, 0.0, 1.0] and set angle to 0.0
        small_angle = np.abs(angle) < 1e-4
        axis = np.where(small_angle, np.array([0.0, 0.0, 1.0]), axis_angle / angle)
        R_d = geom.axisangletoR(axis, angle)
        R_e = R_d.T @ Q
        angle_err = geom.vee(R_e - R_e.T)
        # print("angle_desire", print_arr(tf3d.euler.mat2euler(R_d)), "angle", print_arr(tf3d.euler.mat2euler(Q)), "angle_err", print_arr(angle_err))
        # generate desired angular velocity
        omega_d = -self.params.Kp_att * angle_err
        omega_d = np.clip(omega_d, -self.params.max_omega, self.params.max_omega)

        msg = Vector3Stamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector = Vector3()
        angle_desire = tf3d.euler.mat2euler(R_d)
        msg.vector = np2vec3(angle_desire)
        self.attitude_desire_pub.publish(msg)
        
        angle = tf3d.euler.mat2euler(Q)
        msg.vector = np2vec3(angle)
        self.attitude_pub.publish(msg)
        
        msg.vector = np2vec3(angle_err)
        self.angle_err_pub.publish(msg)
        
        msg.vector = np2vec3(omega_d)
        self.omega_d_pub.publish(msg)

        msg.vector = np2vec3(state.omega)
        self.omega_pub.publish(msg)

        # generate action
        return np.concatenate([np.array([thrust]), omega_d])
