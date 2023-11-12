from dataclasses import dataclass
import numpy as np
from . import geom
import transforms3d as tf3d
import rclpy
from geometry_msgs.msg import Vector3Stamped, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from .util import np2vec3, np2point, np2quat

import time
import pickle

mmddhhmmss = time.strftime("%m%d%H%M%S", time.localtime())
log_path = f"/home/pcy/Research/code/crazyswarm2-adaptive/cflog/att_pid_{mmddhhmmss}.txt"


@dataclass
class PIDParam:
    m: float
    g: float
    max_thrust: float
    max_omega: np.ndarray

    Kp: np.ndarray
    Kd: np.ndarray
    Ki: np.ndarray
    Kp_att: np.ndarray
    Ki_att: np.ndarray

    dt: float


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
        
        self.angle_err_pub = self.create_publisher(Vector3Stamped, "angle_err", 10)
        self.target_pub = self.create_publisher(Odometry, "target", 10)
        self.state_pub = self.create_publisher(Odometry, "state", 10)
        
        self.params = params
        self.err_i = np.zeros(3)
        self.err_i_att = np.zeros(3)

        self.log_path = log_path
        self.log = {
            "pos_err": [],
            "vel_err": [],
            "err_i": [],
            "angle_err": [],
            "err_i_att": [],
            "time": [],
            "pos_cur": [],
            "vel_cur": [],
            "omega_cur": [],
            "ang_cur": [],
            "pos_tar": [],
            "vel_tar": [],
            "omega_tar": [],
            "ang_tar": []
        }

    def __call__(self, state: PIDState, target: PIDState) -> np.ndarray:
        # position control
        Q = geom.qtoQ(state.quat)
                
        f_d = self.params.m * (
            np.array([0.0, 0.0, self.params.g])
            - self.params.Kp * (state.pos - target.pos)
            - self.params.Kd * (state.vel - target.vel)
            - self.params.Ki * self.err_i
            + target.acc
        )

        self.err_i += (state.pos - target.pos) * self.params.dt

        thrust = (Q.T @ f_d)[2]
        thrust = np.clip(thrust, 0.0, self.params.max_thrust)
        # print("f_d", f_d, "thrust", thrust)

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
        omega_d = -self.params.Kp_att * angle_err - self.params.Ki_att * self.err_i_att
        omega_d = np.clip(omega_d, -self.params.max_omega, self.params.max_omega)

        self.err_i_att += angle_err * self.params.dt
        print("angle_err", self.params.Kp_att * angle_err, "omega_d", omega_d, "err_i_att", self.err_i_att * self.params.Ki_att)
        print("pos_err: ", self.params.Kp * (state.pos - target.pos), "vel_err: ", self.params.Kd * (state.vel - target.vel), "err_i: ", self.err_i * self.params.Ki)

        # log
        self.log["pos_err"].append(self.params.Kp * (state.pos - target.pos))
        self.log["vel_err"].append(self.params.Kd * (state.vel - target.vel))
        self.log["err_i"].append(self.err_i * self.params.Ki)
        self.log["angle_err"].append(self.params.Kp_att * angle_err)
        self.log["err_i_att"].append(self.err_i_att * self.params.Ki_att)
        self.log["time"].append(time.time())
        self.log["pos_cur"].append(state.pos)
        self.log["vel_cur"].append(state.vel)
        self.log["omega_cur"].append(state.omega)
        self.log["ang_cur"].append(tf3d.euler.mat2euler(Q))
        self.log["pos_tar"].append(target.pos)
        self.log["vel_tar"].append(target.vel)
        self.log["omega_tar"].append(target.omega)
        self.log["ang_tar"].append(tf3d.euler.mat2euler(R_d))


        msg = Vector3Stamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector = np2vec3(angle_err)
        self.angle_err_pub.publish(msg)
        
        msg = Odometry()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position = np2point(state.pos)
        msg.pose.pose.orientation = np2quat(state.quat)
        msg.twist.twist.linear = np2vec3(state.vel)
        msg.twist.twist.angular = np2vec3(state.omega)
        self.state_pub.publish(msg)
        
        msg = Odometry()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position = np2point(target.pos)
        msg.pose.pose.orientation = np2quat(target.quat)
        msg.twist.twist.linear = np2vec3(target.vel)
        msg.twist.twist.angular = np2vec3(omega_d)
        self.target_pub.publish(msg)

        # generate action
        return np.concatenate([np.array([thrust]), omega_d])


    def save_log(self):
        with open(self.log_path, "wb") as f:
            pickle.dump(self.log, f)
        print("log saved to", self.log_path)