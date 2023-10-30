from __future__ import annotations

from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from rclpy.time import Time
from ..sim_data_types import State, Action
# import ROS messages for publishing rpm
from std_msgs.msg import Float32MultiArray, Float32

import numpy as np
import rowan

class Backend:
    """Backend that uses newton-euler rigid-body dynamics implemented in numpy"""

    def __init__(self, node: Node, names: list[str], states: list[State]):
        self.node = node
        self.names = names
        self.clock_publisher = node.create_publisher(Clock, 'clock', 10)

        # debug publisher
        # publish rpm to the rpm topic with message type float array 
        self.rpm_publisher = node.create_publisher(Float32MultiArray, 'rpm', 10)
        # publish tau_u
        self.tau_u_publisher = node.create_publisher(Float32MultiArray, 'tau_u', 10)
        # publish alpha
        self.alpha_publisher = node.create_publisher(Float32MultiArray, 'alpha', 10)
        # publish f_u
        self.f_u_publisher = node.create_publisher(Float32, 'f_u', 10)
        # publish omega
        self.omega_publisher = node.create_publisher(Float32MultiArray, 'omega', 10)

        self.t = 0
        self.dt = 0.0005

        self.uavs = []
        for state in states:
            uav = Quadrotor(state)
            self.uavs.append(uav)

    def time(self) -> float:
        return self.t

    def step(self, states_desired: list[State], actions: list[Action]) -> list[State]:
        # advance the time
        self.t += self.dt

        next_states = []

        for uav, action in zip(self.uavs, actions):
            
            normed_rpm = []
            for rpm in action.rpm:
                normed_rpm.append(rpm / (25e3-1))
                if normed_rpm[-1] > 0.99:
                    print(f"WARNING: RPM is too high, was {normed_rpm}")
            # publish normed_rpm
            rpm_message = Float32MultiArray()
            rpm_message.data = normed_rpm
            self.rpm_publisher.publish(rpm_message)
            self.tau_u_publisher.publish(Float32MultiArray(data=uav.tau_u))
            self.alpha_publisher.publish(Float32MultiArray(data=uav.alpha))
            self.f_u_publisher.publish(Float32(data=uav.f_u))
            self.omega_publisher.publish(Float32MultiArray(data=uav.state.omega))

            uav.step(action, self.dt)
            next_states.append(uav.state)

        # publish the current clock
        clock_message = Clock()
        clock_message.clock = Time(seconds=self.time()).to_msg()
        self.clock_publisher.publish(clock_message)

        return next_states

    def shutdown(self):
        pass


class Quadrotor:
    """Basic rigid body quadrotor model (no drag) using numpy and rowan"""

    def __init__(self, state):
        # parameters (Crazyflie 2.0 quadrotor)
        self.mass = 0.027 # kg
        # self.J = np.array([
        # 	[16.56,0.83,0.71],
        # 	[0.83,16.66,1.8],
        # 	[0.72,1.8,29.26]
        # 	]) * 1e-6  # kg m^2
        self.J = np.array([16.571710e-6, 16.655602e-6, 29.261652e-6])

        # Note: we assume here that our control is forces
        arm_length = 0.046 # m
        arm = 0.707106781 * arm_length
        t2t = 0.006 # thrust-to-torque ratio
        self.B0 = np.array([
            [1, 1, 1, 1],
            [-arm, -arm, arm, arm],
            [-arm, arm, arm, -arm],
            [-t2t, t2t, -t2t, t2t]
            ])
        self.g = 9.81 # not signed

        if self.J.shape == (3,3):
            self.inv_J = np.linalg.pinv(self.J) # full matrix -> pseudo inverse
        else:
            self.inv_J = 1 / self.J # diagonal matrix -> division

        self.state = state

        # debug state
        self.f_u = 0.0
        self.tau_u = np.zeros(3)
        self.alpha = np.zeros(3)

    def step(self, action, dt):

        # convert RPM -> Force
        def rpm_to_force(rpm):
            # polyfit using data and scripts from https://github.com/IMRCLab/crazyflie-system-id
            p = [2.55077341e-08, -4.92422570e-05, -1.51910248e-01]
            force_in_grams = np.polyval(p, rpm)
            force_in_newton = force_in_grams * 9.81 / 1000.0
            return np.maximum(force_in_newton, 0)

        force = rpm_to_force(action.rpm)

        # compute next state
        eta = np.dot(self.B0, force)
        f_u = np.array([0,0,eta[0]])
        tau_u = np.array([eta[1],eta[2],eta[3]])
        self.tau_u = tau_u
        self.f_u = f_u[2]

        # dynamics 
        # dot{p} = v 
        pos_next = self.state.pos + self.state.vel * dt
        # mv = mg + R f_u 
        vel_next = self.state.vel + (np.array([0,0,-self.g]) + rowan.rotate(self.state.quat,f_u) / self.mass) * dt

        # dot{R} = R S(w)
        # to integrate the dynamics, see
        # https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/, and
        # https://arxiv.org/pdf/1604.08139.pdf
        q_next = rowan.normalize(rowan.calculus.integrate(self.state.quat, self.state.omega, dt))

        # mJ = Jw x w + tau_u 
        self.alpha =(self.inv_J * (np.cross(self.J * self.state.omega, self.state.omega) + tau_u))
        omega_next = self.state.omega + self.alpha * dt

        self.state.pos = pos_next
        self.state.vel = vel_next
        self.state.quat = q_next
        self.state.omega = omega_next

        # if we fall below the ground, set velocities to 0
        if self.state.pos[2] < 0:
            self.state.pos[2] = 0
            self.state.vel = [0,0,0]
            self.state.omega = [0,0,0]
