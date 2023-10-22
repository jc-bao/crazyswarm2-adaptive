from crazyflie_py import Crazyswarm
import numpy as np
from icecream import ic
import tf2_ros
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped, Point, Quaternion
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
import os
from .pid_controller import PIDController, PIDParam, PIDState
from nav_msgs.msg import Path

def np2pos(pos:np.ndarray):
    return Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))

def np2quat(quat:np.ndarray):
    return Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))

class Crazyflie:

    def __init__(self) -> None:
        # set numpy print precision
        np.set_printoptions(precision=3, suppress=True)
        
        # parameters
        self.world_center = np.array([0.5, 0.0, 1.5])
        self.mass = 0.027
        self.g = 9.81
        self.command_timelimit = 10.0
        self.traj_timelimit = 120.0

        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.allcfs = self.swarm.allcfs
        self.cf = self.allcfs.crazyflies[0]
        self.cf_name = self.cf.prefix[1:]
        self.cf_id = 0

        self.cf_num = 1
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(buffer=self.tf_buffer, node=self.allcfs)

        # control parameters
        self.max_vel = 6.0
        self.rate = 50
        self.xyz_min = np.array([-2.0, -3.0, -2.0])
        self.xyz_max = np.array([2.0, 2.0, 1.5])
        
        self.pos_pid_param = PIDParam(m=self.mass, g=self.g, max_thrust=0.6, max_omega=np.array([1.0, 1.0, 1.0])*0.1, Kp=np.array([1.0, 1.0, 1.0])*2.0, Kd=np.array([1.0, 1.0, 1.0])*2.0, Kp_att=np.array([1.0, 1.0, 1.0])*0.05)
        
        self.pos_pid = PIDController(self.pos_pid_param)

        # initialize parameters
        self.step_cnt = 0
        self.xyz_drone_kf = np.zeros(3)
        self.xyz_drone_tf = np.zeros(3)
        self.quat_drones_kf = np.array([0.0, 0.0, 0.0, 1.0])
        self.quat_drones_tf = np.array([0.0, 0.0, 0.0, 1.0])
        
        self.traj = Path()
        self.xyz_drone = np.zeros(3, dtype=np.float32)
        self.xyz_drone_target = np.zeros(3, dtype=np.float32)
        self.last_xyz_drone_target = np.zeros(3, dtype=np.float32)
        self.vxyz_drone = np.zeros(3)
        self.vxyz_drone_target = np.zeros(3)
        self.last_xyz_drone = np.zeros(3)
        self.quat_drones = np.array([0.0, 0.0, 0.0, 1.0])
        self.omega_drone = np.zeros(3)
        self.last_quat_drone = np.array([0.0, 0.0, 0.0, 1.0])
        self.quat_target = np.array([0.0, 0.0, 0.0, 1.0])
        self.omega_target = np.zeros(3)

        # ROS related initialization
        self.allcfs.create_subscription(PoseStamped, f'{self.cf.prefix}/pose', self.state_callback_cf, self.rate)

        # initialize publisher
        self.pose_pub = self.allcfs.create_publisher(PoseStamped, 'cmd_pose', self.rate)
        self.target_pub = self.allcfs.create_publisher(PoseStamped, 'target_pose', self.rate)
        self.traj_pub = self.allcfs.create_publisher(Path, 'traj', self.rate)

        self.msg = PoseStamped()
        self.msg.header.frame_id = 'world'

    def state_callback_cf(self, data):
        pos = data.pose.position
        quat = data.pose.orientation
        self.xyz_drone_kf = np.array([pos.x, pos.y, pos.z])
        self.quat_drones_kf = np.array([quat.x, quat.y, quat.z, quat.w])
        # print("state_callback_cf: ", self.xyz_drone_kf[cfid], self.quat_drones_kf[cfid])

    def get_drone_state(self):
        trans_mocap = self.tf_buffer.lookup_transform('world', self.cf_name, rclpy.time.Time())
        pos = trans_mocap.transform.translation
        quat = trans_mocap.transform.rotation
        xyz_drone = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
        quat_drone = np.array([quat.x, quat.y, quat.z, quat.w])
        self.xyz_drone_tf = xyz_drone.copy()
        self.quat_drones_tf = quat_drone.copy()
        return self.xyz_drone_tf, self.quat_drones_tf # self.quat_drones_kf

    def set_attirate(self, omega_target, thrust_target):
        # convert to degree
        omega_target = omega_target / np.pi * 180.0
        acc_z_target = thrust_target / self.mass

        omega = omega_target
        acc_z = acc_z_target

        # print("set_attirate: ", acc_z, omega)
        self.cf.cmdFullState(
            np.zeros(3),np.zeros(3),np.array([0,0,acc_z]), 0.0, omega)
    
    def reset(self):
        for _ in range(10):
            self.set_attirate(np.zeros(3), 0.0)
            self.timeHelper.sleepForRate(10.0)
        return self.soft_reset()

    def soft_reset(self):
        self.step_cnt = 0
        self.traj = self._generate_traj()
        self.traj.header.frame_id = "world"
        self.traj.header.stamp = rclpy.time.Time().to_msg()
        self.traj_pub.publish(self.traj)

        self.xyz_drone, self.quat_drones = self.get_drone_state()
        self.last_xyz_drone, self.last_quat_drone = self.xyz_drone.copy(), self.quat_drones.copy()
        self.timeHelper.sleepForRate(self.rate)

        info = self._get_info()

        return None, info

    def _get_info(self):
        return {
            'xyz_target': self.xyz_drone_target,
            'vxyz_target': self.vxyz_drone_target,
            'xyz_drone_kf': self.xyz_drone_kf,
            'quat_drone_kf': self.quat_drones_kf,
            'xyz_drone': self.xyz_drone,
            'vxyz_drone': self.vxyz_drone,
            'quat_drone': self.quat_drones,
            'omega_drone': self.omega_drone,
            'quat_target': self.quat_target,
            'omega_target': self.omega_target,
            'xyz_drone_target': self.xyz_drone_target,
        }

    def _pub_obs(self):
        # Create a PoseStamped message
        pose = PoseStamped()
        pose.header.frame_id = 'world'
        # set current time from rclpy
        pose.header.stamp = rclpy.time.Time().to_msg()
        pose.pose.position = np2pos(self.xyz_drone)
        # convert quat_drone to quaterion
        pose.pose.orientation = np2quat(self.quat_drones)
        # Publish the PoseStamped message to the /cf_tf topic
        self.pose_pub.publish(pose)

        pose.header.frame_id = 'world'
        pose.pose.position = np2pos(self.xyz_drone_target)
        # convert quat_target to quaterion
        pose.pose.orientation = np2quat(self.quat_target)
        # Publish the PoseStamped message to the /target topic
        self.target_pub.publish(pose)
        
    def step(self, action):
        thrust_target = np.zeros(self.cf_num)
        target_roll_rate = action[0]
        target_pitch_rate = action[1]
        target_yaw_rate = action[2]
        self.omega_target = np.array([target_roll_rate, target_pitch_rate, target_yaw_rate])
        thrust_target = action[3]
        self.set_attirate(self.omega_target, thrust_target)
        self.timeHelper.sleepForRate(self.rate)
        
        # observation
        self.xyz_drone, self.quat_drones = self.get_drone_state()
        self.vxyz_drone = (self.xyz_drone - self.last_xyz_drone) * self.rate
        # calculate angular velocity with quaternion self.quat_drones and self.last_quat_drone
        quat_deriv = (self.quat_drones - self.last_quat_drone) * self.rate
        self.omega_drone = 2 * quat_deriv[:-1] / (np.linalg.norm(self.quat_drones[:-1], axis=-1, keepdims=True)+1e-3)
        try:
            pos = self.traj.poses[self.step_cnt].pose.position
        except:
            pos = self.traj.poses[-1].pose.position
        self.last_xyz_drone_target = self.xyz_drone_target.copy()
        self.xyz_drone_target = np.array([pos.x, pos.y, pos.z])
        self.vxyz_drone_target = (self.xyz_drone_target - self.last_xyz_drone_target) * self.rate

        next_info = self._get_info()
        self.last_xyz_drone = self.xyz_drone.copy()
        self.last_quat_drone = self.quat_drones.copy()

        # publish observation
        self._pub_obs()

        # if np.any(self.xyz_drone > (self.xyz_max + self.world_center)) or np.any(self.xyz_drone < (self.xyz_min + self.world_center)):
        #     print(f'{self.xyz_drone} is out of bound')
        #     self.emergency(next_obs)

        # reward
        self.step_cnt += 1
        reward = None
        done = False

        return None, reward, done, next_info

    def emergency(self, obs):
        # stop
        # for _ in range(int(2.0*self.rate)):
        #     obs['xyz_target'] = self.last_xyz_drone.copy()
        #     obs['xyz_target'][2] = 0.5
        #     obs['vxyz_target'] = np.zeros(3)
        #     action = self.pid_controller(info)
        #     obs, reward, done, info = self.step(action)
        # move to origin
        # for _ in range(int(4.0*self.rate)):
        #     obs['xyz_target'] = self.world_center.copy()
        #     obs['xyz_target'][2] = 0.2
        #     obs['vxyz_target'] = np.zeros(3)
        #     action = self.pid_controller(info)
        #     obs, reward, done, info = self.step(action)
        self.reset()
        for cf in self.allcfs.crazyflies:
            cf.emergency()
        raise ValueError
    
    def pid_controller(self, info):
        xyz_targets = info['xyz_target']
        vxyz_targets = info['vxyz_target']
        xyz_drones = info['xyz_drone']
        vxyz_drones = info['vxyz_drone']
        quat_drones = info['quat_drone']   # [x, y, z, w]
        omega_drones = info['omega_drone']


        # print("rpy:", ", ".join(f"{value:.3f}" for value in tf3d.euler.quat2euler((quat_drones[0,3], quat_drones[0,0], quat_drones[0,1], quat_drones[0,2]))))

        xyz_target = xyz_targets
        vxyz_target = vxyz_targets
        xyz_drone = xyz_drones
        vxyz_drone = vxyz_drones
        quat_drone = quat_drones
        omega_drone = omega_drones

        state = PIDState(pos=xyz_drone, vel=vxyz_drone, quat=quat_drone, omega=omega_drone)
        target = PIDState(pos=xyz_target, vel=vxyz_target)
        
        thrust, roll_rate, pitch_rate, yaw_rate = self.pos_pid(state, target)
    
        act = np.array([roll_rate, pitch_rate, yaw_rate, thrust])

        return act


    def _generate_traj(self):

        def line_traj(start:np.ndarray, end:np.ndarray, t:float, frame_id:str="world"):
            #line trajectory
            step = int(t * self.rate)
            traj = Path()
            traj.header.frame_id = frame_id

            delta = end - start
            for i in range(step):
                pose = PoseStamped()
                pose.header.frame_id = frame_id
                pose.pose.position = np2pos(start + delta * i / step)
                traj.poses.append(pose)
            return traj
        

        #generate trajectory
        traj = Path()
        traj.header.frame_id = "world"
        base_w = 2 * np.pi / 4.0
        t = np.arange(0, int(5.0*self.rate)) / self.rate
        t = np.tile(t, (3,1)).transpose()
        traj_xyz = np.zeros((len(t), 3))
        traj_vxyz = np.zeros((len(t), 3))
        A = np.array([0.6, 0.6, 0.0])
        w = np.array([base_w, base_w, base_w*2.0])
        phase = np.array([np.pi/2,0.0,np.pi])
        traj_xyz = A * np.sin(t*w+phase)
        traj_vxyz = w * A * np.cos(t*w+phase)
        traj_xyz += self.world_center

        for j in range(traj_xyz.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = "world"
            pose.pose.position = np2pos(traj_xyz[j])
            traj.poses.append(pose)



        # takeoff trajectory
        target_point = self.xyz_drone_tf.copy()
        target_point[2] = 1.0
        current_point = self.xyz_drone_tf.copy()
        takeoff_traj_poses = line_traj(current_point, target_point, 2.0).poses + line_traj(target_point, traj_xyz[0], 2.0).poses


        # landing trajectory
        target_point = self.xyz_drone_tf.copy()
        target_point[2] = 0.0
        current_point = traj_xyz[-1].copy()
        landing_traj_poses = line_traj(current_point, target_point, 3.0).poses


        traj.poses = takeoff_traj_poses + traj.poses + landing_traj_poses

        return traj


    
def main():

    cfctl = Crazyflie()

    # PPO controller
    # load PPO controller
    # loaded_agent = np.load('/home/pcy/Documents/crazyswarm/ros_ws/src/crazyswarm/scripts/results/ppo_track_robust.pt', map_location='cpu')
    # policy = loaded_agent['actor']
    # compressor = loaded_agent['compressor']

    print('reset...')
    obs, info = cfctl.reset()

    print('take off')
    while cfctl.step_cnt < len(cfctl.traj.poses):
    # while cfctl.step_cnt < 10:
        # PID controller
        # print(info)
        action = cfctl.pid_controller(info) * 1.0
        obs, reward, done, info = cfctl.step(action)

    # stop
    obs, reward, done, info = cfctl.step(np.array([0.0, 0.0, 0.0, 0.0]))


    rclpy.shutdown()

if __name__ == "__main__":
    main()
