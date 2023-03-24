from crazyflie_py import Crazyswarm
import numpy as np
from icecream import ic
import tf2_ros
import transforms3d as tf3d
import rclpy
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
import pandas as pd

class Crazyflie:

    def __init__(self) -> None:
        
        # parameters
        self.world_center = np.array([0.0, 0.0, 1.0])
        self.mocap_rpy_drift = np.array([0.0, 0.00, 0.00])
        self.mass = 0.03
        self.obj_mass = 0.00
        self.g = 9.81
        self.command_timelimit = 10.0
        self.traj_timelimit = 120.0

        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.allcfs = self.swarm.allcfs
        self.cf = self.allcfs.crazyflies[0]
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(buffer=self.tf_buffer, node=self.allcfs)

        # control parameters
        self.max_vel = 6.0
        self.rate = 10.0
        self.xyz_min = np.array([-2.0, -3.0, -1.5])
        self.xyz_max = np.array([2.0, 2.0, 1.5])

        # initialize parameters
        self.step_cnt = 0
        self.xyz_drone_kf = np.zeros(3)
        self.rpy_drone_kf = np.zeros(3)
        self.traj_xyz = np.zeros((int(self.command_timelimit*self.rate), 3))
        self.traj_vxyz = np.zeros((int(self.command_timelimit*self.rate), 3))
        self.xyz_drone = np.zeros(3)
        self.vxyz_drone = np.zeros(3)
        self.last_xyz_drone = np.zeros(3)
        self.rpy_drone = np.zeros(3)
        self.vrpy_drone = np.zeros(3)
        self.last_rpy_drone = np.zeros(3)
        self.rpy_target = np.zeros(3)
        self.vrpy_target = np.zeros(3)

        # ROS related initialization
        self.allcfs.create_subscription(PoseStamped, '/cf4/pose', self.state_callback, 10)
        self.pose_pub = self.allcfs.create_publisher(PoseStamped, '/cf_tf', 10)
        self.target_pub = self.allcfs.create_publisher(PoseStamped, '/target', 10)

    def state_callback(self, data):
        pos = data.pose.position
        quat = data.pose.orientation
        rpy = np.array(tf3d.euler.quat2euler(np.array([quat.w, quat.x, quat.y, quat.z])))
        self.xyz_drone_kf = np.array([pos.x, pos.y, pos.z])
        self.rpy_drone_kf = rpy - self.mocap_rpy_drift

    def thrust2cmd(self, thrust):
        a, b, c = 2.130295e-11*4.0, 1.032633e-6*4.0, 5.484560e-4*4.0
        return (-b + np.sqrt(b**2 - 4*a*(c-thrust)))/(2*a)

    def get_drone_state(self):
        trans_mocap = self.tf_buffer.lookup_transform('world', 'cf4', rclpy.time.Time())
        pos = trans_mocap.transform.translation
        xyz_drone = np.array([pos.x, pos.y, pos.z])
        return xyz_drone, self.rpy_drone_kf
    
    def reset(self):
        for _ in range(10):
            self.cf.cmdVelLegacy(0.0, 0.0, 0.0, 0)
            self.timeHelper.sleepForRate(10.0)
        return self.soft_reset()

    def soft_reset(self):
        self.step_cnt = 0
        self.traj_xyz, self.traj_vxyz = self._generate_traj()
        self.xyz_drone, self.rpy_drone = self.get_drone_state()
        self.last_xyz_drone, self.last_rpy_drone = self.xyz_drone.copy(), self.rpy_drone.copy()
        self.timeHelper.sleepForRate(self.rate)

        info = self._get_info()
        obs = self._get_obs()

        return obs, info

    def _get_info(self):
        return {
            'xyz_target': self.traj_xyz[self.step_cnt],
            'vxyz_target': self.traj_vxyz[self.step_cnt],
            'xyz_drone_kf': self.xyz_drone_kf,
            'rpy_drone_kf': self.rpy_drone_kf,
            'xyz_drone': self.xyz_drone,
            'vxyz_drone': self.vxyz_drone,
            'rpy_drone': self.rpy_drone,
            'vrpy_drone': self.vrpy_drone,
            'rpy_target': self.rpy_target,
            'vrpy_target': self.vrpy_target,
        }

    def _get_obs(self):
        xyz_drone = self.xyz_drone - self.world_center
        xyz_drone_normed = (xyz_drone - np.zeros(3)) / np.ones(3)
        xyz_obj = self.xyz_drone - np.array([0,0,0.2]) -  self.world_center
        xyz_obj_normed = (xyz_obj - np.zeros(3)) / np.ones(3)
        xyz_target = self.traj_xyz[self.step_cnt] - self.world_center
        xyz_target_normed = (xyz_target - np.zeros(3)) / (np.ones(3)*0.7)
        vxyz_drone = self.vxyz_drone
        vxyz_drone_normed = (vxyz_drone - np.zeros(3)) / (np.ones(3) * 2.0)
        vxyz_obj = self.vxyz_drone
        vxyz_obj_normed = (vxyz_obj - np.zeros(3)) / (np.ones(3) * 2.0)
        rpy_drone = self.rpy_drone
        rpy_drone_normed = (rpy_drone - np.zeros(3)) / np.array([np.pi/3, np.pi/3, 1.0])
        future_traj_x = self.traj_xyz[self.step_cnt:self.step_cnt+5].copy()
        future_traj_x = future_traj_x.flatten()
        future_traj_v = self.traj_vxyz[self.step_cnt:self.step_cnt+5].flatten()
        return np.concatenate(
            [
                xyz_drone_normed,
                xyz_obj_normed,
                xyz_target_normed,
                vxyz_drone_normed,
                vxyz_obj_normed,
                rpy_drone_normed,
                xyz_obj - xyz_target,
                vxyz_obj - self.traj_vxyz[self.step_cnt],
                future_traj_x,
                future_traj_v,
            ],
            axis=0,
        )

    def _pub_obs(self):
        # Create a PoseStamped message
        pose = PoseStamped()
        pose.header.frame_id = 'world'
        # set current time from rclpy
        pose.header.stamp = rclpy.time.Time().to_msg()
        pose.pose.position.x = self.xyz_drone[0]
        pose.pose.position.y = self.xyz_drone[1]
        pose.pose.position.z = self.xyz_drone[2]
        # convert rpy_drone to quaterion
        quat_drone = np.array(tf3d.euler.euler2quat(self.rpy_drone[0], self.rpy_drone[1], self.rpy_drone[2]))
        pose.pose.orientation.x = quat_drone[0]
        pose.pose.orientation.y = quat_drone[1]
        pose.pose.orientation.z = quat_drone[2]
        pose.pose.orientation.w = quat_drone[3]
        # Publish the PoseStamped message to the /cf_tf topic
        self.pose_pub.publish(pose)

        pose.header.frame_id = 'world'
        pose.pose.position.x = self.traj_xyz[self.step_cnt][0]
        pose.pose.position.y = self.traj_xyz[self.step_cnt][1]
        pose.pose.position.z = self.traj_xyz[self.step_cnt][2]
        # convert rpy_target to quaterion
        quat_target = np.array(tf3d.euler.euler2quat(self.rpy_target[0], self.rpy_target[1], self.rpy_target[2]))
        pose.pose.orientation.x = quat_target[0]
        pose.pose.orientation.y = quat_target[1]
        pose.pose.orientation.z = quat_target[2]
        pose.pose.orientation.w = quat_target[3]
        # Publish the PoseStamped message to the /target topic
        self.target_pub.publish(pose)
        
    def step(self, action):
        target_roll_rate = action[0]
        target_pitch_rate = action[1]
        target_yaw_rate = action[2]
        # self.rpy_target = np.array([target_roll, target_pitch, target_yaw_rate])
        self.vrpy_target = np.array([target_roll_rate, target_pitch_rate, target_yaw_rate])
        target_thrust = action[3]
        self.cf.cmdVelLegacy(roll=target_roll_rate/np.pi*180, pitch=target_pitch_rate/np.pi*180, yaw_rate=target_yaw_rate/np.pi*180, thrust=self.thrust2cmd(target_thrust))
        self.timeHelper.sleepForRate(self.rate)
        
        # observation
        self.xyz_drone, self.rpy_drone = self.get_drone_state()
        self.vxyz_drone = (self.xyz_drone - self.last_xyz_drone) * self.rate
        self.vrpy_drone = (self.rpy_drone - self.last_rpy_drone) * self.rate
        next_obs = self._get_obs()
        next_info = self._get_info()
        self.last_xyz_drone = self.xyz_drone.copy()
        self.last_rpy_drone = self.rpy_drone.copy()

        # publish observation
        self._pub_obs()

        if np.any(self.xyz_drone > (self.xyz_max + self.world_center)) or np.any(self.xyz_drone < (self.xyz_min + self.world_center)):
            print(f'{self.xyz_drone} is out of bound')
            self.emergency(next_obs)

        # reward
        self.step_cnt += 1
        reward = None
        done = False

        return next_obs, reward, done, next_info

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
        raise ValueError

    def pid_controller(self, info):
        xyz_target = info['xyz_target']
        vxyz_target = info['vxyz_target']
        xyz_drone = info['xyz_drone']
        vxyz_drone = info['vxyz_drone']
        rpy_drone = info['rpy_drone']    
        vrpy_drone = info['vrpy_drone']
        rpy_drone = (rpy_drone + np.pi) % (2*np.pi) - np.pi

        delta_xyz_target = np.clip(xyz_target - xyz_drone, -self.max_vel/self.rate, self.max_vel/self.rate)
        xyz_target = xyz_drone + delta_xyz_target

        gravity_drone = np.array([0.0, 0.0, -self.g * (self.mass+self.obj_mass)])
        force_target = - gravity_drone - (xyz_drone - xyz_target) * np.array([0.2, 0.2, 0.2]) - (vxyz_drone - vxyz_target) * np.array([0.15, 0.15, 0.15])
        rotmat_drone = np.array(tf3d.euler.euler2mat(rpy_drone[0], rpy_drone[1], rpy_drone[2]))
        total_force_drone_projected = (rotmat_drone@force_target)[2]
        thrust_pid = np.clip(total_force_drone_projected, 0.0, 0.6)
        ctl_roll_pid = np.clip(np.arctan2(-force_target[1], np.sqrt(force_target[0]**2 + force_target[2]**2)), -np.pi/6, np.pi/6)
        ctl_roll_rate_pid = (ctl_roll_pid - rpy_drone[0]) * 4.0 - vrpy_drone[0] * 0.05
        ctl_pitch_pid = np.clip(np.arctan2(force_target[0], force_target[2]), -np.pi/6, np.pi/6)
        ctl_pitch_rate_pid = - (ctl_pitch_pid - rpy_drone[1]) * 4.0 + vrpy_drone[1] * 0.05
        ctl_yaw_rate_pid =  + (rpy_drone[2]) * 6.0 + vrpy_drone[2] * 0.00
        return np.array([ctl_roll_pid, ctl_pitch_pid, ctl_yaw_rate_pid, thrust_pid])

    def _generate_traj(self):
        base_w = 2 * np.pi / 6.0
        t = np.arange(0, int(self.traj_timelimit*self.rate)) / self.rate
        t = np.tile(t, (3,1)).transpose()
        traj_xyz = np.zeros((len(t), 3))
        traj_vxyz = np.zeros((len(t), 3))
        for i in range(0,1,1):
            A = 0.5 * (np.random.rand(3) * 0.3 + 0.7) * (2.0**(-i))

            # DEBUG
            A = np.array([0.8, 0.0, 0.4])

            w = base_w*(2**i)

            # DEBUG
            w = np.array([base_w, base_w, base_w*2.0])

            phase = np.random.rand(3) * 2 * np.pi
            
            # DEBUG
            phase = np.array([0,0,0])

            traj_xyz += A * np.sin(t*w+phase)
            traj_vxyz += w * A * np.cos(t*w+phase)
        traj_xyz += self.world_center
        return traj_xyz, traj_vxyz

class Logger():
    def __init__(self) -> None:
        self.xyz_target = []
        self.rpy_target = []
        self.vrpy_target = []
        self.vxyz_target = []
        self.xyz_drone = []
        self.rpy_drone = []
        self.xyz_drone_kf = []
        self.rpy_drone_kf = []
        self.vxyz_drone = []
        self.vrpy_drone = []

    def log(self, obs):
        self.xyz_target.append(obs['xyz_target'])
        self.rpy_target.append(obs['rpy_target'])
        self.vrpy_target.append(obs['vrpy_target'])
        self.vxyz_target.append(obs['vxyz_target'])
        self.xyz_drone.append(obs['xyz_drone'])
        self.rpy_drone.append(obs['rpy_drone'])
        self.vxyz_drone.append(obs['vxyz_drone'])
        self.vrpy_drone.append(obs['vrpy_drone'])
        self.xyz_drone_kf.append(obs['xyz_drone_kf'])
        self.rpy_drone_kf.append(obs['rpy_drone_kf'])
    
    def plot(self):
        # convert to numpy array
        self.xyz_target = np.array(self.xyz_target)
        self.rpy_target = np.array(self.rpy_target)
        self.vrpy_target = np.array(self.vrpy_target)
        self.xyz_drone_kf = np.array(self.xyz_drone_kf)
        self.rpy_drone_kf = np.array(self.rpy_drone_kf)
        self.vxyz_target = np.array(self.vxyz_target)
        self.xyz_drone = np.array(self.xyz_drone)
        self.rpy_drone = np.array(self.rpy_drone)
        self.vxyz_drone = np.array(self.vxyz_drone)
        self.vrpy_drone = np.array(self.vrpy_drone)

        # plot
        # create 3*4 subplot
        fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20, 9))
        title_list = ['x','y','z']
        for i in range(3):
            ax = axs[i, 0]
            ax.plot(self.xyz_target[:, i], label='target', linestyle='--')
            ax.plot(self.xyz_drone[:, i], label='drone')
            ax.plot(self.xyz_drone_kf[:, i], label='drone_kf')
            ax.set_ylabel(title_list[i])
            ax.legend()
        title_list = ['roll', 'pitch', 'yaw']
        for i in range(3):
            ax = axs[i, 1]
            ax.plot(self.rpy_target[:, i], label='target', linestyle='--')
            ax.plot(self.rpy_drone[:, i], label='drone')
            ax.plot(self.rpy_drone_kf[:, i], label='drone_kf')
            ax.set_ylabel(title_list[i])
            ax.legend()
        title_list = ['vx', 'vy', 'vz']
        for i in range(3):
            ax = axs[i, 2]
            ax.plot(self.vxyz_target[:, i], label='target', linestyle='--')
            ax.plot(self.vxyz_drone[:, i], label='drone')
            ax.set_ylabel(title_list[i])
            ax.legend()
        title_list = ['vr', 'vp', 'vy']
        for i in range(3):
            ax = axs[i, 3]
            ax.plot(self.vrpy_drone[:, i], label='drone')
            ax.plot(self.vrpy_target[:, i], label='target', linestyle='--')
            ax.set_ylabel(title_list[i])
            ax.legend()

        # plt.show()
        print('mocap drift fix value: ', -self.rpy_drone.mean(axis=0))
        plt.savefig('results/plot.png')

        # save all values as a csv file with pandas
        df = pd.DataFrame({
            'x_target': self.xyz_target[:, 0],
            'y_target': self.xyz_target[:, 1],
            'z_target': self.xyz_target[:, 2],
            'roll_target': self.rpy_target[:, 0],
            'pitch_target': self.rpy_target[:, 1],
            'yaw_target': self.rpy_target[:, 2],
            'vx_target': self.vxyz_target[:, 0],
            'vy_target': self.vxyz_target[:, 1],
            'vz_target': self.vxyz_target[:, 2],
            'x_drone': self.xyz_drone[:, 0],
            'y_drone': self.xyz_drone[:, 1],
            'z_drone': self.xyz_drone[:, 2],
            'roll_drone': self.rpy_drone[:, 0],
            'pitch_drone': self.rpy_drone[:, 1],
            'yaw_drone': self.rpy_drone[:, 2],
            'vx_drone': self.vxyz_drone[:, 0],
            'vy_drone': self.vxyz_drone[:, 1],
            'vz_drone': self.vxyz_drone[:, 2],
            'vr_drone': self.vrpy_drone[:, 0],
            'vp_drone': self.vrpy_drone[:, 1],
            'vy_drone': self.vrpy_drone[:, 2],
        })
        df.to_csv('results/data.csv', index=False)
    
def main():

    cfctl = Crazyflie()

    logger = Logger()

    # PPO controller
    # load PPO controller
    # loaded_agent = torch.load('/home/pcy/Documents/crazyswarm/ros_ws/src/crazyswarm/scripts/results/ppo_track_robust.pt', map_location='cpu')
    # policy = loaded_agent['actor']
    # compressor = loaded_agent['compressor']

    print('reset...')
    obs, info = cfctl.reset()

    print('take off')
    target_point = cfctl.last_xyz_drone.copy()
    target_point[2] = 1.0
    for i in range(int(3.0 * cfctl.rate)):
        if i < int(1.0 * cfctl.rate):
            info['xyz_target'] = target_point + np.array([0.0, 0.0, 0.03*i])
            info['vxyz_target'] = np.zeros(3)
            info['vxyz_target'][2] = 0.03
        else:
            info['xyz_target'] = target_point
            info['vxyz_target'] = np.zeros(3)
        action = cfctl.pid_controller(info) * 1.0
        # logger.log(info)
        obs, reward, done, info = cfctl.step(action)

    target_point = cfctl.traj_xyz[0]
    print('go to center', target_point)
    for _ in range(int(4.0 * cfctl.rate)):
        info['xyz_target'] = target_point
        info['vxyz_target'] = np.zeros(3)
        action = cfctl.pid_controller(info) * 1.0
        # logger.log(info)
        obs, reward, done, info = cfctl.step(action)

    # ==== main function ===
    # print('set roll to np.pi/24')
    # total_time = 4.0
    # total_step = int(total_time * cfctl.rate)
    # for i in range(total_step):
    #     roll = info['rpy_drone'][0]
    #     pitch = info['rpy_drone'][1]
    #     xyz_drone = info['xyz_drone']
    #     vxyz_drone = info['vxyz_drone']
    #     ctl_pitch = np.pi/24 if (i%total_step < (total_step//4) or i%total_step >= 3*(total_step//4)) else -np.pi/24
    #     ctl_pitch_rate = (ctl_pitch - pitch) * 4.0 - info['vrpy_drone'][1] * 0.01
    #     ctl_roll_rate = - roll * 4.0 - info['vrpy_drone'][0] * 0.01
    #     thrust = (0.027*9.81 + 0.20 * (cfctl.world_center[2] - xyz_drone[2]) - 0.13 * vxyz_drone[2]) / np.cos(np.sqrt(pitch**2+roll**2))
    #     action = np.array([ctl_roll_rate, ctl_pitch_rate, (info['rpy_drone'][2] - cfctl.mocap_rpy_drift[2]) * 4.0 + info['vrpy_drone'][2] * 0.0, thrust])
    #     info['xyz_target'] = cfctl.world_center.copy()
    #     info['vxyz_target'] = np.zeros(3)
    #     logger.log(info)
    #     obs, reward, done, info = cfctl.step(action)
    print('main task')
    obs, info = cfctl.soft_reset()
    for _ in range(int(12.0 * cfctl.rate)):
        # PID controller
        action = cfctl.pid_controller(info) * 1.0

        # PPO controller
        # with torch.no_grad():
        #     action_ppo = policy(torch.tensor(obs).float(), compressor(torch.zeros(9).float()))
        # action_ppo = action_ppo.cpu().numpy().squeeze()
        # ctl_thrust = action_ppo[0] * 0.3 + 0.3
        # ctl_roll = action_ppo[1] * np.pi/6
        # ctl_pitch = action_ppo[2] * np.pi/6
        # ctl_yaw_rate_pid =  (info['rpy_drone'][2] - cfctl.mocap_rpy_drift[2]) * 4.0 + info['vrpy_drone'][2] * 0.00
        # action = np.array([ctl_roll, ctl_pitch, ctl_yaw_rate_pid, ctl_thrust])

        logger.log(info)
        obs, reward, done, info = cfctl.step(action)

    print('to world center...')
    for _ in range(int(3.0*cfctl.rate)):
        info['xyz_target'] = cfctl.world_center.copy()
        info['vxyz_target'] = np.zeros(3)
        action = cfctl.pid_controller(info)*1.0
        # logger.log(info)
        obs, reward, done, info = cfctl.step(action)

    print('landing...')
    target_point = cfctl.last_xyz_drone.copy()
    target_point[2] = 0.05
    for _ in range(int(2.0*cfctl.rate)):
        info['xyz_target'] = target_point
        info['vxyz_target'] = np.zeros(3)
        action = cfctl.pid_controller(info)*1.0
        # logger.log(info)
        obs, reward, done, info = cfctl.step(action)

    # obs, info = cfctl.reset()

    logger.plot()

    rclpy.shutdown()

if __name__ == "__main__":
    main()
