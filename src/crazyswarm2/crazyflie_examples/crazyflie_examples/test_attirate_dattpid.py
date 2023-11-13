from crazyflie_py import Crazyswarm
import numpy as np
import tf2_ros
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose, Transform, Vector3Stamped
from tf2_geometry_msgs import do_transform_pose
from .pid_controller import PIDController, PIDParam, PIDState
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Header
from .util import np2point, np2quat, np2vec3, line_traj
from rosbag2_py import SequentialWriter
import time
from . import geom


class Crazyflie:

    def __init__(self) -> None:
        # set numpy print precision
        np.set_printoptions(precision=3, suppress=True)
        
        # parameters
        self.world_center = np.array([0.0, 0.0, 1.0])
        self.mass = 0.027
        self.g = 9.81
        self.command_timelimit = 10.0
        self.traj_timelimit = 120.0

        # initialize crazyflie
        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper
        self.allcfs = self.swarm.allcfs
        self.cf = self.allcfs.crazyflies[0]
        self.cf_name = self.cf.prefix[1:]
        self.cf_id = 0
        self.cf_num = 1
        
        # check if simulation
        self.is_sim = self.allcfs.get_parameter("use_sim_time").get_parameter_value().bool_value
        # check if bag_path is set
        bag_path_param = self.allcfs.declare_parameter('bag_path', '')
        self.bag_path = bag_path_param.get_parameter_value().string_value
        if not self.bag_path:
            self.allcfs.get_logger().warn("bag_path is not set, bag will not be recorded, set bag_path to record bag")
        else:
            pass
        
        # check if no action output
        no_action_param = self.allcfs.declare_parameter('no_action', False)
        self.no_action = no_action_param.get_parameter_value().bool_value
        if self.no_action:
            self.allcfs.get_logger().warn("no_action is set to True, no action will be output")

        # control parameters
        self.max_vel = 6.0
        self.rate = 50
        self.xyz_min = np.array([-2.0, -3.0, -2.0])
        self.xyz_max = np.array([2.0, 2.0, 1.5])
        
        self.pos_pid_param = PIDParam(
            m=self.mass, 
            g=self.g, max_thrust=10.0, 
            max_omega=np.array([1.0, 1.0, 1.0])*3.0, 
            Kp=np.array([1.0, 1.0, 1.0])*6.0, 
            Kd=np.array([1.0, 1.0, 1.0])*4.0, 
            Ki=np.array([1.0, 1.0, 1.0])*3.0,
            Kp_att=np.array([1.0, 1.0, 1.0])*10.0,
            # Kp_att=np.array([1.0, 1.0, 1.0])*0.040,
            Ki_att=np.array([1.0, 1.0, 1.0])*0.000,
            dt = 1.0/self.rate)
        
        self.pos_pid = PIDController(self.pos_pid_param)

        # initialize parameters
        self.step_cnt = 0
        
        self.traj = Path()
        self.drone_state = PIDState(
            pos=np.zeros(3), 
            vel=np.zeros(3), 
            quat=np.array([0.0, 0.0, 0.0, 1.0]), 
            omega=np.zeros(3))
        self.drone_target = PIDState(
            pos=np.zeros(3), 
            vel=np.zeros(3), 
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            omega=np.zeros(3))

        # ROS related initialization
        self.allcfs.create_subscription(Odometry, f'{self.cf.prefix}/odom', self.odom_callback_cf, 10)
        self.allcfs.create_subscription(PoseStamped, f'{self.cf.prefix}/pose', self.pose_callback_cf, 10)
        print(f"subscribe to {self.cf.prefix}/odom")

        # initialize publisher
        self.target_pub = self.allcfs.create_publisher(PoseStamped, 'target_pose', 10)
        self.traj_pub = self.allcfs.create_publisher(Path, 'traj', 10)
        self.omega_debug_pub = self.swarm.allcfs.create_publisher(Vector3Stamped, 'omega_debug', 10)
        
        # initialize tf
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(buffer=self.tf_buffer, node=self.allcfs)
        self.tf_publisher = tf2_ros.TransformBroadcaster(node=self.allcfs)
        self.static_tf_publisher = tf2_ros.StaticTransformBroadcaster(node=self.allcfs)
        static_transformStamped = TransformStamped(
            header=Header(frame_id="world"), 
            child_frame_id="map", 
            transform=Transform(translation=np2vec3(self.world_center), 
                                rotation=np2quat(np.array([0.0, 0.0, 0.0, 1.0]))))
        self.static_tf_publisher.sendTransform(static_transformStamped) 
       


    def enable_logging(self):  
        print(self.cf_name)                  
        self.cf.setParam('usd.logging', 1)

    def disable_logging(self):
        self.cf.setParam('usd.logging', 0)

    def pose_callback_cf(self, pose_msg:PoseStamped):
        pos = pose_msg.pose.position
        quat = pose_msg.pose.orientation
        
        pos = np.array([pos.x, pos.y, pos.z])
        quat = np.array([quat.x, quat.y, quat.z, quat.w])

        self.drone_state.vel = (pos - self.drone_state.pos) * self.rate
        self.drone_state.pos = pos

        quat_deriv = (quat - self.drone_state.quat) * self.rate
        quat_conj = np.array([-quat[0], -quat[1], -quat[2], quat[3]])
        omega_diff = 2 * geom.multiple_quat(quat_conj, quat_deriv)[:-1]
        self.drone_state.omega = omega_diff
        self.drone_state.quat = quat
        
        # self.allcfs.get_logger().info(f"pos: {self.drone_state.pos}, quat: {self.drone_state.quat}", throttle_duration_sec=1.0)   
            
    
    def odom_callback_cf(self, odom_msg:Odometry):
        pos = odom_msg.pose.pose.position
        quat = odom_msg.pose.pose.orientation
        vel = odom_msg.twist.twist.linear
        omega = odom_msg.twist.twist.angular
        
        quat_tmp = np.array([quat.x, quat.y, quat.z, quat.w])
        quat_deriv = (quat_tmp - self.drone_state.quat) * self.rate
        quat_conj = np.array([-quat.x, -quat.y, -quat.z, quat.w])
        omega_diff = 2 * geom.multiple_quat(quat_conj, quat_deriv)[:-1]
        
        self.drone_state.pos = np.array([pos.x, pos.y, pos.z])
        self.drone_state.quat = np.array([quat.x, quat.y, quat.z, quat.w])
        self.drone_state.vel = np.array([vel.x, vel.y, vel.z])
        self.drone_state.omega = np.array([omega.x, omega.y, omega.z])
                
        msg = Vector3Stamped()
        msg.header = Header()
        msg.header.stamp = rclpy.time.Time().to_msg()
        msg.vector = np2vec3(omega_diff/np.pi*180)
        self.omega_debug_pub.publish(msg)
        
        self.drone_state.omega = omega_diff
        
        
        # self.allcfs.get_logger().info(f"pos: {self.drone_state.pos}, quat: {self.drone_state.quat}, vel: {self.drone_state.vel}, omega: {self.drone_state.omega}", throttle_duration_sec=1.0)
    
    def get_drone_target(self):
        # update target
        try:
            pos_in_map = self.traj.poses[self.step_cnt]
        except:
            pos_in_map = self.traj.poses[-1]
            pos_in_map.header.stamp = rclpy.time.Time().to_msg()
        
        transform = self.tf_buffer.lookup_transform("world", "map", rclpy.time.Time())
        pos = do_transform_pose(pos_in_map.pose, transform).position
        
        pos = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
        # calculate velocity using finite difference, change to differential flatness later
        self.drone_target.vel = (pos - self.drone_target.pos) * self.rate
        self.drone_target.omega = np.zeros(3)
        self.drone_target.pos = pos
        self.drone_target.quat = np.array([0.0, 0.0, 0.0, 1.0])
        
        self.target_pub.publish(pos_in_map)
       

    def set_attirate(self, omega_target:np.ndarray, thrust_target:float):
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
        self.traj.header.stamp = rclpy.time.Time().to_msg()
        self.traj_pub.publish(self.traj)

        self.timeHelper.sleepForRate(self.rate)
        
    def step(self, action):
        
        # send control signal
        target_roll_rate = action[0]
        target_pitch_rate = action[1]
        target_yaw_rate = action[2]
        omega_target = np.array([target_roll_rate, target_pitch_rate, target_yaw_rate])
        thrust_target = action[3]
        
        if self.no_action:
            self.set_attirate(np.zeros(3), 0.0)
        else:    
            self.set_attirate(omega_target, thrust_target)
            
        self.timeHelper.sleepForRate(self.rate)

        
        # observation
        # self.get_drone_target(omega_target)
        

        # if np.any(self.xyz_drone > (self.xyz_max + self.world_center)) or np.any(self.xyz_drone < (self.xyz_min + self.world_center)):
        #     print(f'{self.xyz_drone} is out of bound')
        #     self.emergency(next_obs)

        self.step_cnt += 1


    def emergency(self, obs):
        # stop
             
        self.reset()
        for cf in self.allcfs.crazyflies:
            cf.emergency()
        raise ValueError
    
    def pid_controller(self):
        thrust, roll_rate, pitch_rate, yaw_rate = self.pos_pid(self.drone_state, self.drone_target)
        return np.array([roll_rate, pitch_rate, yaw_rate, thrust])


    def _generate_traj(self):

        #generate trajectory
        traj = Path()
        traj.header.frame_id = "map"
        
        base_w = 2 * np.pi / 2.0
        t = np.arange(0, int(10.0*self.rate)) / self.rate
        t = np.tile(t, (3,1)).transpose()
        traj_xyz = np.zeros((len(t), 3))
        traj_vxyz = np.zeros((len(t), 3))
        A = np.array([0.5, 0.5, 0.0])
        w = np.array([base_w, base_w, base_w*2.0])
        phase = np.array([np.pi/2,0.0,np.pi])
        traj_xyz = A * np.sin(t*w+phase)
        traj_vxyz = w * A * np.cos(t*w+phase)


        for j in range(traj_xyz.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position = np2point(traj_xyz[j])
            traj.poses.append(pose)

        current_point = self.tf_buffer.lookup_transform('map', f"{self.cf_name}", rclpy.time.Time()).transform.translation
        current_point = np.array([current_point.x, current_point.y, current_point.z])

        # takeoff trajectory
        target_point = current_point.copy()
        target_point[2] += 0.5
        takeoff_traj_poses = line_traj(self.rate, current_point, target_point, 5.0).poses + line_traj(self.rate, target_point, traj_xyz[0], 5.0).poses

        # landing trajectory
        target_point = current_point.copy()
        target_point[2] += 0.5
        current_point = traj_xyz[-1].copy()
        landing_traj_poses = line_traj(self.rate, current_point, target_point, 7.0).poses

        current_point = target_point.copy() # when close to the land, the drone is not stable
        target_point[2] -= 0.5

        traj.poses = takeoff_traj_poses + traj.poses + landing_traj_poses + line_traj(self.rate, current_point, target_point, 1.0).poses

        # # debug: only takeoff and landing
        # current_point = self.tf_buffer.lookup_transform('map', f"{self.cf_name}", rclpy.time.Time()).transform.translation
        # current_point = np.array([current_point.x, current_point.y, current_point.z])
        # target_point = current_point.copy()
        # target_point[2] += 1.0
        # target_point[0] += 0.5
        # target_point1 = target_point.copy()
        # target_point1[0] += 0.5
        
        # print("current_point", current_point)
        # print("target_point", target_point)
        # traj.poses = line_traj(self.rate, current_point, target_point, 10.0).poses + line_traj(self.rate, target_point, current_point, 10.0).poses
        

        return traj


    
def main():

    cfctl = Crazyflie()

    cfctl.enable_logging()

    try:
        print('reset...')
        cfctl.reset()

        print('take off')
        while cfctl.step_cnt < len(cfctl.traj.poses) and rclpy.ok():
        # while cfctl.step_cnt < 10:
            cfctl.get_drone_target()
            action = cfctl.pid_controller() * 1.0
            cfctl.step(action)


    except KeyboardInterrupt:
        print('KeyboardInterrupt')
    
    finally:
        cfctl.disable_logging()
        cfctl.pos_pid.save_log()
        cfctl.set_attirate(np.zeros(3), 0.0)
        # stop
        print('stop')

        rclpy.shutdown()

if __name__ == "__main__":
    main()