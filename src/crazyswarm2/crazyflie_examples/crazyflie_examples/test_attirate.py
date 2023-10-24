from crazyflie_py import Crazyswarm
import numpy as np
import tf2_ros
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose
from tf2_geometry_msgs import do_transform_pose
from .pid_controller import PIDController, PIDParam, PIDState
from nav_msgs.msg import Path
from std_msgs.msg import Header
from .util import np2point, np2quat, np2vec3, line_traj

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
        

        # control parameters
        self.max_vel = 6.0
        self.rate = 50
        self.xyz_min = np.array([-2.0, -3.0, -2.0])
        self.xyz_max = np.array([2.0, 2.0, 1.5])
        
        self.pos_pid_param = PIDParam(m=self.mass, g=self.g, max_thrust=0.6, max_omega=np.array([1.0, 1.0, 1.0])*0.1, Kp=np.array([1.0, 1.0, 1.0])*2.0, Kd=np.array([1.0, 1.0, 1.0])*2.0, Kp_att=np.array([1.0, 1.0, 1.0])*0.05)
        
        self.pos_pid = PIDController(self.pos_pid_param)

        # initialize parameters
        self.step_cnt = 0
        
        self.traj = Path()
        self.drone_state = PIDState(pos=np.zeros(3), vel=np.zeros(3), quat=np.array([0.0, 0.0, 0.0, 1.0]), omega=np.zeros(3))
        self.drone_target = PIDState(pos=np.zeros(3), vel=np.zeros(3), quat=np.array([0.0, 0.0, 0.0, 1.0]), omega=np.zeros(3))

        # ROS related initialization
        self.allcfs.create_subscription(PoseStamped, f'{self.cf.prefix}/pose', self.state_callback_cf, self.rate)

        # initialize publisher
        self.target_pub = self.allcfs.create_publisher(PoseStamped, 'target_pose', self.rate)
        self.traj_pub = self.allcfs.create_publisher(Path, 'traj', self.rate)
        
        # initialize tf
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(buffer=self.tf_buffer, node=self.allcfs)
        self.tf_publisher = tf2_ros.TransformBroadcaster(node=self.allcfs)
        self.static_tf_publisher = tf2_ros.StaticTransformBroadcaster(node=self.allcfs)
        static_transformStamped = TransformStamped()
        static_transformStamped.header.stamp = rclpy.time.Time().to_msg()
        static_transformStamped.header.frame_id = "world"
        static_transformStamped.child_frame_id = "map"
        static_transformStamped.transform.translation = np2vec3(self.world_center)
        static_transformStamped.transform.rotation = np2quat(np.array([0.0, 0.0, 0.0, 1.0]))
        self.static_tf_publisher.sendTransform(static_transformStamped)        
        
        if self.is_sim:
            static_transformStamped.header.frame_id = f"{self.cf_name}"
            static_transformStamped.child_frame_id = f"{self.cf_name}/kf"
            static_transformStamped.transform.translation = np2vec3(np.array([0.0, 0.0, 0.0]))
            static_transformStamped.transform.rotation = np2quat(np.array([0.0, 0.0, 0.0, 1.0]))
            self.static_tf_publisher.sendTransform(static_transformStamped)
            
            

    def state_callback_cf(self, data):
        msg = TransformStamped()
        msg.header.stamp = rclpy.time.Time().to_msg()
        msg.header.frame_id = data.header.frame_id
        msg.child_frame_id = f"{self.cf_name}/kf"
        msg.transform.translation = data.pose.position
        msg.transform.rotation = data.pose.orientation
        self.tf_publisher.sendTransform(msg)


    def get_drone_state(self):
        msg = self.tf_buffer.lookup_transform('world', f"{self.cf_name}/kf", rclpy.time.Time())
        pos = msg.transform.translation
        quat = msg.transform.rotation
        
        # calculate velocity using finite difference, change to KF later
        pos = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
        quat = np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float32)
        self.drone_state.vel = (pos - self.drone_state.pos) * self.rate
        quat_deriv = (quat - self.drone_state.quat) * self.rate
        self.drone_state.omega = 2 * quat_deriv[:-1] / (np.linalg.norm(self.drone_state.quat[:-1], axis=-1, keepdims=True)+1e-3)
        self.drone_state.pos = pos
        self.drone_state.quat = quat
    
    def get_drone_target(self, omega_target:np.ndarray):
        # update target
        try:
            pos_in_map = self.traj.poses[self.step_cnt]
        except:
            pos_in_map = self.traj.poses[-1]
        
        transform = self.tf_buffer.lookup_transform("world", "map", rclpy.time.Time())
        pos = do_transform_pose(pos_in_map.pose, transform).position
        
        pos = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
        # calculate velocity using finite difference, change to differential flatness later
        self.drone_target.vel = (pos - self.drone_target.pos) * self.rate
        self.drone_target.omega = omega_target
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

        self.get_drone_state()
        self.timeHelper.sleepForRate(self.rate)
        
    def step(self, action):
        
        # send control signal
        target_roll_rate = action[0]
        target_pitch_rate = action[1]
        target_yaw_rate = action[2]
        omega_target = np.array([target_roll_rate, target_pitch_rate, target_yaw_rate])
        thrust_target = action[3]
        self.set_attirate(omega_target, thrust_target)
        self.timeHelper.sleepForRate(self.rate)
        
        # observation
        self.get_drone_state()
        self.get_drone_target(omega_target)
        

        # if np.any(self.xyz_drone > (self.xyz_max + self.world_center)) or np.any(self.xyz_drone < (self.xyz_min + self.world_center)):
        #     print(f'{self.xyz_drone} is out of bound')
        #     self.emergency(next_obs)

        self.step_cnt += 1


    def emergency(self, obs):
        # stop
        
        self.traj.clear()
        current_point = self.drone_state.pos.copy()
        target_point = current_point.copy()
        target_point[2] = 0.0
        current_point = PoseStamped(header=Header(frame_id="world"), pose=Pose(position=np2point(current_point), orientation=np2quat(self.drone_state.quat)))
        target_point = PoseStamped(header=Header(frame_id="world"), pose=Pose(position=np2point(target_point), orientation=np2quat(self.drone_state.quat)))
        transform = self.tf_buffer.lookup_transform("map", "world", rclpy.time.Time())
        current_point = do_transform_pose(current_point.pose, transform).position
        target_point = do_transform_pose(target_point.pose, transform).position
        landing_traj_poses = line_traj(self.rate, current_point, target_point, 3.0).poses
        self.traj.poses = landing_traj_poses
        self.traj.header.stamp = rclpy.time.Time().to_msg()
        self.traj_pub.publish(self.traj)
        
        print('emergency, landing...')

        return        
        # self.reset()
        # for cf in self.allcfs.crazyflies:
        #     cf.emergency()
        # raise ValueError
    
    def pid_controller(self):
        thrust, roll_rate, pitch_rate, yaw_rate = self.pos_pid(self.drone_state, self.drone_target)
        return np.array([roll_rate, pitch_rate, yaw_rate, thrust])


    def _generate_traj(self):

        #generate trajectory
        traj = Path()
        traj.header.frame_id = "map"
        
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


        for j in range(traj_xyz.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position = np2point(traj_xyz[j])
            traj.poses.append(pose)

        current_point = self.tf_buffer.lookup_transform('map', f"{self.cf_name}/kf", rclpy.time.Time()).transform.translation
        current_point = np.array([current_point.x, current_point.y, current_point.z])

        # takeoff trajectory
        target_point = current_point.copy()
        target_point[2] += 1.0
        takeoff_traj_poses = line_traj(self.rate, current_point, target_point, 2.0).poses + line_traj(self.rate, target_point, traj_xyz[0], 2.0).poses


        # landing trajectory
        target_point = current_point.copy()
        current_point = traj_xyz[-1].copy()
        landing_traj_poses = line_traj(self.rate, current_point, target_point, 3.0).poses


        traj.poses = takeoff_traj_poses + traj.poses + landing_traj_poses

        return traj


    
def main():

    cfctl = Crazyflie()

    print('reset...')
    cfctl.reset()

    print('take off')
    while cfctl.step_cnt < len(cfctl.traj.poses):
    # while cfctl.step_cnt < 10:
        action = cfctl.pid_controller() * 1.0
        cfctl.step(action)

    # stop
    cfctl.step(np.array([0.0, 0.0, 0.0, 0.0]))


    rclpy.shutdown()

if __name__ == "__main__":
    main()
