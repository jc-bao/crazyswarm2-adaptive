import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs


class RobotPosePublisher(Node):

    def __init__(self):
        super().__init__('robot_pose_publisher')
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pose_pub = self.create_publisher(PoseStamped, '/cf_tf', 10)
        self.timer = self.create_timer(0.01, self.publish_robot_pose)

    def publish_robot_pose(self):
        try:
            # Get the transformation from the world frame to the robot frame
            trans = self.tf_buffer.lookup_transform('world', 'cf2', rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return

        # Convert the transform to a PoseStamped message
        pose = tf2_geometry_msgs.PoseStamped()
        pose.header.frame_id = 'world'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = trans.transform.translation.x
        pose.pose.position.y = trans.transform.translation.y
        pose.pose.position.z = trans.transform.translation.z
        pose.pose.orientation = trans.transform.rotation

        # Publish the PoseStamped message to the /robot_pose topic
        self.pose_pub.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    node = RobotPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # Destroy the node explicitly
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
