import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray as MsgFloat
import numpy as np

class Visualizer(Node):

    def __init__(self):
        super().__init__('visualizer',
                            allow_undeclared_parameters=True,
                            automatically_declare_parameters_from_overrides=True)
        
        # Get parameters from launcher
        self.node_frequency = self.get_parameter('node_frequency').value
        self.goal_pose_arrived = False

        #######################################################################################

        # Create the publisher that will communicate with Rviz
        self.timer = self.create_timer(1.0/self.node_frequency, self.publish_data)
        self.publisher = self.create_publisher(
                                                Marker, 
                                                '/visualization_topic', 
                                                1)
        
        self.create_subscription(
                        MsgFloat, 
                        '/goal_topic', #  topic_name
                        self.goal_callback, 
                        10)

        # Initialize the current_pose method (in this example you can also use list or np.array)                                         
        self.current_pose = Pose()

    def goal_callback(self, goal_msg):
        self.goal_pose = goal_msg.data
        self.goal_pose_arrived = True
        
            
    def publish_data(self):
        
        if self.goal_pose_arrived:

            # Set the type of message to send to Rviz -> Marker
            # (see http://docs.ros.org/en/noetic/api/visualization_msgs/html/index-msg.html)
            marker = Marker()

            # Select the name of the reference frame, without it markers will be not visualized
            marker.header.frame_id = 'my_frame'
            marker.header.stamp = self.get_clock().now().to_msg()

            # Select the type of marker
            marker.type = Marker.CUBE

            # set the pose of the marker (orientation is omitted in this example)
            marker.pose.position.x = self.goal_pose[0]
            marker.pose.position.y = self.goal_pose[1]
            marker.pose.position.z = self.goal_pose[2]

            # Select the marker action (ADD, DELATE)
            marker.action = Marker.ADD

            # Select the namespace of the marker
            marker.ns = 'goal_pose'


            # Specify the scale of the marker
            scale = 0.2
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale

            # Specify the color of the marker as RGBA

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            # Let's publish the marker
            self.publisher.publish(marker)

def main():
    rclpy.init()

    visualizer = Visualizer()

    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        print("----- Visualizer stopped cleanly -----")
    finally:
        rclpy.shutdown() 

if __name__ == '__main__':
    main()