import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64  
from sensor_msgs.msg import Image
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose, Point, Quaternion
from ignition import math
import cv2
import numpy as np
from cv_bridge import CvBridge
import time
import numpy as np
import subprocess

'''
ign service --reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean --service /world/ocean/set_pose --timeout 5000 --req 'name: "slug", position: {y: 6}'
ign service --reqtype ignition.msgs.Physics --reptype ignition.msgs.Boolean --service /world/ocean/set_physics --timeout 5000 --req 'max_step_size: 0'
'''

class ResetEpisodeNode(Node):
    def __init__(self):
        super().__init__('reset_episode_node')
        self.set_state = self.create_client(SetEntityState, '/world/ocean/set_pose')
        self.timer = self.create_timer(1.0, self.timer_callback)  # Publish every 1 second

        self.target_position = np.zeros(3, dtype = np.float32)

    def set_position(self, name, x, y, z, qx, qy, qz, qw):
        position_str = '{' + 'x: {}, y: {}, z: {}'.format(x, y, z) + '}'
        orientation_str = '{' + 'x: {}, y: {}, z: {}, w: {}'.format(qx, qy, qz, qw) + '}'
        command = [
            'ign',
            'service',
            '--reqtype',
            'ignition.msgs.Pose',
            '--reptype',
            'ignition.msgs.Boolean',
            '--service',
            '/world/ocean/set_pose',
            '--timeout',
            '5000',
            '--req',
            """'name: "{}", position: {}, orientation: {}'""".format(name, position_str, orientation_str)
        ]

        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    def toggle_simulation(self, toggle):
        command = [
            'ign',
            'service',
            '--reqtype',
            'ignition.msgs.Physics',
            '--reptype',
            'ignition.msgs.Boolean',
            '--service',
            '/world/ocean/set_physics',
            '--timeout',
            '5000',
            '--req',
            'max_step_size:',
            "{}".format(toggle * 0.001)
        ]

        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def timer_callback(self):
        self.target = np.array([3, 0, 3], dtype = np.float32) + (np.random.rand(3) * 1)
        start_position = np.array([0, 0, 3], dtype = np.float32)

        self.toggle_simulation(False)

        self.set_position('slug', *self.target, 0, 0, 0, 1)
        self.set_position('bluerov2', *start_position, 0, 0, 0, 1)

        self.toggle_simulation(True)

        self.get_logger().info('Reset robot and slug')

class ThrusterCommandPublisher(Node):
    def __init__(self):
        super().__init__('thruster_command_publisher')
        self.publisher_ = self.create_publisher(Float64, '/bluerov2/cmd_thruster4', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)  # Publish every 1 second
        self.command_value = 0.0

    def timer_callback(self):
        msg = Float64()
        msg.data = self.command_value  # Set the command value
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing command to thruster1: {msg.data}')
        self.command_value += 0.1  # Increment the command value for demonstration

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.subscription = self.create_subscription(
            Image,
            '/bluerov2/image',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.image_saved = False  # Flag to ensure we save only one image

    def image_callback(self, msg):
        if not self.image_saved:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imwrite('saved_image.png', cv_image)
            self.get_logger().info('Image saved as saved_image.png')
            # self.image_saved = True  

class PositionReader(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.subscription = self.create_subscription(
            Pose,
            '/bluerov2/pose_gt',
            self.position_callback,
            10
        )
        self.position = Pose()
    
    def position_callback(self, msg):
        self.position = msg

def main(args=None):
    rclpy.init(args=args)
    reset_publisher = ResetEpisodeNode()
    # thruster_command_publisher = ThrusterCommandPublisher()
    # image_saver = ImageSaver()
    rclpy.spin_once(reset_publisher)
    # rclpy.spin(thruster_command_publisher)
    # rclpy.spin_once(image_saver)

    # thruster_command_publisher.destroy_node()
    # image_saver.destroy_node()
    reset_publisher.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
