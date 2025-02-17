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
import torch
from scipy.spatial.transform import Rotation


'''
ign service --reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean --service /world/ocean/set_pose --timeout 5000 --req 'name: "slug", position: {y: 6}'
ign service --reqtype ignition.msgs.Physics --reptype ignition.msgs.Boolean --service /world/ocean/set_physics --timeout 5000 --req 'max_step_size: 0'
'''

class EnvManager(Node):
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

# Add model and spin to test the agent
class ThrusterCommandPublisher(Node):
    def __init__(self, position_reader):
        super().__init__('thruster_command_publisher')
        # Used to read the position to predict
        self.position_reader = position_reader
        # One publisher for each thruster
        self.publishers_ = [self.create_publisher(Float64, '/bluerov2/cmd_thruster{}'.format(i + 1), 10) for i in range(6)]
        self.timer = self.create_timer(1.0, self.timer_callback)  # Publish every 1 second
        self.model = None

    def timer_callback(self):
        policy = [0] * 6
        if self.model is not None:
            observation = self.position_reader.get_observation()
            policy = self.model(observation)

        for power, publisher in zip(policy, self.publishers_):
            msg = Float64()
            msg.data = power
            publisher.publish(msg)

        self.get_logger().info('Executed thruster policy')
    
    def execute(self, action):
        for power, publisher in zip(action, self.publishers_):
            msg = Float64()
            msg.data = power
            publisher.publish(msg)

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
    def __init__(self, env_manager):
        super().__init__('image_saver')

        # Used to get target position
        self.env_manager = env_manager

        self.subscription = self.create_subscription(
            Pose,
            '/bluerov2/pose_gt',
            self.position_callback,
            10
        )
        self.position = Pose()
    
    def position_callback(self, msg):
        self.position = msg
    
    def get_observation(self):
        agent_pos = np.array(self.position.position)
        target_pos = np.array(self.env_manager.target_position)
        
        # Compute displacement vector from agent to target (in world coordinates)
        displacement = target_pos - agent_pos
        
        # Normalize the agent's quaternion to ensure it's a unit quaternion
        agent_quat = np.array(self.position.orientation)
        norm = np.linalg.norm(agent_quat)
        if np.isclose(norm, 0.0):
            raise ValueError("Agent's quaternion has zero magnitude.")
        agent_quat_normalized = agent_quat / norm
        
        # Create a rotation object from the agent's quaternion
        rotation = Rotation.from_quat(agent_quat_normalized)
        
        # Invert the rotation to transform world→local coordinates
        inv_rotation = rotation.inv()
        
        # Rotate the displacement vector into the agent's local frame
        local_displacement = inv_rotation.apply(displacement)
        
        # Extract x (forward/backward) and y (left/right) components
        x = local_displacement[0]
        y = local_displacement[1]
        
        # Compute the straight-line 3D distance
        distance = np.linalg.norm(displacement)
        
        return (x, y, distance)

# def main(args=None):
#     rclpy.init(args=args)
#     env_manager = EnvManager()
#     position_reader = PositionReader(env_manager)
#     thruster_command_publisher = ThrusterCommandPublisher(position_reader)
#     # image_saver = ImageSaver()
#     # rclpy.spin(thruster_command_publisher)
#     # rclpy.spin_once(image_saver)

#     # image_saver.destroy_node()
#     thruster_command_publisher.destroy_node()
#     position_reader.destroy_node()
#     env_manager.destroy_node()

#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
