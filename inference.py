from datetime import datetime
import time
import rclpy
import torch
from thrust import EnvManager, PositionReader, ThrusterCommandPublisher
from train import RobotActor
import numpy as np
import os

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    rclpy.init()
    robot_actor = RobotActor(6)
    try:
        robot_actor.load_state_dict(torch.load('checkpoints/actor.pth'))
    except:
        print('No model found')

    env_manager = EnvManager()
    position_reader = PositionReader(env_manager)
    thruster_command_publisher = ThrusterCommandPublisher(position_reader)

    rclpy.spin_once(position_reader)
    pose = position_reader.position
    position = [pose.position.x, pose.position.y, pose.position.z]
    orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    env_manager.timer_callback(position, orientation)

    goal_distance = 3

    target_distances = []
    center_distances = []
    for i in range(1000):
        rclpy.spin_once(position_reader)
        is_front, x, y, distance = position_reader.get_observation()

        target_distance = np.abs(distance - goal_distance)
        center_distance = np.sqrt(x**2 + y**2)

        obs = torch.tensor([x, y, distance], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            policy, _ = robot_actor(obs)
        thrust = policy
        print(policy)
        thruster_command_publisher.execute(thrust)
        rclpy.spin_once(thruster_command_publisher, timeout_sec = 0.1)
         
        target_distances.append(target_distance)
        center_distances.append(center_distance)
        if i % 10 == 0:
            print('i:', i)
        time.sleep(0.1)

    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d_%H:%M:%S")
    with open('data/distance_to_target_{}.txt'.format(formatted_date), 'w') as f:
        for val in target_distances:
            f.write('{}\n'.format(val))

    with open('data/distance_from_center_{}.txt'.format(formatted_date), 'w') as f:
        for val in enumerate(center_distances):
            f.write('{}\n'.format(val))