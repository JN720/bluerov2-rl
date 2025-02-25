from datetime import datetime
import time
import rclpy
import torch
from thrust import EnvManager, PositionReader, ThrusterCommandPublisher
from train import RobotActor
import numpy as np
import os
import sys

ROTATION_THRESHOLD = 0.3
DISTANCE_THRESHOLD = 0.5
POWER = 1

def procedural_policy(x, y, distance):
    forward = 0
    right = 0
    down = 0
    if distance > goal_distance + DISTANCE_THRESHOLD:
        forward = 1
    elif distance < goal_distance - DISTANCE_THRESHOLD:
        forward = -1
    
    if x > ROTATION_THRESHOLD:
        right = 1
    elif x < -ROTATION_THRESHOLD:
        right = -1
    
    if y > ROTATION_THRESHOLD:
        down = 1
    elif y < -ROTATION_THRESHOLD:
        down = -1
    
    policy = np.zeros(6, dtype = np.float32)
    if forward == 1:
        policy[0] = POWER
        policy[1] = POWER
        if right == 1:
            policy[4] = POWER
        elif right == -1:
            policy[3] = POWER
    elif forward == -1:
        policy[3] = POWER
        policy[4] = POWER
        if right == 1:
            policy[2] = POWER
        elif right == -1:
            policy[1] = POWER
    
    if down == 1:
        policy[5] = POWER
    elif down == -1:
        policy[6] = POWER

    return policy

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    use_procedural = False
    if len(sys.argv) > 1:
        if sys.argv[1] == 'procedural':
            use_procedural = True

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

        policy = None
        if use_procedural:
            policy = procedural_policy(x, y, distance)
        else:
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