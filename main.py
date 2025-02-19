import gymnasium
from gymnasium import spaces
import numpy as np
import rclpy
import os
import sys
import time
import logging
import ray

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms.ppo import PPO, PPOConfig

from thrust import EnvManager, PositionReader, ThrusterCommandPublisher

rclpy.init()

# env_mngr = EnvManager()
# pos_reader = PositionReader(env_mngr)
# thruster_publisher = ThrusterCommandPublisher(pos_reader)

class GazeboEnv(gymnasium.Env):
    def __init__(self, config=None):
        try:
            rclpy.init()
        except:
            pass
        self.action_space = spaces.Box(shape = (6,), high = 1, low = -1)  
        self.observation_space = spaces.Box(low = -10, high = 10, shape=(3,), dtype=np.float32)  

        # if env_mngr is not None:
        #     self.env_manager = env_mngr
        # else:
        self.env_manager = EnvManager()
        
        # if pos_reader is not None:
        #     self.position_reader = pos_reader 
        # else:
        self.position_reader = PositionReader(self.env_manager)

        # if thruster_publisher is not None:
        #     self.thruster_command_publisher = thruster_publisher
        # else:
        self.thruster_command_publisher = ThrusterCommandPublisher(self.position_reader)

        self.goal_distance = 3

        # rclpy.spin_once(self.env_manager)
        self.env_manager.timer_callback()
        time.sleep(1)
        # rclpy.spin(self.position_reader)
        # rclpy.spin(self.thruster_command_publisher)
        super(GazeboEnv, self).__init__()

    def reset(self, seed = None, options = None):
        # Reset the environment
        # rclpy.spin_once(self.env_manager)
        self.env_manager.timer_callback()
        time.sleep(1)
        obs = self.position_reader.get_observation()
        
        return np.array([obs[0], obs[1], obs[2] - self.goal_distance], dtype = np.float32), {}

    def step(self, action):
        time.sleep(0.5)
        self.thruster_command_publisher.execute(action)
        x, y, distance = self.position_reader.get_observation()
        self.state = np.array([x, y, distance - self.goal_distance])
        reward = np.abs(self.goal_distance - distance) + np.sqrt(2) - np.sqrt((x ** 2) + (y ** 2)) + 1
        done = np.random.random() > 0.95  
        info = {}  
        return self.state.astype(np.float32), reward, done, False, info
    
    def __getstate__(self):
        pose = self.position_reader.position
        target_pos = self.env_manager.target_position.copy()

        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        return  {'position': position, 'orientation': orientation, 'target': target_pos}
    
    def toggle_simulation(self, toggle):
        try:
            self.env_manager.toggle_simulation(toggle)
        except:
            self.env_manager = EnvManager()
            self.env_manager.toggle_simulation(toggle)

# This is used for simulation control
null_env = GazeboEnv()

class CustomCallbacks(RLlibCallback):
    # def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs):
    #     print("Before training on batch")
    #     # Pause simulation for training logic
    #     self.env_manager.toggle_simulation(False)
    #     super().on_learn_on_batch(policy=policy, train_batch=train_batch, result=result, **kwargs)

    def on_train_result(self, *, algorithm, metrics_logger = None, result: dict, **kwargs):
        print("After training on batch")
        # Continue the simulation after training logic
        null_env.toggle_simulation(True)

    def on_sample_end(self, *, env_runner = None, metrics_logger = None, samples, worker = None, **kwargs):
        print("After sampling")
        # Pause the simulation after sampling logic
        null_env.toggle_simulation(False)

    def on_episode_start(self, **kwargs):
        print("Episode start")
        # Continue the simulation after episode start logic
        null_env.toggle_simulation(True)

if __main__ == '__main__':
    ppo_config = PPOConfig()
    ppo_config.environment(GazeboEnv).callbacks(CustomCallbacks).framework("torch").learners(num_learners = 1, num_gpus_per_learner = 1).api_stack(enable_rl_module_and_learner = True, enable_env_runner_and_connector_v2 = True).env_runners(num_envs_per_env_runner = 1, num_env_runners = 1)
    ppo_config.enable_new_api_stack = True
    ppo_config.enable_rl_module_and_learner = True
    ppo_config.enable_env_runner_and_connector_v2 = True

    ppo_config.sample_timeout_s = 600
    ppo_config.rollout_fragment_length = 50  
    ppo_config.reuse_actors = True  

    trainer = PPO(config=ppo_config)

    logging.basicConfig(level=logging.DEBUG)
    ray.init(logging_level=logging.DEBUG, ignore_reinit_error = True)

    result = trainer.train()

    checkpoint_dir = os.path.join("checkpoints")

    trainer.save(os.path.join("checkpoints"))