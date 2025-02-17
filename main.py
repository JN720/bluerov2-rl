import gym
from gym import spaces
import numpy as np
import rclpy
import os

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms.ppo import PPO

from thrust import EnvManager, PositionReader, ThrusterCommandPublisher

rclpy.init()

class GazeboEnv(gym.Env):
    def __init__(self, config=None):
        super(GazeboEnv, self).__init__()
        self.action_space = spaces.Box(shape = (6,), high = 1, low = -1)  
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)  

        self.env_manager = EnvManager()
        self.position_reader = PositionReader(self.env_manager)
        self.thruster_command_publisher = ThrusterCommandPublisher(self.position_reader)

        rclpy.spin(self.position_reader)
        rclpy.spin(self.thruster_command_publisher)

    def reset(self):
        # Reset the environment
        rclpy.spin_once(self.env_manager)
        return self.position_reader.get_observation()

    def step(self, action):
        self.thruster_command_publisher.execute(action)
        self.state = self.position_reader.get_observation()
        reward = np.sum(self.state)  
        done = np.random.random() > 0.95  
        info = {}  
        return self.state, reward, done, info

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

config = {
    "env": GazeboEnv,  
    "callbacks": CustomCallbacks,
    "framework": "torch",
    "num_workers": 1,  
    "env_config": {},  
    "epochs": 1
}

trainer = PPO(config=config)

trainer.train()

trainer.save(os.path.join("checkpoints"))
