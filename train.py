import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import rclpy
import numpy as np
import os
from datetime import datetime

CAMERA_HEIGHT = 4
CAMERA_WIDTH = 4
A=0.1
B=0.1
C=5

class RobotActor(nn.Module):
    def __init__(self, actions: int):
        super(RobotActor, self).__init__()
        self.actions = actions
        self.network = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, actions)
        )

        self.network_std = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, actions)
        )
    def forward(self, x):
        result = self.network(x)[0]
        log_std = self.network_std(x)[0]
        return result, log_std
    
class RobotCritic(nn.Module):
    def __init__(self):
        super(RobotCritic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.network(x)

def reward_function(x, y, distance, goal_distance):
    reward = (-np.abs(goal_distance - distance) ** 2 * A) - (B * np.sqrt(((x / CAMERA_WIDTH) ** 2) + ((y / CAMERA_HEIGHT) ** 2))) + C
    return reward

def out_of_bounds(x, y, distance):
    return np.abs(x / distance) > CAMERA_WIDTH or np.abs(y / distance) > CAMERA_HEIGHT

from thrust import EnvManager, ThrusterCommandPublisher, PositionReader, ImageReader
import time

class GzEnv():
    def __init__(self):
        self.env_manager = EnvManager()
        self.position_reader = PositionReader(self.env_manager)
        self.thruster_command_publisher = ThrusterCommandPublisher(self.position_reader, False)
        self.image_reader = ImageReader()
        self.goal_distance = 3
        self.steps = 0

    def step(self, action):
        self.steps += 1
        # Execute the action and wait a bit
        self.thruster_command_publisher.execute(action)

        rclpy.spin_once(self.thruster_command_publisher, timeout_sec = 0.2)
        time.sleep(0.1)

        # Update position in reader
        rclpy.spin_once(self.position_reader)
        rclpy.spin_once(self.image_reader)

        self.image_observation = self.image_reader.cur_image
        is_front, x, y, distance = self.position_reader.get_observation()
        self.observation = np.array([x, y, distance], dtype = np.float32)
        reward = reward_function(x, y, distance, self.goal_distance)
        
        return self.observation, reward, out_of_bounds(x, y, distance) or not is_front, self.steps > 50, {}

    def reset(self):
        self.steps = 0
        self.thruster_command_publisher.execute([0] * 6)

        rclpy.spin_once(self.thruster_command_publisher, timeout_sec = 0.2)
        # Let the velocity level out a bit
        time.sleep(3)
        rclpy.spin_once(self.position_reader)
        pose = self.position_reader.position
        position = np.array([pose.position.x, pose.position.y, pose.position.z], dtype = np.float32)
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], dtype = np.float32)

        self.env_manager.timer_callback(position, orientation)
        # Update position in reader
        rclpy.spin_once(self.image_reader)

        self.image_observation = self.image_reader.cur_image
        is_front, x, y, distance = self.position_reader.get_observation()
        self.observation = np.array([x, y, distance], dtype = np.float32)

        return self.observation - np.array([0, 0, -self.goal_distance], dtype = np.float32), {}
    
    def close(self):
        self.env_manager.destroy_node()
        self.position_reader.destroy_node()
        self.thruster_command_publisher.destroy_node()
        rclpy.shutdown()

class Agent:
    def __init__(self, actions, input_dims, alpha, gamma, epsilon, gae_lambda, epochs, batch_size, learn_iters):
        self.actor = RobotActor(actions)
        self.critic = RobotCritic()
        self.action_size = actions
        self.input_dims = input_dims
        self.actor.opt = torch.optim.Adam(self.actor.parameters(), lr = alpha)
        self.critic.opt = torch.optim.Adam(self.critic.parameters(), lr = alpha)
        self.gamma = gamma
        self.epsilon = epsilon
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        self.learn_iters = learn_iters
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def reset_mem(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def store_mem(self, state, prob, action, reward, value, done):
        self.states.append(state.tolist())
        self.probs.append(prob)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
    def batching(self):
        batch_size_indices = np.arange(0, len(self.states), self.batch_size)
        batch_indices = np.arange(0, self.batch_size, dtype = np.int64)
        np.random.shuffle(batch_indices)
        batch_indices = batch_indices.tolist()
        rvalue = []
        for i in batch_size_indices:
            rvalue += batch_indices[i:i + self.batch_size]
        return np.array(rvalue, dtype = np.int64)
        
    def act(self, obs):
        obs = obs.unsqueeze(0)
        # Actor outputs mean and log standard deviation for each of the 6 thrusters
        mean, log_std = self.actor(obs)
        std = log_std.exp()  # Convert log std to std
        # Create a Gaussian distribution for each thruster
        dist = Normal(mean, std)
        # Sample actions for all 6 thrusters
        action = dist.sample()
        # Compute log probability of the sampled actions
        log_prob = dist.log_prob(action).sum(dim=-1).item()  # Sum log probs across thrusters
        # Get the value estimate from the critic
        value = self.critic(obs).squeeze().item()
        return action.tolist(), log_prob, value

    def learn(self):
        advantage = np.zeros(len(self.rewards) - 1, dtype=np.float32)
        self.states = np.array(self.states, dtype=np.float32)

        for _ in range(self.epochs):
            batch_indices = np.array(self.batching(), dtype=np.int64)
            # Compute GAE (Generalized Advantage Estimation)
            for j in range(len(self.rewards) - 1):
                discount = 1
                a = 0
                for k in range(j, len(self.rewards) - 1):
                    a += discount * (((1 - self.dones[k]) * self.gamma * self.values[k + 1]) + self.rewards[k] - self.values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[j] = a
            advantage = torch.Tensor(advantage).float()

            # Prepare batches
            state_batches = []
            old_log_probs = []
            action_batches = []
            value_batches = []
            advantage_batches = []
            for i in batch_indices:
                state_batches.append(self.states[i])
                old_log_probs.append(self.probs[i])
                action_batches.append(self.actions[i])
                value_batches.append(self.values[i])
                advantage_batches.append(advantage[i])

            state_batches = torch.Tensor(np.array(state_batches)).float()
            old_log_probs = torch.Tensor(np.array(old_log_probs)).float()
            action_batches = torch.Tensor(np.array(action_batches)).float()
            value_batches = torch.Tensor(np.array(value_batches)).float()
            advantage_batches = torch.Tensor(advantage_batches).float()

            # Predict new actions and values
            mean, log_std = self.actor(state_batches)
            std = log_std.exp()
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(action_batches).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()  # Sum entropy across thrusters and average over batch

            # Actor loss (clipped surrogate objective)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage_batches
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage_batches
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss (MSE between predicted and target values)
            value_pred = self.critic(state_batches).squeeze()
            target_values = advantage_batches + value_batches
            critic_loss = F.mse_loss(value_pred, target_values)

            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy  # Adjust entropy coefficient as needed

            # Update networks
            self.actor.opt.zero_grad()
            self.critic.opt.zero_grad()
            loss.backward()
            self.actor.opt.step()
            self.critic.opt.step()

        self.reset_mem()


if __name__ == '__main__':
    os.makedirs('models', exist_ok = True)
    os.makedirs('results', exist_ok = True)

    ACTIONS = 6
    INPUT_DIMS = 3
    LR = 5e-4
    DISCOUNT_FACTOR = 0.99
    POLICY_CLIP = 0.1
    SMOOTHING = 0.95
    EPOCHS = 4
    BATCH_SIZE = 5
    LEARN_ITERS = 20

    LOAD_MODELS = False

    rclpy.init()

    env = GzEnv()

    agent = Agent(ACTIONS, INPUT_DIMS, LR, DISCOUNT_FACTOR, POLICY_CLIP, SMOOTHING, EPOCHS, BATCH_SIZE, LEARN_ITERS)

    if LOAD_MODELS:
        try:
            agent.actor.load_state_dict(torch.load('actor.pth'))
            agent.critic.load_state_dict(torch.load('critic.pth'))
        except:
            print("No models found")

    EPISODES = 200

    steps = 0
    nobs = 0

    reward_histories = []
    best_avg = -9999

    for i in range(EPISODES):
        reward_history = []
        obs = torch.tensor(env.reset()[0])
        done = False
        score = 0
        while not done:
            action, prob, value = agent.act(obs)
            nobs, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            reward_history.append(reward)

            nobs = torch.tensor(nobs)
            agent.store_mem(obs, prob, action, reward, value, done)
            score += reward
            steps += 1
            obs = nobs
            if steps % agent.learn_iters == 0:
                agent.learn()
        print("Episode: {} Score: {}".format(i + 1, score))
        if np.mean(np.array(reward_history)) > best_avg:
            best_avg = np.mean(np.array(reward_history))
            torch.save(agent.actor.state_dict(), 'models/actor-best-{}.pth'.format(i))
        reward_histories.append(reward_history)
    env.close()

    torch.save(agent.actor.state_dict(), 'models/actor.pth')
    torch.save(agent.critic.state_dict(), 'models/critic.pth')


    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d_%H:%M:%S")
    with open('training_results_{}.txt'.format(formatted_date), 'w') as f:
        for i, rewards in enumerate(reward_histories):
            for reward in rewards:
                f.write('{},'.format(reward))
            f.write('\n') 