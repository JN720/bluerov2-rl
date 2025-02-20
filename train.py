import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import rclpy
import numpy as np

class RobotActor(nn.Module):
    def __init__(self, actions: int):
        super(RobotActor, self).__init__()
        self.actions = actions
        self.network = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, actions * 2)
        )
    def forward(self, x):
        result = self.network(x)[0]
        return result[0:self.actions], result[self.actions:]
    
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
    reward = np.abs(goal_distance - distance) + np.sqrt(2) - np.sqrt((x ** 2) + (y ** 2)) + 1
    return reward

from thrust import EnvManager, ThrusterCommandPublisher, PositionReader
import time

class GzEnv():
    def __init__(self):
        self.env_manager = EnvManager()
        self.position_reader = PositionReader(self.env_manager)
        self.thruster_command_publisher = ThrusterCommandPublisher(self.position_reader, False)
        self.goal_distance = 3

    def step(self, action):
        # Execute the action and wait a bit
        self.thruster_command_publisher.execute(action)

        rclpy.spin_once(self.thruster_command_publisher, timeout_sec = 0.2)
        time.sleep(0.1)

        # Update position in reader
        rclpy.spin_once(self.position_reader)

        x, y, distance = self.position_reader.get_observation()
        self.observation = np.array([x, y, distance], dtype = np.float32)
        reward = reward_function(x, y, distance, self.goal_distance)
        
        return self.observation, reward, np.random.rand() > 0.95, False, {}

    def reset(self):
        self.env_manager.timer_callback()
        # Update position in reader
        rclpy.spin_once(self.position_reader)
        x, y, distance = self.position_reader.get_observation()
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
        
    def actt(self, obs):
        obs = obs.unsqueeze(0)
        policy = self.actor(obs).squeeze()
        policy1 = Categorical(F.softmax(policy[0:5], dim = -1))
        policy2 = Categorical(F.softmax(policy[5:8], dim = -1))
        policy3 = Categorical(F.softmax(policy[8:10], dim = -1))
        policy4 = Categorical(F.softmax(policy[10:12], dim = -1))
        value = self.critic(obs).squeeze().item()
        #action and log probs will be of size 3
        action1 = policy1.sample()
        action2 = policy2.sample()
        action3 = policy3.sample()
        action4 = policy4.sample()
        #since these log probs are passed directly into store mem,
        #and the same is done with the new probs, only the sum is returned
        prob1 = policy1.log_prob(action1).item()
        prob2 = policy2.log_prob(action2).item()
        prob3 = policy3.log_prob(action3).item()
        prob4 = policy3.log_prob(action3).item()
        
        return [action1.item(), action2.item(), action3.item(), action4.item()], prob1 + prob2 + prob3 + prob4, value
        
    def learnn(self):
        advantage = np.zeros(len(self.rewards) - 1, dtype = np.float32)
        self.states = np.array(self.states, dtype = np.float32)
        
        for i in range(self.epochs):
            batch_indices = np.array(self.batching(), dtype = np.int64)
            #gae
            #summation of memory
            for j in range(len(self.rewards) - 1):
                #delta coefficient
                discount = 1
                #advantage
                a = 0
                for k in range(j, len(self.rewards) - 1):
                    #delta of timestep = (done coefficient * gamma * next state value) + reward - current state value
                    #basically new value + reward - cur value
                    a += discount * (((1 - self.dones[k]) * self.gamma * self.values[k + 1]) + self.rewards[k] - self.values[k])
                    #gae lamba^n * gamma^n
                    discount *= self.gamma * self.gae_lambda
                #advantage at each timestep
                advantage[j] = a
            advantage = torch.Tensor(advantage).float()
            state_batches = []
            p1 = []
            ab = []
            vb = []
            advantage_batch = []
            #sampling of random memory
            for i in batch_indices:
                state_batches.append(self.states[i])
                p1.append(self.probs[i])
                ab.append(self.actions[i])
                vb.append(self.values[i])
                advantage_batch.append(advantage[i])
            state_batches = torch.Tensor(np.array(state_batches)).float()
            #these 2 are size 4 for the multi discrete implementation
            p1 = torch.Tensor(np.array(p1)).float()
            ab = torch.Tensor(np.array(ab)).long()
            vb = torch.Tensor(np.array(vb)).float()
            advantage_batch = torch.Tensor(advantage_batch).float()
            #predictions
            apred = self.actor(state_batches)
            apred1 = Categorical(F.softmax(apred[0, 0:5], dim = -1))
            apred2 = Categorical(F.softmax(apred[0, 5:8], dim = -1))
            apred3 = Categorical(F.softmax(apred[0, 8:10], dim = -1))
            apred4 = Categorical(F.softmax(apred[0, 10:12], dim = -1))
            cpred = self.critic(state_batches)
            #get new log probs corresponding to past actions from memory
            #there are 3 of these now
            #in the 37 implementation details thingy, they multiplied the probs for each distribution
            #since these are logits, they shall be added
            p2 = apred1.log_prob(ab[:, 0]) + apred2.log_prob(ab[:, 1]) + apred3.log_prob(ab[:, 2]) + apred4.log_prob(ab[:, 3])
            #actor loss calculation: this is the same now that the probs are combined
            pratio = p2.exp() / p1.exp()
            wpratio = pratio * advantage_batch
            cwpratio = torch.clamp(pratio, 1 - self.epsilon, 1 + self.epsilon) * advantage_batch
            aloss = (-torch.min(wpratio, cwpratio)).mean()
            #critic loss: gae + state value MSE'd with raw network prediction
            #gae + state value = new state + reward
            #in other words, optimize state value to become new state + reward
            ctarget = advantage_batch + vb
            criterion = torch.nn.MSELoss()
            #closs = ((ctarget - cpred) ** 2).mean()
            closs = criterion(ctarget.unsqueeze(-1), cpred)
            #now includes entropy term
            entropy = (0.1 * apred1.entropy()) + (0.8 * apred2.entropy()) + (0.2 * apred3.entropy()) + (0.1 * apred4.entropy())
            loss = aloss + (0.5 * closs) - (0.4 * entropy)
            self.actor.opt.zero_grad()
            self.critic.opt.zero_grad()
            loss.backward()
            self.actor.opt.step()
            self.critic.opt.step()
        self.reset_mem()

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
    ACTIONS = 6
    INPUT_DIMS = 3
    LR = 5e-4
    DISCOUNT_FACTOR = 0.99
    POLICY_CLIP = 0.1
    SMOOTHING = 0.95
    EPOCHS = 4
    BATCH_SIZE = 5
    LEARN_ITERS = 20

    rclpy.init()

    env = GzEnv()

    agent = Agent(ACTIONS, INPUT_DIMS, LR, DISCOUNT_FACTOR, POLICY_CLIP, SMOOTHING, EPOCHS, BATCH_SIZE, LEARN_ITERS)

    EPISODES = 5

    steps = 0
    nobs = 0

    for i in range(EPISODES):
        obs = torch.tensor(env.reset()[0])
        done = False
        score = 0
        while not done:
            action, prob, value = agent.act(obs)
            nobs, reward, done, _, _ = env.step(action)
            nobs = torch.tensor(nobs)
            agent.store_mem(obs, prob, action, reward, value, done)
            score += reward
            steps += 1
            obs = nobs
            if steps % agent.learn_iters == 0:
                agent.learn()
        print("Episode: {} Score: {}".format(i + 1, score))
    env.close()

    torch.save(agent.actor.state_dict(), 'actor.pth')