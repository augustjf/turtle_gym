import numpy as np
import math
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from gymnasium.wrappers import FlattenObservation
from gymnasium.spaces.utils import flatten
import matplotlib.pyplot as plt
import sanitizer_env

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DQN(nn.Module):
    def __init__(self, input_shape, n_action):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*input_shape[1]*input_shape[2], 128) 
        self.fc2 = nn.Linear(128, n_action)
        self.dropout = nn.Dropout(p=0.5)
        self.saved_log_probs = []
        self.rewards = [1]

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * x.shape[1]*x.shape[2])
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        p = F.softmax(self.fc2(x), dim=1)
        return p
    
    
class Learner():
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.001
        self.LR = 1e-4
        self.NUM_EPISODES = 1000
        self.NUM_STEPS = 300
        self.LOG_INTERVAL = 10  
        self.EPS = np.finfo(np.float32).eps.item()
        self.PATH = 'dqn_sanitizer.pth'

        self.env = sanitizer_env.SanitizerWorld()
        
        self.obs_shape = self.env.observation_space.shape
        self.n_action = self.env.action_space.n
        self.dqn = DQN(self.obs_shape, self.n_action)
        self.dqn.to(torch.device(device))
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.LR)
        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        state.to(self.device)
        probs = self.dqn(state)
        m = Categorical(probs)
        # action = m.sample()
        action = torch.argmax(probs).unsqueeze(0)
        self.dqn.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def params_update(self):
        R = 0
        policy_loss = []
        returns = deque()
        for r in self.dqn.rewards[::-1]:
            R = r + self.GAMMA * R
            returns.appendleft(R)

        returns = torch.tensor(returns,dtype=torch.float32)
        returns.to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + self.EPS)
        
        for log_prob, R in zip(self.dqn.saved_log_probs, returns):
            if math.isnan(log_prob*R):
                policy_loss.append(torch.nan_to_num(log_prob * R))
            else:
                policy_loss.append(log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = -torch.cat(policy_loss).sum() 
        policy_loss.to(self.device)  
        policy_loss.backward()
        self.optimizer.step()
        del self.dqn.rewards[:]
        del self.dqn.saved_log_probs[:]

    def train(self):
        self.env.reset()
        running_reward = -200
        reward_list = []
        time_step_list = []
        avg_reward_list = []
        for n in range(self.NUM_EPISODES):
            obs = self.env.reset()
            ep_reward = 0
            for t in range(self.NUM_STEPS):
                action = self.select_action(obs)
                obs, reward, done, _, _ = self.env.step(action)
                self.dqn.rewards.append(reward)
                ep_reward += reward
                if done:
                    self.dqn.rewards.append(0) #To prevent nan in loss function
                    time_step_list.append(t)
                    break

            reward_list.append(ep_reward)

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            avg_reward_list.append(running_reward)
            self.params_update()
            if n % self.LOG_INTERVAL == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    n, ep_reward, running_reward))
            
            if running_reward > self.env.reward_threshold:
                print(f"Reward exceeded the threshold! Running reward is now {running_reward}, last episode duration: {t} time steps!")
                break
        
        # Save image of the reward list
        plt.figure()
        plt.plot(reward_list)
        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.savefig('learning_curve.png')
        # Save image of the time step list
        plt.figure()
        plt.plot(time_step_list)
        plt.ylabel('Time Steps')
        plt.xlabel('Episode')
        plt.savefig('time_steps_curve.png')
        # Save image of the average reward list
        plt.figure()
        plt.plot(avg_reward_list)
        plt.ylabel('Average Reward')
        plt.xlabel('Episode')
        plt.savefig('average_reward_curve.png')
        
        torch.save(self.dqn, self.PATH)


    def test(self):     
        total_reward = 0
        num_eval_episodes = 10
        self.dqn = DQN(self.obs_shape, self.n_action)
        self.dqn.to(torch.device(device))
        self.dqn = torch.load(self.PATH, map_location=torch.device(device))
        self.dqn.eval()

        input('Enter to start testing....')
    
        for _ in range(num_eval_episodes):
            obs = self.env.reset()
            print('New episode')
            while True:
                self.env.render()
                action = self.select_action(obs)
                obs, reward, done, _, info = self.env.step(action)
                total_reward += reward
                if done:
                    break
        self.env.close()

def main():
    learner = Learner()
    learner.train()
    learner.test()

main()
