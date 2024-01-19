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

import sanitizer_env

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class DQN(nn.Module):
    def __init__(self, n_obs, n_action):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(n_obs, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, n_action)
        self.dropout = nn.Dropout(p=0.6) # Used to prevent overfitting
        self.saved_log_probs = []
        self.rewards = [1]
        

    def forward(self, x):
        # Goes through the network and returns the action to take
        x = self.l1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.l3(x)
        p = F.softmax(x, dim=1)
        return p
    
class Learner():
    def __init__(self):
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-5
        self.NUM_EPISODES = 10000
        self.NUM_STEPS = 1000
        self.LOG_INTERVAL = 10  
        self.EPS = np.finfo(np.float32).eps.item()
        self.PATH = 'dqn_sanitizer.pth'

        #self.env = FlattenObservation(sanitizer_env.SanitizerWorld())
        self.env = sanitizer_env.SanitizerWorld()
        self.n_obs = self.env.observation_space.shape[0]
        self.n_action = self.env.action_space.n
        self.dqn = DQN(self.n_obs, self.n_action)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.LR)
        

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.dqn(state)
        m = Categorical(probs)
        action = m.sample()
        self.dqn.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def params_update(self):
        R = 0
        policy_loss = []
        returns = deque()
        for r in self.dqn.rewards[::-1]:
            R = r + self.GAMMA * R
            returns.appendleft(R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.EPS)
        
        for log_prob, R in zip(self.dqn.saved_log_probs, returns):
            if math.isnan(log_prob*R):
                policy_loss.append(torch.nan_to_num(log_prob * R))
            else:
                policy_loss.append(log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = -torch.cat(policy_loss).sum()   
        policy_loss.backward()
        self.optimizer.step()
        del self.dqn.rewards[:]
        del self.dqn.saved_log_probs[:]

    def train(self):
        self.env.reset()
        running_reward = 10
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
                    break

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            self.params_update()
            if n % self.LOG_INTERVAL == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    n, ep_reward, running_reward))
            
            if running_reward > self.env.reward_threshold:
                print(f"Reward exced the threhsold! Running reward is now {running_reward}, last episode duration: {t} time steps!")
                break
        torch.save(self.dqn.state_dict(), self.PATH)


    def test(self):     
        total_reward = 0
        num_eval_episodes = 10
        model = DQN(self.n_obs, self.n_action)
        model.load_state_dict(torch.load(self.PATH))
        model.eval()

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
