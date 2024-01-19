import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Dict, Discrete
import numpy as np
import matplotlib
import copy
import matplotlib.pyplot as plt
import time


import os
os.environ['DISPLAY'] = '$(awk \'/nameserver / {print $2; exit}\' /etc/resolv.conf 2>/dev/null):0'

# Use the Agg backend for Matplotlib
matplotlib.use('Agg')


class SanitizerWorld(gym.Env):
   def __init__(self, grid_size=[20, 30], 
                      render_available =True,
                      seed = 42):
      

      super(SanitizerWorld, self).__init__()

      # Copy input arguments to class variables
      self.seed = seed
      self.render_available = render_available
      self.grid_size = grid_size
      self.cell_size = 0.2

      # Set seed
      np.random.seed(seed)

      # Initialize the map, obstacles, and energy level
      self.map          = np.zeros(self.grid_size)
      self.obstacles    = np.zeros_like(self.map)
      self.energy_level = np.zeros_like(self.map)
      self.sanitized    = np.zeros_like(self.map)
      self.sum_sanitized      = 0
      self.reward_threshold   = 100
      self.time_step          = 15
      self.san_thresh         = 10*1e-3

      # Gen random starting pose in the map  
      self.current_pos = np.array((np.random.randint(0, self.grid_size[0]-1), 
                                   np.random.randint(0, self.grid_size[1]-1)))
      self.start_pos       = None
      self.new_pos         = None
      self.prev_direction  = None
      self.direction       = None

      # Define action and observation space
      self.action_space = Discrete(4+1)  # Up, Down, Left, Right, DoNothing
      # TODO: Define observation space
      # observation and action may vary depends on how we define the problem
      # observation is a key part of the problem definition because it defines the network as well
      self.observation_space = MultiDiscrete(np.array([grid_size[0]-1, grid_size[1]-1]))
      # self.observation_space = Dict(
      #    {
      #       "position" : MultiDiscrete(np.array([grid_size[0]-1, grid_size[1]-1])),
      #       "sanitized": MultiDiscrete(np.ones_like(self.map))
      #    })
      # Initialize the rendering if needed
      if self.render_available:
         self.fig, self.ax = plt.subplots()
         matrix = np.zeros_like(self.map)  # Initial random image data
         self.ax.set_xlim(-0.5,self.map.shape[1]-0.5)
         self.ax.set_ylim(-0.5,self.map.shape[0]-0.5)
         self.ax.set_xticks(np.arange(0,self.map.shape[1]-1,2))
         self.ax.set_yticks(np.arange(0,self.map.shape[0]-1,2))

         self.image = self.ax.imshow(matrix)
         # plt.ion()
         # plt.show()
         self.fig.savefig('demo.png', bbox_inches='tight')


   def reset(self, seed=42, options=None):
      """
      Reset the environment to the initial state.
      """
      super().reset(seed=seed, options=options)
      self.start_pos   = np.array((np.random.randint(0, self.grid_size[0]), 
                                   np.random.randint(0, self.grid_size[1])))
      self.current_pos = self.start_pos
      self.new_pos = self.current_pos
      self.obstacles = self._generate_obstacles()
      self.sanitized = np.zeros_like(self.map)
      self.sum_sanitized = 0
      self.energy_level = np.zeros_like(self.map)
      obs = self._get_observation()
      return obs

   def step(self, action):
      # Update position based on action
      self.start_pos = self.current_pos
      self.prev_direction = self.direction
      possible_new_pos = self._get_new_position(action)

      # Check if the new position is valid
      if self._is_valid_position(possible_new_pos):
         self.new_pos = copy.deepcopy(possible_new_pos)
         self.current_pos = copy.deepcopy(self.new_pos)
         self.direction = action
      self._update_energy_level()
      self.sanitized[self.energy_level > self.san_thresh] = 1

      done   = self._get_done()
      obs    = self._get_observation()
      reward = self._get_reward()
      truncated = False

      info = {}

      return obs, reward, done, truncated, info

   def _generate_obstacles(self):

      obstacles = np.zeros_like(self.map)

      # Generate random obstacles
      # if random p > 0.05, then obstacle
      # TODO: (Additional) Implement the logic to more complex obstacles (e.g. walls)
      #obstacles[np.random.random_sample(obstacles.shape) > 0.95] = 1
      for i in range(self.grid_size[0]):
         obstacles[i,0] = 1
         obstacles[i,-1] = 1
      for i in range(self.grid_size[1]):
         obstacles[0,i] = 1
         obstacles[-1,i] = 1

      return obstacles

   def _update_energy_level(self):
      """
      Update the energy level based on the current position.
      """
      # TODO: Implement the logic to update the energy level
      P = 100*1e-6
      new_energy_level = np.zeros_like(self.energy_level)
      for i in range(self.grid_size[0]):
         for j in range(self.grid_size[1]):
            if i == self.current_pos[0] and j == self.current_pos[1]:
               continue
            else:
               E = P*self.time_step / (self.cell_size*((i-self.current_pos[0])**2 + (j-self.current_pos[1])**2))
               new_energy_level[i,j] = E
      
      self.energy_level = self.energy_level + new_energy_level
      

   def _get_done(self):
      """
      Return 'done' signal based on the current position.
      This can be customized based on the specific done logic.
      """

      # TODO: Implement the logic to end the episode
      # the done signal can be defined in various ways:
      # Max number of steps reached ...
      # Agent reached the goal ...
      # Agent is not moving ...
      # Agent perform wrong movement ...

      # Return randomly true or false
      if np.random.random_sample() > 0.98:
         return True
      elif (self.grid_size[0]*self.grid_size[1] - np.sum(self.obstacles)) == self.sum_sanitized:
         return True
      else:
         return False

   def _get_observation(self):
      """
      Get the current observation of the environment.
      This can be customized based on the specific observation logic.
      """
      # TODO: Implement the logic to shape the observation
      # Observation defined the network input, so it's critical to define it properly
      # The observation should contain all the information the agent needs to make a decision
      # unnecessary information can be removed to reduce the network complexity
      # obs = {}
      # obs['position'] = self.current_pos
      # obs['sanitized'] = self.sanitized

      return self.current_pos

   def _get_reward(self):
      """
      Get the current reward based on the agent's position.
      This can be customized based on the specific reward logic.
      """
      # TODO: Implement the logic to shape the reward
      # Reward is the signal that the network is trying to maximize
      # Reward is a combination of positive and negative (penalizing) signals
      
      reward = np.sum(self.sanitized) - self.sum_sanitized #Num of new sanitized pixels
      self.sum_sanitized = np.sum(self.sanitized)
      reward -= 1 #Penalty for each step
      if self.prev_direction == self.direction:
         reward += 1 #Reward for moving in the same direction

      if self.new_pos[0] == self.start_pos[0] and self.new_pos[1] == self.start_pos[1]:
         reward = -5
      return reward

   def _get_new_position(self, action):
      """
      Get the new position based on the current position and action.
      """
      # Considering the action as a movement in the grid
      # TODO: (Additional) Implement the logic to more complex movements (e.g. diagonal)
      new_pos = copy.deepcopy(self.current_pos)
 
      if action == 1:  # Up
         new_pos[0] -= 1
      elif action == 3:  # Down
         new_pos[0] += 1
      elif action == 2:  # Left
         new_pos[1] -= 1
      elif action == 4:  # Right
         new_pos[1] += 1
      elif action == 0:  # DoNothing 
         pass
      
      new_pos[0] = np.clip(new_pos[0], 0, self.grid_size[0] - 1)
      new_pos[1] = np.clip(new_pos[1], 0, self.grid_size[1] - 1)

      return new_pos

   def _is_valid_position(self, pos):
      """
      Check if the given position is valid (not an obstacle and within bounds).
      """
      if pos[0] < 0 or pos[0] >= self.grid_size[0] or pos[1] < 0 or pos[1] >= self.grid_size[1]:
         print('Out of bounds')
         return False
      elif self.obstacles[pos[0], pos[1]] == 1:
         return False
      else:
         return True

   def update_render_data(self):
      matrix = np.ones((self.map.shape[0], self.map.shape[1], 3)) * 0.9

      # Make obstacles element black
      matrix[np.where(self.obstacles == 1)] = [0, 0, 0]  # Example: Setting the top-left element to red

      # Make sanitized pixels green
      san_mat = np.zeros_like(self.map)
      san_mat[np.where(self.sanitized == 1)] = 1
      san_mat[np.where(self.obstacles == 1)] = 0
      matrix[np.where(san_mat == 1)] = [0, 1, 0]

      # Make starting position element blue
      matrix[self.start_pos[0], self.start_pos[1]] = [0, 0, 1]

      # Make goal position element red
      matrix[self.new_pos[0], self.new_pos[1]] = [1, 0, 0]
      
      #TODO: Plot the reached energy level
      ...

      return matrix

   def render(self):
      """
      Render the current state of the environment.
      """
      # Example of the rendering.
      # More complex rendering can be implemented using the same logic.

      if not self.render_available:
         # Skip if render is not available
         print('Render not available')
         return 

      matrix = self.update_render_data()

      # Update the image with new data
      self.image.set_array(matrix)

      # Redraw the plot
      self.fig.canvas.draw()

      #plt.pause(0.1)
      self.fig.savefig('demo.png', bbox_inches='tight')
      time.sleep(0.1)
