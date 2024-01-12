import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib
import copy
import matplotlib.pyplot as plt


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
      ...

      # Set seed
      np.random.seed(seed)

      # Initialize the map, obstacles, and energy level
      self.map          = np.zeros(self.grid_size)
      self.obstacles    = np.zeros_like(self.map)
      self.energy_level = np.zeros_like(self.map)
      self.sanitized    = np.zeros_like(self.map)
      self.sum_sanitized = 0

      # Gen random starting pose in the map  
      self.current_pos = np.array((np.random.randint(0, self.grid_size[0]-1), 
                                   np.random.randint(0, self.grid_size[1]-1)))
      self.start_pos = None
      self.new_pos   = None

      # Define action and observation space
      self.action_space = spaces.Discrete(4+1)  # Up, Down, Left, Right, DoNothing
      # TODO: Define observation space
      # observation and action may vary depends on how we define the problem
      # observation is a key part of the problem definition because it defines the network as well
      self.observation_space = spaces.MultiDiscrete([grid_size[0]-1, grid_size[1]-1])
      
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


   def reset(self, seed=42):
      """
      Reset the environment to the initial state.
      """
      super().reset(seed=seed)
      self.start_pos   = np.array((np.random.randint(0, self.grid_size[0]), 
                                   np.random.randint(0, self.grid_size[1])))
      self.current_pos = self.start_pos
      self.new_pos = self.current_pos
      self.obstacles = self._generate_obstacles()
      self.sanitized = np.zeros_like(self.map)
      self.energy_level = np.zeros_like(self.map)
      obs = self._get_observation()
      return obs

   def step(self, action):
      """
      Take a step in the environment based on the given action.

      Parameters:
      - action (int): The action to take (0: Up, 1: Down, 2: Left, 3: Right).

      Returns:
      - observation: Agent's observation of the current environment.
      - reward (float): Amount of reward returned after the step.
      - done (bool): Whether the episode has ended.
      - info (dict): Additional information.
      """
      # Update position based on action
      self.start_pos = self.current_pos
      possible_new_pos = self._get_new_position(action)

      # Check if the new position is valid
      if self._is_valid_position(possible_new_pos):
         self.new_pos = copy.deepcopy(possible_new_pos)
         self.current_pos = copy.deepcopy(self.new_pos)
      
      self.sanitized[self.start_pos[0], self.start_pos[1]] = 1

      done   = self._get_done()
      obs    = self._get_observation()
      reward = self._get_reward()
      truncated = False

      # Info is an empty dict for now, you can use to pass additional information for example for logging
      info = {}

      return obs, reward, done, truncated, info


   def _generate_obstacles(self):

      obstacles = np.zeros_like(self.map)

      # Generate random obstacles
      # if random p > 0.05, then obstacle
      # TODO: (Additional) Implement the logic to more complex obstacles (e.g. walls)
      obstacles[np.random.random_sample(obstacles.shape) > 0.95] = 1

      return obstacles

   def _update_energy_level(self):
      """
      Update the energy level based on the current position.
      """
      # TODO: Implement the logic to update the energy level
      new_energy_level = self.energy_level
      new_energy_level = new_energy_level + np.zeros_like(new_energy_level)

      return new_energy_level

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
      return self.current_pos

   def _get_reward(self):
      """
      Get the current reward based on the agent's position.
      This can be customized based on the specific reward logic.
      """
      # TODO: Implement the logic to shape the reward
      # Reward is the signal that the network is trying to maximize
      # Reward is a combination of positive and negative (penalizing) signals
      
      reward = 0
      if np.sum(self.sanitized) > self.sum_sanitized:
         reward = 1
      self.sum_sanitized = np.sum(self.sanitized)
      if self.new_pos[0] == self.start_pos[0] and self.new_pos[1] == self.start_pos[1]:
         reward = -10
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
      matrix[np.where(self.sanitized == 1)] = [0, 1, 0]

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
