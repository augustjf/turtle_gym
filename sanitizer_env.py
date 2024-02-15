import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Dict, Discrete, MultiBinary
import numpy as np
import matplotlib
import copy
import matplotlib.pyplot as plt
import time


matplotlib.use('Agg')

class SanitizerWorld(gym.Env):
   def __init__(self, grid_size=[10, 15], 
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
      self.reward_threshold   = 500
      self.time_step          = 5
      self.san_thresh         = 10*1e-3

      # Gen random starting pose in the map  
      self.current_pos = np.array((np.random.randint(1, self.grid_size[0]-1), 
                                   np.random.randint(1, self.grid_size[1]-1)))
      self.current_pos = np.array([1,1])
      
      self.start_pos       = None
      self.new_pos         = None
      self.prev_direction  = None
      self.direction       = None

      # Define action and observation space
      self.action_space = Discrete(4+1)  # Up, Down, Left, Right, DoNothing

      self.observation_space = gym.spaces.Box(low=-1, high=self.san_thresh, shape=(1, grid_size[0]-2, grid_size[1]-2), dtype=np.float32)

      # Initialize the rendering if needed
      if self.render_available:
         self.fig, self.ax = plt.subplots()
         matrix = np.zeros_like(self.map)  # Initial random image data
         self.ax.set_xlim(-0.5,self.map.shape[1]-0.5)
         self.ax.set_ylim(-0.5,self.map.shape[0]-0.5)
         self.ax.set_xticks(np.arange(0,self.map.shape[1]-1,2))
         self.ax.set_yticks(np.arange(0,self.map.shape[0]-1,2))

         self.image = self.ax.imshow(matrix)

         self.fig.savefig('demo.png', bbox_inches='tight')


   def reset(self, seed=42, options=None):
      """
      Reset the environment to the initial state.
      """
      super().reset(seed=seed, options=options)
      self.start_pos   = np.array((np.random.randint(1, self.grid_size[0]-1), 
                                   np.random.randint(1, self.grid_size[1]-1)))
      self.start_pos = np.array([1,1])
   
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
         if action != 0:
            self.direction = action
      else:
         self.new_pos = copy.deepcopy(self.current_pos)
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
      P = 100*1e-6
      new_energy_level = np.zeros_like(self.energy_level)
      for i in range(1,self.grid_size[0]-1):
         for j in range(1,self.grid_size[1]-1):
            if i == self.current_pos[0] and j == self.current_pos[1]:
               continue
            else:
               E = P*self.time_step / (self.cell_size*((i-self.current_pos[0])**2 + (j-self.current_pos[1])**2))
               new_energy_level[i,j] = E
      self.energy_level = self.energy_level + new_energy_level

   def _get_done(self):
      if (self.sanitized[1:-1, 1:-1].max())>1:
         print("ERROR: Sanitized map has values greater than 1")
      if ((self.grid_size[0]-2)*(self.grid_size[1]-2)) <= self.sum_sanitized:
         return True
      else:
         return False

   def _get_observation(self):
      """
      Get the current observation of the environment.
      This can be customized based on the specific observation logic.
      """

      # Make observation a copy of the self.sanitized, excluding the borders
      obs = copy.deepcopy(self.energy_level[1:-1, 1:-1])
      # Set current position in the map to -1
      obs[self.current_pos[0]-1, self.current_pos[1]-1] = -1
      return obs

   def _get_reward(self):
      """
      Get the current reward based on the agent's position.
      This can be customized based on the specific reward logic.
      """
      reward = 0

      # Reward the robot for sanitizing the map (exploration)
      reward += (np.sum(self.sanitized) - self.sum_sanitized) #Num of new sanitized pixels

      self.sum_sanitized = np.sum(self.sanitized[1:-1, 1:-1])

      # Reward the robot for reaching the goal 
      if self._get_done():
         reward += 100

      reward -= 1 #Penalty for each step

      return reward

   def _get_new_position(self, action):
      """
      Get the new position based on the current position and action.
      """
      # Considering the action as a movement in the grid
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

