from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from sanitizer_env import SanitizerWorld

# SanitizerWorld.reset = lambda self, **kwargs: self.env.reset(**kwargs)

env = SanitizerWorld()
model = DQN('MlpPolicy', env, verbose=1, exploration_final_eps=0.2)
model.learn(total_timesteps=50000, progress_bar=True)
model.save("dqn_sanitizer")
del model 
model = DQN.load("dqn_sanitizer", env=env)
# Set the number of episodes
num_episodes = 10
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, render=False,)

vec_env = model.get_env()
obs = vec_env.reset()
for episode in range(num_episodes):
   # Reset the environment for a new episode
   observation = env.reset()
   res = 0
   # Run the episode until it's done
   while True:
      # Render the environment (optional)
      env.render()
      
      # Take a random action (0 for left, 1 for right)
      action, _states = model.predict(obs, deterministic=True)
      observation, reward, done, _ = vec_env.step(action)
      res += reward
      print(res)
      # Perform the action and get the next state, reward, and done status
      #action = env.action_space.sample()
      #observation, reward, done, _ = env.step(action)
      
      # Break the loop if the episode is done
      if done:
         break

# Close the environment
env.close()
