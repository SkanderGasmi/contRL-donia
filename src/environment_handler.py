import gym
import numpy as np

class EnvironmentHandler:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.reset()
    
    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return next_obs, reward, done, info

    def collect_observations(self, num_steps):
        observations = []
        for _ in range(num_steps):
            action = self.env.action_space.sample()
            obs, _, _, _ = self.step(action)
            observations.append(obs)
        
        # Convert to numpy array for consistency with ML frameworks
        observations = np.array(observations)
        return observations
