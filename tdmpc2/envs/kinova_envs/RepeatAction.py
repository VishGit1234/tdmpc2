import gymnasium as gym
import torch
from random import random

class RepeatAction(gym.Wrapper):
  """Wrapper that repeats the action for a specified number of steps."""
  def __init__(self, env, scale_factor=0.05, num_repeats=50, action_noise=0.2):
    super().__init__(env)
    self.num_repeats = num_repeats
    self.original_scale_factor = scale_factor
    self.scale_factor = scale_factor
    self.action_space = env.action_space
    self.observation_space = env.observation_space
    self.frames = []
    self.is_rendered = False
    self.action_noise = action_noise

  def reset(self, **kwargs):
    self.scale_factor = self.original_scale_factor + self.action_noise * (random() - 0.5)
    return self.env.reset(**kwargs)
    
  def step(self, action):
    action = action * self.scale_factor
    done = False
    self.frames = []
    for _ in range(self.num_repeats):
      obs, reward, terminated, truncated, info = self.env.step(action)
      done = terminated | truncated
      if self.is_rendered:
        self.frames.append(self.env.render())  # Render the environment
      if torch.any(done):
        break
    return obs, reward, terminated, truncated, info
  
  def render(self):
    return self.frames