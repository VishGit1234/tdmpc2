import gymnasium as gym
import torch
from random import random

class RepeatAction(gym.Wrapper):
  """Wrapper that repeats the action for a specified number of steps."""
  def __init__(self, env, max_repeats=50, episode_length=20):
    super().__init__(env)
    self.max_repeats = max_repeats
    self.action_space = env.action_space
    self.observation_space = env.observation_space
    self.frames = []
    self._is_rendered = False
    self.episode_length = episode_length
    self.step_count = 0

  def set_rendered(self, value):
    self._is_rendered = value

  def get_rendered(self):
    return self._is_rendered

  def reset(self, **kwargs):
    self.step_count = 0
    self.frames = []
    return self.env.reset(**kwargs)

  def step(self, action):
    done = False
    self.frames = []
    target_pos = action[:, :4].clone()
    for i in range(self.max_repeats):
      obs, reward, terminated, truncated, info = self.env.step(action)
      terminated = torch.ones_like(terminated)*(self.step_count >= self.episode_length)
      done = terminated | truncated
      if self._is_rendered:
        self.frames.append(self.env.render())  # Render the environment
      # Stop the gripper if the gripper target is reached
      if torch.any(done) or torch.all(torch.norm(obs[:, :4] - target_pos, dim=1) < 0.01):
        break
    self.step_count += 1
    return obs, reward, terminated, truncated, info
  
  def render(self):
    return self.frames