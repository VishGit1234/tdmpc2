import gymnasium as gym
import torch

class RepeatAction(gym.Wrapper):
  """Wrapper that repeats the action for a specified number of steps."""
  def __init__(self, env, scale_factor=0.05, num_repeats=50):
    super().__init__(env)
    self.num_repeats = num_repeats
    self.scale_factor = scale_factor
    self.action_space = env.action_space
    self.observation_space = env.observation_space
    self.frames = []
    self.is_rendered = False
    
  def step(self, action):
    action = action * self.scale_factor
    act = action.clone()
    done = False
    self.frames = []
    for _ in range(self.num_repeats):
      obs, reward, terminated, truncated, info = self.env.step(act)
      done = terminated | truncated
      if self.is_rendered:
        self.frames.append(self.env.render())  # Render the environment
      if torch.any(done):
        break
    return obs, reward, terminated, truncated, info
  
  def render(self):
    return self.frames