import gymnasium as gym
import numpy as np
import torch

class GaussianObsNoise(gym.ObservationWrapper, gym.ActionWrapper):
  """
  Adds Gaussian noise to observations.
  """
  def __init__(self, env, mean=0.0, std=0.01):
    super().__init__(env)
    self.mean = mean
    self.std = std
    self.noise_indices = [
      0, 1, 2, # ee position
      3, # gripper state
      8, 9, 10, 11, 12, 13, 14, 15, # cubeA position and orientation
    ]
    if env.unwrapped.include_cubeB:
      self.noise_indices += [19, 20, 21, 22, 23, 24, 25, 26]

  def observation(self, obs):
    if isinstance(obs, torch.Tensor):
      noise = torch.normal(mean=self.mean, std=self.std, size=obs.shape, device=obs.device, dtype=obs.dtype)
      noise[..., self.noise_indices] = 0.0
    else:
      noise = np.random.normal(self.mean, self.std, size=np.shape(obs))
      noise[..., self.noise_indices] = 0.0
    return obs + noise
