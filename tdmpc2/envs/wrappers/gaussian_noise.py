import gymnasium as gym
import numpy as np
import torch

class GaussianObsNoise(gym.ObservationWrapper, gym.ActionWrapper):
  """
  Adds Gaussian noise to observations.
  """
  def __init__(self, env, mean=0.0, std=0.01, noise_indices=[]):
    super().__init__(env)
    self.mean = mean
    self.std = std
    self.noise_indices = noise_indices

  def observation(self, obs):
    if isinstance(obs, torch.Tensor):
      noise = torch.normal(mean=self.mean, std=self.std, size=obs.shape, device=obs.device, dtype=obs.dtype)
      noise[..., self.noise_indices] = 0.0
    else:
      noise = np.random.normal(self.mean, self.std, size=np.shape(obs))
      noise[..., self.noise_indices] = 0.0
    return obs + noise
