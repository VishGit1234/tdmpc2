import torch
import gymnasium as gym

class ScaleAction(gym.ActionWrapper):  
	def __init__(self, env, scale_factor=1, x_limits=[-0.4, 0.1], y_limits=[-0.6, 0.6]):
		super().__init__(env)
		self.scale_factor = scale_factor
		self.x_limits = x_limits
		self.y_limits = y_limits
		self.action_space = gym.spaces.Box(
			low=-1.0,
			high=1.0,
			shape=(2,) if len(self.env.action_space.shape) == 1 else (self.env.action_space.shape[0], 2),
			dtype=self.env.action_space.dtype
		)

	def action(self, action):
		x = self.env.unwrapped.agent.tcp_pos[:, 0]
		action[:, 0] = torch.where(torch.logical_and(self.x_limits[0] < x, x < self.x_limits[1]), action[:, 0], 0)
		y = self.env.unwrapped.agent.tcp_pos[:, 1]
		action[:, 1] = torch.where(torch.logical_and(self.y_limits[0] < y, y < self.y_limits[1]), action[:, 1], 0)
		# Add additional 5 dims for the other part of the action spaces
		action = torch.cat([action, torch.zeros(action.shape[0], 5, device=action.device)], dim=1)
		return action*self.scale_factor