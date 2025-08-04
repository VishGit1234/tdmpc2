import torch
import gymnasium as gym

class ScaleAction(gym.ActionWrapper):  
	def __init__(self, env, scale_factor=1, x_limits=[-0.4, 0.1], y_limits=[-0.6, 0.6], z_limits=[0.02, 0.6], gripper_control=True):
		super().__init__(env)
		self.scale_factor = scale_factor
		self.x_limits = x_limits
		self.y_limits = y_limits
		self.z_limits = z_limits
		self.gripper_control = gripper_control
		self.action_space = gym.spaces.Box(
			low=-1.0,
			high=1.0,
			shape=self.env.action_space.shape,
			dtype=self.env.action_space.dtype
		)

	def action(self, action):
		x = self.env.unwrapped.agent.tcp_pos[:, 0]
		action[:, 0] = torch.where(torch.logical_and(self.x_limits[0] < x, x < self.x_limits[1]), action[:, 0], 0)
		y = self.env.unwrapped.agent.tcp_pos[:, 1]
		action[:, 1] = torch.where(torch.logical_and(self.y_limits[0] < y, y < self.y_limits[1]), action[:, 1], 0)
		z = self.env.unwrapped.agent.tcp_pos[:, 2]
		action[:, 2] = torch.where(torch.logical_and(self.z_limits[0] < z, z < self.z_limits[1]), action[:, 2], 0)
		if not self.gripper_control:
			action[:, 3:] = 0
		return action*self.scale_factor