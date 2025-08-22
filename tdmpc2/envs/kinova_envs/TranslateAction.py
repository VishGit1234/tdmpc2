import torch
import gymnasium as gym

# Translate Action from root frame to world frame
# And convert action from relative to absolute
class TranslateAction(gym.ActionWrapper):  
	def __init__(self, env, action_scale=0.05):
		super().__init__(env)
		shape = list(self.env.action_space.shape)
		shape[-1] = 4
		self.action_space = gym.spaces.Box(
			low=-1.0,
			high=1.0,
			shape=tuple(shape),
			dtype=self.env.action_space.dtype
		)
		self.root_pos = self.env.unwrapped.agent.robot.get_root().pose.p.clone()
		self.action_scale = action_scale
		self.gripper_scale = 0.4
		self.x_limits=[-0.4, 0.1]
		self.y_limits=[-0.6, 0.6]
		self.z_limits=[0.01, 0.6]

	def action(self, action):
		# Clamp action to [-1, 1]
		action = torch.clamp(action, min=-1.0, max=1.0)
		# Apply delta to current gripper position
		action[:, 3] = action[:, 3] * self.gripper_scale * 0.8178
		action[:, 3] = action[:, 3] + self.env.unwrapped.get_gripper_state()
		# Clamp gripper position to [0, 1]
		action[:, 3] = torch.clamp(action[:, 3], min=0.0, max=1.0)
		# Apply delta to current ee position
		action[:, :3] = action[:, :3] * self.action_scale
		action[:, :3] = action[:, :3] + self.env.unwrapped.agent.tcp.pose.p
		# Clamp ee position to workspace limits
		action[:, 0] = torch.clamp(action[:, 0], 
			min=torch.tensor(self.x_limits[0], device=action.device, dtype=action.dtype),
			max=torch.tensor(self.x_limits[1], device=action.device, dtype=action.dtype)
		)
		action[:, 1] = torch.clamp(action[:, 1], 
			min=torch.tensor(self.y_limits[0], device=action.device, dtype=action.dtype),
			max=torch.tensor(self.y_limits[1], device=action.device, dtype=action.dtype)
		)
		action[:, 2] = torch.clamp(action[:, 2], 
			min=torch.tensor(self.z_limits[0], device=action.device, dtype=action.dtype),
			max=torch.tensor(self.z_limits[1], device=action.device, dtype=action.dtype)
		)
		# Translate from world frame to root frame
		action[:, :3] = action[:, :3] - self.root_pos
		# Fix the rotation
		rot = torch.zeros(action.shape[0], 3, device=action.device, dtype=action.dtype)
		rot[:, 0] = torch.pi
		action = torch.cat([action[:, :3], rot, action[:, 3:]], dim=1)
		return action