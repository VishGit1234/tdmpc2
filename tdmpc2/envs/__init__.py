from copy import deepcopy
import warnings

import gymnasium as gym

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.tensor import TensorWrapper
from envs.wrappers.frame_stack import FrameStack

def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
	from envs.dmcontrol import make_env as make_dm_control_env
except:
	make_dm_control_env = missing_dependencies
try:
	from envs.maniskill import make_env as make_maniskill_env
except:
	make_maniskill_env = missing_dependencies
try:
	from envs.metaworld import make_env as make_metaworld_env
except:
	make_metaworld_env = missing_dependencies
try:
	from envs.myosuite import make_env as make_myosuite_env
except:
	make_myosuite_env = missing_dependencies
try:
	from envs.mujoco import make_env as make_mujoco_env
except:
	make_mujoco_env = missing_dependencies

from envs.kinova_env import KinovaPushCubeEnv

warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_multitask_env(cfg):
	"""
	Make a multi-task environment for TD-MPC2 experiments.
	"""
	print('Creating multi-task environment with tasks:', cfg.tasks)
	envs = []
	for task in cfg.tasks:
		_cfg = deepcopy(cfg)
		_cfg.task = task
		_cfg.multitask = False
		env = make_env(_cfg)
		if env is None:
			raise ValueError('Unknown task:', task)
		envs.append(env)
	env = MultitaskWrapper(cfg, envs)
	cfg.obs_shapes = env._obs_dims
	cfg.action_dims = env._action_dims
	cfg.episode_lengths = env._episode_lengths
	return env

import torch
class ScaleAction(gym.ActionWrapper):  
	def __init__(self, env, scale_factor=1, x_limits=[-0.4, 0.1], y_limits=[-0.6, 0.6]):
		super().__init__(env)
		self.scale_factor = scale_factor
		self.x_limits = x_limits
		self.y_limits = y_limits
		self.action_space = gym.spaces.Box(
			low=-1.0,
			high=1.0,
			# only x and y are allowed to move, the gripper is fixed
			shape=(self.env.action_space.shape[0],2),
			dtype=self.env.action_space.dtype
		)

	def action(self, action):
		x = self.env.agent.tcp_pos[:, 0]
		action[:, 0] = torch.where(torch.logical_and(self.x_limits[0] < x, x < self.x_limits[1]), action[:, 0], 0)
		y = self.env.agent.tcp_pos[:, 1]
		action[:, 1] = torch.where(torch.logical_and(self.y_limits[0] < y, y < self.y_limits[1]), action[:, 1], 0)
		# Add additional 5 dims for the other part of the action spaces
		action = torch.cat([action, torch.zeros(action.shape[0], 5, device=action.device)], dim=1)
		return action*self.scale_factor

def make_env(cfg):
	"""
	Make kinova environment for TD-MPC2 experiments.
	"""
	# Init kwargs dict
	kwargs = {
		"block_offset": cfg.init_box_pos,
		"block_gen_range": cfg.box_gen_range,
		"target_offset": cfg.target_pos,
		"goal_radius": cfg.termination_if_cube_goal_dist_less_than,
	}

	env = gym.make(
		"KinovaPushCube",
		num_envs=cfg.num_envs,
		render_mode="rgb_array",
		control_mode="pd_ee_delta_pose",
		**kwargs
	)

	env = ScaleAction(env, scale_factor=cfg.action_scale)  # Scale down the action space
	env = FrameStack(env, num_stack=cfg.obs_buffer_size)
	cfg.obs_shape = {cfg.get('obs', 'state'): (env.observation_space.shape[1], )}
	cfg.action_dim = env.action_space.shape[1]
	cfg.episode_length = 50 # manually set in KinovaPushCubeEnv
	cfg.seed_steps = max(1000, 5*cfg.episode_length) * cfg.num_envs
	return env

# def make_env(cfg):
# 	"""
# 	Make an environment for TD-MPC2 experiments.
# 	"""
# 	gym.logger.set_level(40)
# 	if cfg.multitask:
# 		env = make_multitask_env(cfg)

# 	else:
# 		env = None
# 		for fn in [make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env, make_mujoco_env]:
# 			try:
# 				env = fn(cfg)
# 			except ValueError:
# 				pass
# 		if env is None:
# 			raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
# 		env = TensorWrapper(env)
# 	try: # Dict
# 		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
# 	except: # Box
# 		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
# 	cfg.action_dim = env.action_space.shape[0]
# 	cfg.episode_length = env.max_episode_steps
# 	cfg.seed_steps = max(1000, 5*cfg.episode_length)
# 	return env
