from copy import deepcopy
import warnings

import gymnasium as gym

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.tensor import TensorWrapper
from envs.wrappers.frame_stack import FrameStack
from envs.wrappers.gaussian_noise import GaussianObsNoise
from envs.kinova_envs.ClipAction import ClipAction
from envs.kinova_envs.RepeatAction import RepeatAction

def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

from envs.kinova_env import make_env as make_kinova_env

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

def wrap_env(cfg, env):
	env = GaussianObsNoise(env, std=cfg.noise_std)  # Add Gaussian noise to observations
	env = FrameStack(env, num_stack=cfg.obs_buffer_size)
	env = ClipAction(env)  # Scale down the action space
	env = RepeatAction(env, num_repeats=cfg.action_repeat, scale_factor=cfg.action_scale)  # Repeat actions
	return env

def make_env(cfg):
	"""
	Make kinova environment for TD-MPC2 experiments.
	"""
	env, eval_env = make_kinova_env(cfg)
	env = wrap_env(cfg, env)
	cfg.obs_shape = {cfg.get('obs', 'state'): (env.observation_space.shape[1], )}
	cfg.action_dim = env.action_space.shape[1]
	cfg.seed_steps = max(1000, 5*cfg.episode_length) * cfg.num_envs
	eval_env = wrap_env(cfg, eval_env)
	return env, eval_env

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
