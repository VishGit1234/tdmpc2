from envs.kinova_envs import SUPPORTED_TASKS
from envs.kinova_envs.ScaleAction import ScaleAction
import gymnasium as gym

from envs.kinova_envs.KinovaMultitaskEnv import KinovaMultitaskEnv

def make_env(cfg):
  """
  Make Kinova Maniskill environment.
  """
  if cfg.task not in SUPPORTED_TASKS:
    raise ValueError('Unknown task:', cfg.task)

  # Environment options that are common across tasks
  general_kwargs = {
    "render_mode": "rgb_array",
    "control_mode": "pd_ee_delta_pose",
    # Initial positions and generation ranges for cubes
    "cubeA_init_pos": cfg.cubeA_init_pos,
    "cubeA_gen_range": cfg.cubeA_gen_range,
    "cubeB_offset": cfg.cubeB_offset,
    "cubeB_gen_range": cfg.cubeB_gen_range,
    # Cube randomization ranges
    "cube_randomization_ranges": cfg.cube_randomization_ranges
  }
  
  if cfg.task == "kinova_multitask":
    push_cube_kwargs = dict(general_kwargs, target_offset=cfg.push_target_offset, goal_radius=cfg.push_goal_radius)
    pick_cube_kwargs = dict(general_kwargs, target_offset=cfg.pick_target_offset, goal_radius=cfg.pick_goal_radius)
    stack_cube_kwargs = dict(general_kwargs)
    kwargs = {
      "push_cube_kwargs": push_cube_kwargs,
      "pick_cube_kwargs": pick_cube_kwargs,
      "stack_cube_kwargs": stack_cube_kwargs
    }
    env = KinovaMultitaskEnv(num_envs = cfg.num_envs, **kwargs)
    eval_kwargs = {
      "push_cube_kwargs": push_cube_kwargs,
      "pick_cube_kwargs": pick_cube_kwargs,
      "stack_cube_kwargs": stack_cube_kwargs,
    }
    eval_env = KinovaMultitaskEnv(num_envs = cfg.num_eval_envs, **eval_kwargs)
  else:
    if cfg.task == 'kinova_push_cube':
      task_name = "KinovaPushCube"
      kwargs = dict(general_kwargs, target_offset=cfg.push_target_offset, goal_radius=cfg.push_goal_radius)
    elif cfg.task == 'kinova_pick_cube':
      task_name = "KinovaPickCube"
      kwargs = dict(general_kwargs, target_offset=cfg.pick_target_offset, goal_radius=cfg.pick_goal_radius)
    elif cfg.task == 'kinova_stack_cube':
      task_name = "KinovaStackCube"
    env = gym.make(task_name, num_envs = cfg.num_envs, **kwargs)
    eval_env = gym.make(task_name, num_envs = cfg.num_eval_envs, **kwargs)

  return env, eval_env
