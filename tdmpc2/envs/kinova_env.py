from envs.kinova_envs import SUPPORTED_TASKS
from envs.kinova_envs.ScaleAction import ScaleAction
import gymnasium as gym

def make_env(cfg):
  """
  Make Kinova Maniskill environment.
  """
  if cfg.task not in SUPPORTED_TASKS:
    raise ValueError('Unknown task:', cfg.task)
  kwargs = {}
  if cfg.task == 'kinova_push_cube' or cfg.task == 'kinova_pick_cube' or cfg.task == 'kinova_stack_cube':
    if cfg.task == 'kinova_push_cube':
      task_name = "KinovaPushCube"
      target_offset = cfg.push_target_offset
    elif cfg.task == 'kinova_pick_cube':
      task_name = "KinovaPickCube"
      target_offset = cfg.pick_target_offset
    elif cfg.task == 'kinova_stack_cube':
      task_name = "KinovaStackCube"
      target_offset = cfg.stack_target_offset
    # Init kwargs dict
    kwargs = {
      "block_offset": cfg.init_box_pos,
      "block_gen_range": cfg.box_gen_range,
      "target_offset": target_offset,
      "goal_radius": cfg.termination_if_cube_goal_dist_less_than,
      "cube_randomization_ranges": cfg.cube_randomization_ranges,
    }
    env = gym.make(
      task_name,
      num_envs=cfg.num_envs,
      render_mode="rgb_array",
      control_mode="pd_ee_delta_pose",
      **kwargs
    )
    eval_env = gym.make(
      task_name,
      num_envs=cfg.num_eval_envs,
      render_mode="rgb_array",
      control_mode="pd_ee_delta_pose",
      **kwargs
    )
  return env, eval_env
