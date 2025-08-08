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
  if cfg.task == 'kinova_push_cube' or cfg.task == 'kinova_pick_cube':
    # Init kwargs dict
    kwargs = {
      "block_offset": cfg.init_box_pos,
      "block_gen_range": cfg.box_gen_range,
      "target_offset": cfg.push_target_offset if cfg.task == "kinova_push_cube" else cfg.pick_target_offset,
      "goal_radius": cfg.termination_if_cube_goal_dist_less_than,
      "cube_randomization_ranges": cfg.cube_randomization_ranges,
    }
    task_name = 'KinovaPushCube' if cfg.task == 'kinova_push_cube' else 'KinovaPickCube'
    env = gym.make(
      task_name,
      num_envs=cfg.num_envs,
      render_mode="rgb_array",
      control_mode="pd_ee_delta_pose",
      **kwargs
    )
  return env
