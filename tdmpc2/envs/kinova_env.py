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
  if cfg.task == 'kinova_push_cube':
    # Init kwargs dict
    kwargs = {
      "block_offset": cfg.init_box_pos,
      "block_gen_range": cfg.box_gen_range,
      "target_offset": cfg.target_pos,
      "goal_radius": cfg.termination_if_cube_goal_dist_less_than,
      "cube_randomization_ranges": cfg.cube_randomization_ranges,
    }

  env = gym.make(
    "KinovaPushCube",
    num_envs=cfg.num_envs,
    render_mode="rgb_array",
    control_mode="pd_ee_delta_pose",
    **kwargs
  )
  env = ScaleAction(env, scale_factor=cfg.action_scale, gripper_control=False)  # Scale down the action space
  return env
