from envs.wrappers.frame_stack import FrameStack
from envs.wrappers.gaussian_noise import GaussianObsNoise
from envs.wrappers.repeat_action import RepeatAction
from envs.kinova_envs.ScaleAction import ScaleAction
import gymnasium as gym
import torch
import time
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from envs.kinova_envs import KinovaMultitaskEnv

render_mode = "rgb_array"  # "human" for interactive display, "rgb_array" for video

# Init kwargs dict
push_cube_kwargs = {
  "render_mode": render_mode,
  "control_mode": "pd_ee_delta_pose",
  # Initial positions and generation ranges for cubes
  "cubeA_init_pos": [-0.3, 0.2],
  "cubeA_gen_range": [0.1, 0.1],
  "cubeB_offset": [-0.2, 0.2],
  "cubeB_gen_range": [0.1, 0.1],
  # Task and goal parameters
  "target_offset": [0., 0.2],  # push cube task
  "goal_radius": 0.05,
  # Cube randomization ranges
  "cube_randomization_ranges": {
    "size": ([0.04, 0.04, 0.04], [0.07, 0.07, 0.07]),
    "dynamic_friction": (0.1, 0.3),
    "static_friction": (0.1, 0.3),
    "restitution": (0.1, 0.3),
    "mass": (0.02, 0.06)
  }
}
pick_cube_kwargs = {
  "render_mode": render_mode,
  "control_mode": "pd_ee_delta_pose",
  # Initial positions and generation ranges for cubes
  "cubeA_init_pos": [-0.3, 0.2],
  "cubeA_gen_range": [0.1, 0.1],
  "cubeB_offset": [-0.2, 0.2],
  "cubeB_gen_range": [0.1, 0.1],
  # Task and goal parameters
  "target_offset": [0., 0., 0.2],  # push cube task
  "goal_radius": 0.05,
  # Cube randomization ranges
  "cube_randomization_ranges": {
    "size": ([0.04, 0.04, 0.04], [0.07, 0.07, 0.07]),
    "dynamic_friction": (0.1, 0.3),
    "static_friction": (0.1, 0.3),
    "restitution": (0.1, 0.3),
    "mass": (0.02, 0.06)
  }
}
stack_cube_kwargs = {
  "render_mode": render_mode,
  "control_mode": "pd_ee_delta_pose",
  # Initial positions and generation ranges for cubes
  "cubeA_init_pos": [-0.3, 0.2],
  "cubeA_gen_range": [0.1, 0.1],
  "cubeB_offset": [-0.2, 0.2],
  "cubeB_gen_range": [0.1, 0.1],
  # Cube randomization ranges
  "cube_randomization_ranges": {
    "size": ([0.04, 0.04, 0.04], [0.07, 0.07, 0.07]),
    "dynamic_friction": (0.1, 0.3),
    "static_friction": (0.1, 0.3),
    "restitution": (0.1, 0.3),
    "mass": (0.02, 0.06)
  }
} 
kwargs = {
  "push_cube_kwargs": push_cube_kwargs,
  "pick_cube_kwargs": pick_cube_kwargs,
  "stack_cube_kwargs": stack_cube_kwargs  
}
num_envs = 2 if render_mode != "human" else 1
env = KinovaMultitaskEnv(num_envs=num_envs, **kwargs)
env = GaussianObsNoise(env, std=0.01)  # Add Gaussian noise to observations
env = FrameStack(env, num_stack=5)
env = ScaleAction(env, scale_factor=0.2)  # Scale down the action space
env = RepeatAction(env, repeat=10)  # Repeat actions
env.print_sim_details()
env.is_rendered = True  # Enable rendering in RepeatAction wrapper
obs, _ = env.reset(seed=0)
done = False
start_time = time.time()
total_rew = 0
frames = []
step_count = 0
original_obs = obs[0, :4].clone()
while not done or render_mode == "human":
  print(f"Step {step_count}")
  if step_count == 0:
    action_value = [1, 1, 1, 1]  # Move down for first 7 steps
  else:
    action_value = [0, 0, 0, 0]
  if num_envs == 1:
    action = torch.tensor([action_value], device=env.get_wrapper_attr('device'))
    obs, rew, terminated, truncated, info = env.step(action)
  else:
    action = torch.tensor([action_value] * num_envs, device=env.get_wrapper_attr('device'))
    obs, rew, terminated, truncated, info = env.step(action)
  frame = env.render()
  frames.extend(frame)
  done = (terminated | truncated).any() # stop if any environment terminates/truncates
  step_count += 1
final_obs = obs[0, :4].clone()
print(f"Total translation: {final_obs - original_obs}")
N = num_envs * step_count
dt = time.time() - start_time
FPS = N / (dt)
print(f"Frames Per Second = {N} / {dt} = {FPS}")
if render_mode == "rgb_array":
    ImageSequenceClip(frames, fps=30).write_videofile("output_video.mp4", codec="libx264")
