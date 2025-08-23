from envs.wrappers.frame_stack import FrameStack
from envs.wrappers.gaussian_noise import GaussianObsNoise
from envs.kinova_envs.TranslateAction import TranslateAction
from envs.kinova_envs.RepeatAction import RepeatAction
import gymnasium as gym
import torch
import time
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from envs.kinova_envs import KinovaPickCubeEnv

from transforms3d.euler import quat2euler

render_mode = "rgb_array"  # "human" for interactive display, "rgb_array" for video
# render_mode = None
control_freq = 2  # Control frequency in Hz
action_scale = 0.05
gripper_scale = 0.4
control_mode = "pd_ee_pose"

# Init kwargs dict
push_cube_kwargs = {
  "control_freq": control_freq,
  "render_mode": render_mode,
  "control_mode": control_mode,
  # Initial positions and generation ranges for cubes
  "cubeA_init_pos": [-0.3, 0.2],
  "cubeA_gen_range": [0.1, 0.1],
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
  "control_freq": control_freq,
  "render_mode": render_mode,
  "control_mode": control_mode,
  # Initial positions and generation ranges for cubes
  "cubeA_init_pos": [-0.3, 0.2],
  "cubeA_gen_range": [0.1, 0.1],
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
  "control_freq": control_freq,
  "render_mode": render_mode,
  "control_mode": control_mode,
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
env = gym.make("KinovaPickCube", num_envs=num_envs, **pick_cube_kwargs)
# env = GaussianObsNoise(env, std=0.005)  # Add Gaussian noise to observations
# env = FrameStack(env, num_stack=5)
# env = RepeatAction(env, max_repeats=20, episode_length=20)
env = TranslateAction(env, action_scale=action_scale, gripper_scale=gripper_scale)
# env.unwrapped.print_sim_details()
# env.set_rendered(True)  # Enable rendering in RepeatAction wrapper
obs, _ = env.reset(seed=0)
done = False
start_time = time.time()
total_rew = 0
frames = []
step_count = 0
original_obs = obs[0, :4].clone()
while not done or render_mode == "human":
  print(f"Step {step_count}")
  print(f"Observation: {obs[0, :4]}")
  action_value = [0., 0., 0., 0.] 
  if num_envs == 1:
    action = torch.from_numpy(env.action_space.sample()).to(env.get_wrapper_attr('device')) # torch.tensor([action_value], device=env.get_wrapper_attr('device'))
    obs, rew, terminated, truncated, info = env.step(action)
  else:
    # action = torch.from_numpy(env.action_space.sample()).to(env.get_wrapper_attr('device')) # torch.tensor([action_value] * num_envs, device=env.get_wrapper_attr('device'))
    action = torch.tensor([action_value] * num_envs, device=env.get_wrapper_attr('device'))
    obs, rew, terminated, truncated, info = env.step(action)
  frame = env.render()
  frames.append(frame)
  done = (terminated | truncated).any() # stop if any environment terminates/truncates
  step_count += 1
final_obs = obs[0, :4].clone()
print(f"Initial pos: {original_obs}")
print(f"Final pos: {final_obs}")
N = num_envs * step_count
dt = time.time() - start_time
FPS = N / (dt)
print(f"Frames Per Second = {N} / {dt} = {FPS}")
if render_mode == "rgb_array":
    ImageSequenceClip(frames, fps=2).write_videofile("output_video.mp4", codec="libx264")
