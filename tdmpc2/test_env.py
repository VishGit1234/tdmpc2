from envs.kinova_envs.KinovaPushCubeEnv import KinovaPushCubeEnv
from envs.wrappers.frame_stack import FrameStack
from envs.wrappers.gaussian_noise import GaussianObsNoise
from envs.wrappers.repeat_action import RepeatAction
from envs.kinova_envs.ScaleAction import ScaleAction
import gymnasium as gym
import torch
import time
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

# Init kwargs dict
kwargs = {
    "block_offset": [-0.3, 0.2],
    "block_gen_range": [0.1, 0.1],
    # "target_offset": [0., 0.2],
    "target_offset": [0., 0., 0.2],
    "goal_radius": 0.05,
    "cube_randomization_ranges": {
        "size": ([0.04, 0.04, 0.04], [0.07, 0.07, 0.07]),
        "dynamic_friction": (0.1, 0.3),
        "static_friction": (0.1, 0.3),
        "restitution": (0.1, 0.3),
        "mass": (0.9, 1.0)
    }
}
# render_mode = "human"
render_mode = "rgb_array"
num_envs = 2 if render_mode != "human" else 1
env = gym.make("KinovaPickCube", num_envs=num_envs, control_mode="pd_ee_delta_pose", render_mode=render_mode, **kwargs)
env = GaussianObsNoise(env, std=0.01)  # Add Gaussian noise to observations
env = FrameStack(env, num_stack=10)
env = ScaleAction(env, scale_factor=0.1)  # Scale down the action space
env = RepeatAction(env, repeat=10)  # Repeat actions
env.unwrapped.print_sim_details()
env.is_rendered = True  # Enable rendering in RepeatAction wrapper
obs, _ = env.reset(seed=0)
done = False
start_time = time.time()
total_rew = 0
frames = []
while not done:
    # note that env.action_space is now a batched action space
    if num_envs == 1:
        action = torch.tensor([[0, 0, 0, -1]], device=env.get_wrapper_attr('device'))
        obs, rew, terminated, truncated, info = env.step(action)
    else:
        action = torch.tensor([[0, 0, 0, -1]] * num_envs, device=env.get_wrapper_attr('device'))
        obs, rew, terminated, truncated, info = env.step(action)
    frame = env.render()
    frames.extend(frame)
    done = (terminated | truncated).any() # stop if any environment terminates/truncates
N = num_envs * info["elapsed_steps"][0].item()
dt = time.time() - start_time
FPS = N / (dt)
print(f"Frames Per Second = {N} / {dt} = {FPS}")
ImageSequenceClip(frames, fps=30).write_videofile("output_video.mp4", codec="libx264")
