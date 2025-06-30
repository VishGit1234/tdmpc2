from envs.kinova_env import KinovaPushCubeEnv
import gymnasium as gym
import torch
import time

# Init kwargs dict
kwargs = {
    "block_offset": [-0.3, 0.2],
    "block_gen_range": [0.1, 0.1],
    "target_offset": [0., 0.2],
    "goal_radius": 0.1,
}
num_envs = 256
env = gym.make("KinovaPushCube", num_envs=num_envs, control_mode="pd_ee_delta_pose", render_mode="rgb_array", **kwargs)
env.unwrapped.print_sim_details()
obs, _ = env.reset(seed=0)
done = False
start_time = time.time()
total_rew = 0
frames = []
while not done:
    # note that env.action_space is now a batched action space
    obs, rew, terminated, truncated, info = env.step(torch.from_numpy(env.action_space.sample()))
    done = (terminated | truncated).any() # stop if any environment terminates/truncates
N = num_envs * info["elapsed_steps"][0].item()
dt = time.time() - start_time
FPS = N / (dt)
print(f"Frames Per Second = {N} / {dt} = {FPS}")

print("Now with rendering...")

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
obs, _ = env.reset(seed=0)
done = False
start_time = time.time()
total_rew = 0
frames = []
while not done:
    # note that env.action_space is now a batched action space
    obs, rew, terminated, truncated, info = env.step(torch.from_numpy(env.action_space.sample()))
    frame = env.render()
    frames.append(frame)
    done = (terminated | truncated).any() # stop if any environment terminates/truncates
N = num_envs * info["elapsed_steps"][0].item()
dt = time.time() - start_time
FPS = N / (dt)
print(f"Frames Per Second = {N} / {dt} = {FPS}")
ImageSequenceClip(frames, fps=30).write_videofile("output_video.mp4", codec="libx264")
