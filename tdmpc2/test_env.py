from envs.kinova_env import KinovaEnv
import genesis as gs
import torch
class EnvConfig:
  def __init__(self):
    self.episode_length_s = 3
    self.init_joint_angles = [6.9761, 1.1129, 1.7474, -2.2817, 7.5884, -1.1489, 1.6530, 0.8213, 0.8200, 0.8209, 0.8208, 0.8217, 0.8210]
    self.init_quat = [0, 0, 0, 1]
    self.bracelet_link_height = 0.25
    self.init_box_pos = [0.2, 0.2, 0.002]
    self.box_size = [0.08, 0.08, 0.02]
    self.clip_actions = 0.01
    self.termination_if_cube_goal_dist_less_than = 0.01
    self.cube_goal_dist_rew_scale = 10
    self.cube_arm_dist_rew_scale = 10
    self.success_reward = 1
    self.target_displacement = 0.3
    self.action_scale = 0.05
    self.prev_obs_buffer = 3

# Create an instance of the configuration
env_cfg = EnvConfig()

# initialize genesis
gs.init(
  backend=gs.gpu,
  # logging_level="warning"
)
num_envs = 32

# Create the environment
env = KinovaEnv(num_envs, env_cfg)

# Run some steps
obs = env.reset()
frames = []
for i in range(3):
  done = torch.zeros(num_envs, dtype=torch.bool)
  while not torch.any(done):
    action = env.rand_act()
    obs, reward, done, info = env.step(action)
    frames.append(env.render())
    # print(f"Obs: {obs}, Reward: {reward}, Done: {done}, Info: {info}")

# Save the frames as a video
import imageio.v2 as imageio
imageio.mimwrite('kinova_env_video.mp4', frames, fps=30)