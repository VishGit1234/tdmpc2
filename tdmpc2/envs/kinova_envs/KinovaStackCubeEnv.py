from typing import Any, Dict

import numpy as np
import sapien
import torch

from mani_skill.utils import sapien_utils
from mani_skill.envs.utils import randomization
from mani_skill.utils.building import actors
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.structs import Pose
from transforms3d.euler import euler2quat
from mani_skill.utils.structs.types import Array


@register_env("KinovaPushCube", max_episode_steps=50)
class KinovaStackCubeEnv(StackCubeEnv):
  SUPPORTED_ROBOTS = [
    "kinova_gen3",
  ]

  def __init__(self, *args, robot_uids="kinova_gen3", **kwargs):
    self.block_offset = kwargs["block_offset"]
    self.block_gen_range = kwargs["block_gen_range"]
    self.target_offset = kwargs["target_offset"]
    self.goal_radius = kwargs["goal_radius"]
    self.cube_half_sizes = kwargs["cube_half_sizes"]
    
    del kwargs["block_offset"]
    del kwargs["block_gen_range"]
    del kwargs["target_offset"]
    del kwargs["goal_radius"]
    del kwargs["cube_half_sizes"]
    super().__init__(*args, robot_uids=robot_uids, **kwargs)

  def _load_scene(self, options: dict):
    # we use a prebuilt scene builder class that automatically loads in a floor and table.
    self.table_scene = TableSceneBuilder(
        env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
    )
    self.table_scene.build()

    # we then add the cube that we want to pick and give it a color and size using a convenience build_cube function
    # we specify the body_type to be "dynamic" as it should be able to move when touched by other objects / the robot
    # finally we specify an initial pose for the cube so that it doesn't collide with other objects initially
    self.cubeA = actors.build_box(
        self.scene,
        half_sizes=self.cube_half_sizes,
        color=np.array([12, 42, 160, 255]) / 255,
        name="cube",
        body_type="dynamic",
        initial_pose=sapien.Pose(p=[0, 0, self.cube_half_sizes[2]]),
    )

    self.cubeB = actors.build_box(
        self.scene,
        half_sizes=self.cube_half_sizes,
        color=np.array([160, 12, 42, 255]) / 255,
        name="cube",
        body_type="dynamic",
        initial_pose=sapien.Pose(p=[1, 0, self.cube_half_sizes[2]]),
    )

  @property
  def _default_human_render_camera_configs(self):
      # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
      pose = sapien_utils.look_at([-0.1, 1.7, 1.2], [-0.1, 0.8, 0.35])
      return CameraConfig(
          "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
      )

  def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
    with torch.device(self.device):
      # the initialization functions where you as a user place all the objects and initialize their properties
      # are designed to support partial resets, where you generate initial state for a subset of the environments.
      # this is done by using the env_idx variable, which also tells you the batch size
      b = len(env_idx)
      # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
      # note that the table scene is built such that z=0 is the surface of the table.
      self.table_scene.initialize(env_idx)

      # here we write some randomization code that randomizes the x, y position of the cube we are pushing in the desired range
      xyz = torch.zeros((b, 3))
      xyz[..., 2] = self.cube_half_size
      block_gen_range = torch.tensor(self.block_gen_range)
      block_offset = torch.tensor(self.block_offset)
      xy = torch.rand((b, 2)) * block_gen_range + (block_offset - block_gen_range/2)
      region = [[-self.block_gen_range, self.block_gen_range], [-self.block_gen_range, self.block_gen_range]]
      sampler = randomization.UniformPlacementSampler(
        bounds=region, batch_size=b, device=self.device
      )
      # we add the last 0.001 to ensure they don't overlap
      radius = torch.linalg.norm(torch.tensor([self.cube_half_sizes, self.cube_half_sizes])) + 0.001
      cubeA_xy = xy + sampler.sample(radius, 100)
      cubeB_xy = xy + sampler.sample(radius, 100)
      
      xyz[:, :2] = cubeA_xy
      qs = randomization.random_quaternions(
            b,
            lock_x=True,
            lock_y=True,
            lock_z=False,
      )
      self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

      xyz[:, :2] = cubeB_xy
      qs = randomization.random_quaternions(
            b,
            lock_x=True,
            lock_y=True,
            lock_z=False,
      )
      self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))
      # set the keyframe for the robot
      self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)

  def evaluate(self):
    info = super().evaluate()
    if "success" in info:
      info["_success"] = info.pop("success")
    info["terminated"] = False
    return info

  def _get_obs_state_dict(self, info: Dict):
    """Get (ground-truth) state-based observations."""
    return dict(
        # agent=self._get_obs_agent(),
        extra=self._get_obs_extra(info),
    )
  
  def render(self):
    if self.render_mode == "rgb_array":
      # Return the rendered image from the first environment in the batch
      return super().render()[0].cpu().numpy()
    else:
      return super().render()
