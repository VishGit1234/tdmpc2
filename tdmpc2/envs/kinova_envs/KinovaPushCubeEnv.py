from typing import Any, Dict

import numpy as np
import sapien
import torch

from envs.kinova_envs.KinovaBaseEnv import KinovaBaseEnv

from mani_skill.utils.building import actors

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from transforms3d.euler import euler2quat
from mani_skill.utils.structs.types import Array

@register_env("KinovaPushCube", max_episode_steps=200)
class KinovaPushCubeEnv(KinovaBaseEnv):
	def __init__(self, *args, robot_uids="kinova_gen3", **kwargs):
		self.target_offset = kwargs["target_offset"]
		self.goal_radius = kwargs["goal_radius"]

		del kwargs["target_offset"]
		del kwargs["goal_radius"]
		super().__init__(*args, robot_uids=robot_uids, **kwargs)

	def _load_scene(self, options: dict):
		super()._load_scene(options)

		# we also add in red/white target to visualize where we want the cube to be pushed to
		# we specify add_collisions=False as we only use this as a visual for videos and do not want it to affect the actual physics
		# we finally specify the body_type to be "kinematic" so that the object stays in place
		self.goal_region = actors.build_red_white_target(
				self.scene,
				radius=self.goal_radius,
				thickness=1e-5,
				name="goal_region",
				add_collision=False,
				body_type="kinematic",
				initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
		)

	def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
		super()._initialize_episode(env_idx, options)
		
		# use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
		with torch.device(self.device):
			# here we set the location of that red/white target (the goal region). In particular here, we set the position to be a desired given position
			# and we further rotate 90 degrees on the y-axis to make the target object face up
			target_region_xyz = self.cubeA.pose.p.clone()
			target_region_xyz[..., :2] += torch.tensor(self.target_offset)
			# set a little bit above 0 so the target is sitting on the table
			target_region_xyz[..., 2] = 1e-3
			self.goal_region.set_pose(
				Pose.create_from_pq(
					p=target_region_xyz,
					q=euler2quat(0, np.pi / 2, 0),
				)
			)

	def evaluate(self):
		info = super().evaluate()
		# success is achieved when the cube's xy position on the table is within the
		# goal region's area (a circle centered at the goal region's xy position) and
		# the cube is still on the surface
		is_obj_placed = (
			torch.linalg.norm(
				self.cubeA.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
			) < self.goal_radius
		) & (self.cubeA.pose.p[..., 2] < self.cubeA_half_sizes[:, 2] + 5e-3)
		info.update({
			"_success": is_obj_placed,
			"goal_pos": self.goal_region.pose.p,
		})
		return info

	def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
		# We also create a pose marking where the robot should push the cube from that is easiest (pushing from behind the cube)
		offsets = torch.zeros_like(self.cubeA_half_sizes, device=self.device)
		offsets[:, 1] = -self.cubeA_half_sizes[:, 1] - 0.005
		tcp_push_pose = Pose.create_from_pq(
				p=self.cubeA.pose.p
				# Note the below line will change based on where the cube is placed
				+ offsets
		)
		tcp_to_push_pose = tcp_push_pose.p - self.agent.tcp.pose.p
		tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
		reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)
		reward = reaching_reward

		# compute a placement reward to encourage robot to move the cube to the center of the goal region
		# we further multiply the place_reward by a mask reached so we only add the place reward if the robot has reached the desired push pose
		# This reward design helps train RL agents faster by staging the reward out.
		reached = tcp_to_push_pose_dist < 0.01
		obj_to_goal_dist = torch.linalg.norm(
				self.cubeA.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
		)
		place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
		reward += place_reward * reached

		# assign rewards to parallel environments that achieved success to the maximum of 3.
		reward[info["_success"]] = 4
		return reward
