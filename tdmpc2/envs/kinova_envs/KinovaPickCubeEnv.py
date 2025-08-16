import numpy as np
import sapien
import torch

from envs.kinova_envs.KinovaBaseEnv import KinovaBaseEnv

from mani_skill.utils.building import actors

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

@register_env("KinovaPickCube", max_episode_steps=200)
class KinovaPickCubeEnv(KinovaBaseEnv):
	def __init__(self, *args, robot_uids="kinova_gen3", **kwargs):
		self.target_offset = kwargs["target_offset"]
		self.goal_radius = kwargs["goal_radius"]

		del kwargs["target_offset"]
		del kwargs["goal_radius"]
		super().__init__(*args, robot_uids=robot_uids, **kwargs)

	def _load_scene(self, options: dict):
		super()._load_scene(options)
		
		# we also add in green sphere target to visualize where we want the cube to be lifted to
		# we specify add_collisions=False as we only use this as a visual for videos and do not want it to affect the actual physics
		# we finally specify the body_type to be "kinematic" so that the object stays in place
		# goal_thresh = goal_radius

		self.goal_site = actors.build_sphere(
			self.scene,
			radius=self.goal_radius,
			color=[0, 1, 0, 1],
			name="goal_site",
			body_type="kinematic",
			add_collision=False,
			initial_pose=sapien.Pose(),
		)

		# optionally you can automatically hide some Actors from view by appending to the self._hidden_objects list. When visual observations
		# are generated or env.render_sensors() is called or env.render() is called with render_mode="sensors", the actor will not show up.
		# This is useful if you intend to add some visual goal sites as e.g. done in PickCube that aren't actually part of the task
		# and are there just for generating evaluation videos.
		self._hidden_objects.append(self.goal_site)

	def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
		super()._initialize_episode(env_idx, options)

		# use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
		with torch.device(self.device):
			# here we set the location of the target (sphere)
			# we set it as a fixed displacement above where the cube was spawned
			target_region_xyz = self.cubeA.pose.p.clone()
			target_region_xyz[..., :3] += torch.tensor(self.target_offset)
			self.goal_site.set_pose(
				Pose.create_from_pq(
					p=target_region_xyz,
					q=[1, 0, 0, 0],
				)
			)
			# store initial block position for computing rewards
			self.initial_block_pos = self.cubeA.pose.p.clone()

	def evaluate(self):
		info = super().evaluate()
		is_obj_placed = (
			torch.linalg.norm(self.goal_site.pose.p - self.cubeA.pose.p, axis=1)
			<= self.goal_radius
		)
		info.update({
			"_success": is_obj_placed,
			"goal_pos": self.goal_site.pose.p,
		})
		return info

	def compute_dense_reward(self, obs, action: torch.Tensor, info: dict):
		tcp_center_pose = self.agent.tcp.pose.p
		tcp_center_pose[:, 2] += 0.1 # the tcp is slightly below the end effector, so we raise it a bit
		tcp_to_obj_dist = torch.linalg.norm(
			self.cubeA.pose.p - tcp_center_pose, axis=1
		)
		reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
		reward = reaching_reward

		is_grasped = info["is_cubeA_grasped"]
		reward += is_grasped

		obj_to_goal_dist = torch.linalg.norm(
			self.goal_site.pose.p - self.cubeA.pose.p, axis=1
		)
		place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
		reward += place_reward * is_grasped

		reward[info["_success"]] = 5
		return reward
