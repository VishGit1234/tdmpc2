import numpy as np
import sapien
import torch

from sapien.physx import PhysxRigidBodyComponent, PhysxRigidBaseComponent

from envs.kinova_envs.KinovaBaseEnv import KinovaBaseEnv

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor

@register_env("KinovaStackCube", max_episode_steps=15)
class KinovaStackCubeEnv(KinovaBaseEnv):
	def __init__(self, *args, robot_uids="kinova_gen3", **kwargs):
		kwargs["include_cubeB"] = True # Ensure cubeB is included in the environment
		super().__init__(*args, robot_uids=robot_uids, **kwargs)

	def compute_dense_reward(self, obs, action: torch.Tensor, info: dict):
		# reaching reward
		tcp_center_pose = self.agent.tcp.pose.p
		tcp_center_pose[:, 2] += 0.1 # the tcp is slightly below the end effector, so we raise it a bit
		cubeA_pos = self.cubeA.pose.p
		cubeA_to_tcp_dist = torch.linalg.norm(tcp_center_pose - cubeA_pos, axis=1)
		reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))

		# grasp and place reward
		cubeA_pos = self.cubeA.pose.p
		cubeB_pos = self.cubeB.pose.p
		goal_xyz = torch.hstack(
			[cubeB_pos[:, :2], (cubeB_pos[:, 2] + self.cubeA_half_sizes[:, 2] + self.cubeB_half_sizes[:, 2]).unsqueeze(-1)]
		)
		cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
		place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)

		reward[info["is_cubeA_grasped"]] = (4 + place_reward)[info["is_cubeA_grasped"]]

		# ungrasp reward
		gripper_width = 0.8178
		is_cubeA_grasped = info["is_cubeA_grasped"]
		ungrasp_reward = 1 - self.get_gripper_state()/gripper_width
		ungrasp_reward[~is_cubeA_grasped] = 1.0
		reward[info["is_cubeA_on_cubeB"]] = (
			6 + ungrasp_reward
		)[info["is_cubeA_on_cubeB"]]

		reward[info["_success"]] = 8

		return reward

	def evaluate(self):
		info = super().evaluate()
		pos_A = self.cubeA.pose.p
		pos_B = self.cubeB.pose.p
		offset = pos_A - pos_B
		xy_flag = (
			torch.linalg.norm(offset[..., :2], axis=1)
			<= torch.linalg.norm(self.cubeA_half_sizes[..., :2]) + 0.005
		)
		z_flag = torch.abs(offset[..., 2] - self.cubeA_half_sizes[..., 2] * 2) <= 0.005
		is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
		is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
		success = is_cubeA_on_cubeB * (~is_cubeA_grasped)
		info.update({
			"is_cubeA_on_cubeB": is_cubeA_on_cubeB,
			"_success": success.bool(),
		})
		return info