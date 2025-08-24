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
		self.cubeB_offset = kwargs["cubeB_offset"]
		self.cubeB_gen_range = kwargs["cubeB_gen_range"]

		del kwargs["cubeB_offset"]
		del kwargs["cubeB_gen_range"]
		super().__init__(*args, robot_uids=robot_uids, **kwargs)

	def _load_scene(self, options: dict):
		super()._load_scene(options)

		min_size, max_size = self.cube_size_range
		max_size = torch.tensor(max_size)
		min_size = torch.tensor(min_size)

		# create the cubeB box for the task (box that is stacked on)
		cubeB_objects = []
		self.cubeB_half_sizes = torch.zeros((self.num_envs, 3), device=self.device)
		for i in range(self.num_envs):
			# CubeB
			builderB = self.scene.create_actor_builder()
			cubeB_size = torch.rand(3) * (max_size - min_size) + min_size
			cubeB_half_sizes = cubeB_size / 2
			builderB.add_box_collision(half_size=cubeB_half_sizes)
			builderB.set_scene_idxs([i])
			builderB.add_box_visual(
				half_size=cubeB_half_sizes,
				material=sapien.render.RenderMaterial(
					base_color=np.array([160, 12, 42, 255]) / 255
				)
			)
			builderB.set_initial_pose(sapien.Pose(p=[1, 0, cubeB_half_sizes[2]]))
			objB = builderB.build(name=f"cubeB_{i}")
			self.remove_from_state_dict_registry(objB)
			cubeB_objects.append(objB)
			self.cubeB_half_sizes[i] = cubeB_half_sizes

		self.cubeB = Actor.merge(cubeB_objects, name="cubeB")
		self.add_to_state_dict_registry(self.cubeB)

		# Randomize object physical and collision properties
		for i, obj in enumerate(self.cubeB._objs):
			# modify the i-th object which is in parallel environment i
			rigid_body_component: PhysxRigidBodyComponent = obj.find_component_by_type(PhysxRigidBodyComponent)
			if rigid_body_component is not None:
				# note the use of _batched_episode_rng instead of torch.rand. _batched_episode_rng helps ensure reproducibility in parallel environments.
				min_mass, max_mass = self.mass_range
				rigid_body_component.mass = torch.rand(1).item() * (max_mass - min_mass) + min_mass

			# modifying per collision shape properties such as friction values
			rigid_base_component: PhysxRigidBaseComponent = obj.find_component_by_type(PhysxRigidBaseComponent)
			for shape in rigid_base_component.collision_shapes:
				min_dyn_fric, max_dyn_fric = self.dynamic_friction_range
				shape.physical_material.dynamic_friction = torch.rand(1).item() * (max_dyn_fric - min_dyn_fric) + min_dyn_fric

				min_static_fric, max_static_fric = self.static_friction_range
				shape.physical_material.static_friction = torch.rand(1).item() * (max_static_fric - min_static_fric) + min_static_fric

				min_restitution, max_restitution = self.restitution_range
				shape.physical_material.restitution = torch.rand(1).item() * (max_restitution - min_restitution) + min_restitution

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
			cubeA_gen_range = torch.tensor(self.cubeA_gen_range)
			cubeA_init_pos = torch.tensor(self.cubeA_init_pos)
			xy = torch.rand((b, 2)) * cubeA_gen_range + (cubeA_init_pos - cubeA_gen_range/2)
			cubeA_xy = xy
			cubeB_xy = xy.clone()
			cubeB_gen_range = torch.tensor(self.cubeB_gen_range)
			cubeB_offset = torch.tensor(self.cubeB_offset)
			cubeB_xy[:, :2] += torch.rand((b, 2)) * cubeB_gen_range + (cubeB_offset - cubeB_gen_range/2)

			xyz[:, :2] = cubeA_xy
			xyz[:, 2] = self.cubeA_half_sizes[:, 2]
			self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone()))

			xyz[:, :2] = cubeB_xy
			xyz[:, 2] = self.cubeB_half_sizes[:, 2]
			self.cubeB.set_pose(Pose.create_from_pq(p=xyz))
			# set the keyframe for the robot
			self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)

	def _get_obs_extra(self, info: dict):
		obs = super()._get_obs_extra(info)
		obs["cubeB_pose"] = self.cubeB.pose.raw_pose
		obs["tcp_to_cubeB_pos"] = self.cubeB.pose.p - self.agent.tcp.pose.p
		obs["cubeA_to_cubeB_pos"] = self.cubeB.pose.p - self.cubeA.pose.p
		return obs

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