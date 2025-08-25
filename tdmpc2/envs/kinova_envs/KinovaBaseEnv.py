from abc import ABC, abstractmethod

import torch
import numpy as np

import sapien
from sapien.physx import PhysxRigidBodyComponent, PhysxRigidBaseComponent

from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import SimConfig

class KinovaBaseEnv(StackCubeEnv, ABC):
	SUPPORTED_ROBOTS = [
		"kinova_gen3",
	]
	
	def __init__(self, *args, robot_uids="kinova_gen3", **kwargs):
		self.cubeA_init_pos = kwargs["cubeA_init_pos"]
		self.cubeA_gen_range = kwargs["cubeA_gen_range"]
		self.cubeB_offset = kwargs["cubeB_offset"]
		self.cubeB_gen_range = kwargs["cubeB_gen_range"]

		self.cube_rand_ranges = kwargs["cube_randomization_ranges"]
		self.cube_size_range = self.cube_rand_ranges["size"]
		self.dynamic_friction_range = self.cube_rand_ranges["dynamic_friction"]
		self.static_friction_range = self.cube_rand_ranges["static_friction"]
		self.restitution_range = self.cube_rand_ranges["restitution"]
		self.mass_range = self.cube_rand_ranges["mass"]

		self.include_cubeB = kwargs.get("include_cubeB", False)
		
		sim_config = SimConfig(control_freq=kwargs.get("control_freq", 2))

		del kwargs["cubeA_init_pos"]
		del kwargs["cubeA_gen_range"]
		del kwargs["cube_randomization_ranges"]
		del kwargs["cubeB_offset"]
		del kwargs["cubeB_gen_range"]
		kwargs.pop("control_freq", None)
		kwargs.pop("include_cubeB", None)
		super().__init__(*args, robot_uids=robot_uids, sim_config=sim_config, **kwargs)

	@property
	def _default_sensor_configs(self):
		return []

	@property
	def _default_human_render_camera_configs(self):
		# registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
		pose = sapien_utils.look_at([0.6, 1.0, 0.6], [0.0, 0.3, 0.35])
		return CameraConfig(
			"render_camera", pose=pose, width=256, height=256, fov=1, near=0.01, far=100
		)

	def _load_scene(self, options: dict):
		# we use a prebuilt scene builder class that automatically loads in a floor and table.
		self.table_scene = TableSceneBuilder(
				env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
		)
		self.table_scene.build()

		min_size, max_size = self.cube_size_range
		max_size = torch.tensor(max_size)
		min_size = torch.tensor(min_size)

		# Create the cubeA boxes for the task
		cubeA_objects = []
		self.cubeA_half_sizes = torch.zeros((self.num_envs, 3), device=self.device)
		for i in range(self.num_envs):
			# CubeA
			builderA = self.scene.create_actor_builder()
			cubeA_size = torch.rand(3) * (max_size - min_size) + min_size
			cubeA_half_sizes = cubeA_size / 2
			builderA.add_box_collision(half_size=cubeA_half_sizes)
			builderA.set_scene_idxs([i])
			builderA.add_box_visual(
				half_size=cubeA_half_sizes,
				material=sapien.render.RenderMaterial(
					base_color=np.array([12, 42, 160, 255]) / 255
				)
			)
			builderA.set_initial_pose(sapien.Pose(p=[0, 0, cubeA_half_sizes[2]]))
			objA = builderA.build(name=f"cubeA_{i}")
			self.remove_from_state_dict_registry(objA)
			cubeA_objects.append(objA)
			self.cubeA_half_sizes[i] = cubeA_half_sizes

		self.cubeA = Actor.merge(cubeA_objects, name="cubeA")
		self.add_to_state_dict_registry(self.cubeA)

		# Randomize object physical and collision properties
		for i, obj in enumerate(self.cubeA._objs):
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
		
		# Add cubeB if the environment requires it
		if self.include_cubeB:	
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
			cubeA_xy = xy.clone()

			xyz[:, :2] = cubeA_xy
			xyz[:, 2] = self.cubeA_half_sizes[:, 2]
			self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone()))

			if self.include_cubeB:
				cubeB_xy = xy.clone()
				cubeB_gen_range = torch.tensor(self.cubeB_gen_range)
				cubeB_offset = torch.tensor(self.cubeB_offset)
				cubeB_xy[:, :2] += torch.rand((b, 2)) * cubeB_gen_range + (cubeB_offset - cubeB_gen_range/2)
				xyz[:, :2] = cubeB_xy
				xyz[:, 2] = self.cubeB_half_sizes[:, 2]
				self.cubeB.set_pose(Pose.create_from_pq(p=xyz))

			# set the keyframe for the robot
			self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)

	def _get_obs_state_dict(self, info: dict):
		"""Get (ground-truth) state-based observations."""
		return dict(
			# agent=self._get_obs_agent(),
			extra=self._get_obs_extra(info),
		)
	
	def get_gripper_state(self):
		return self.agent.robot.joints_map["left_outer_knuckle_joint"].qpos

	def _get_obs_extra(self, info: dict):
		# these obs are common to push, pick, and stack
		obs = dict(
			tcp_pose=self.agent.tcp.pose.p,
			gripper_state=torch.clamp(self.get_gripper_state()/0.8178, min=0.0, max=1.0),  # normalize gripper state to [0, 1]
			is_cubeA_grasped=info["is_cubeA_grasped"],
			goal_pos=torch.zeros((self.num_envs, 3), device=self.device),	# placeholder, to be filled in by subclasses if needed
			cubeA_pose=self.cubeA.pose.raw_pose,
			tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
		)
		# these obs are only relevant if cubeB is part of the environment
		if self.include_cubeB:
			obs.update(
				cubeB_pose=self.cubeB.pose.raw_pose,
				tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
				cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
			)
		return obs

	def render(self):
		if self.render_mode == "rgb_array":
			# Return the rendered image from the first environment in the batch
			return super().render()[0].cpu().numpy()
		else:
			return super().render()

	@abstractmethod
	def evaluate(self):
		return {
			"is_cubeA_grasped": self.agent.is_grasping(self.cubeA),
			"terminated": False
		}
	
	@abstractmethod
	def compute_dense_reward(self, obs, action: torch.Tensor, info: dict):
		raise NotImplementedError("Dense reward computation is not implemented for KinovaBaseEnv. Please implement this method in the subclass.")

	