from typing import Any, Dict

import numpy as np
import sapien
import torch

from sapien.physx import PhysxRigidBodyComponent, PhysxRigidBaseComponent

from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from transforms3d.euler import euler2quat
from mani_skill.utils.structs.types import Array

@register_env("KinovaPickCube", max_episode_steps=50)
class KinovaPickCubeEnv(PickCubeEnv):
    SUPPORTED_ROBOTS = [
        "kinova_gen3",
    ]
    def __init__(self, *args, robot_uids="kinova_gen3", **kwargs):
        self.block_offset = kwargs["block_offset"]
        self.block_gen_range = kwargs["block_gen_range"]
        self.target_offset = kwargs["target_offset"]
        self.goal_radius = kwargs["goal_radius"]

        self.cube_rand_ranges = kwargs["cube_randomization_ranges"]
        self.cube_size_range = self.cube_rand_ranges["size"]
        self.dynamic_friction_range = self.cube_rand_ranges["dynamic_friction"]
        self.static_friction_range = self.cube_rand_ranges["static_friction"]
        self.restitution_range = self.cube_rand_ranges["restitution"]
        self.mass_range = self.cube_rand_ranges["mass"]
        
        del kwargs["block_offset"]
        del kwargs["block_gen_range"]
        del kwargs["target_offset"]
        del kwargs["goal_radius"]
        del kwargs["cube_randomization_ranges"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
      # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
      pose = sapien_utils.look_at([-0.1, 1.7, 1.2], [-0.1, 0.8, 0.35])
      return CameraConfig(
          "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
      )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # we then add the cube that we want to pick and give it a color and size using a convenience build_cube function
        # we specify the body_type to be "dynamic" as it should be able to move when touched by other objects / the robot
        # finally we specify an initial pose for the cube so that it doesn't collide with other objects initially
        max_size, min_size = self.cube_size_range
        max_size = torch.tensor(max_size)
        min_size = torch.tensor(min_size)
        objects = []
        cube_size = torch.rand(3) * (max_size - min_size) + min_size
        cube_half_sizes = cube_size/2
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            # ignore the randomization in height
            cube_half_sizes[2] = self.cube_half_size
            builder.add_box_collision(half_size=cube_half_sizes)
            builder.set_scene_idxs([i])
            builder.add_box_visual(
                half_size=cube_half_sizes,
                material=sapien.render.RenderMaterial(
                    base_color=np.array([12, 42, 160, 255]) / 255
                )
            )
            builder.set_initial_pose(sapien.Pose(p=[0, 0, cube_half_sizes[2]]))
            obj = builder.build(name = f"object_{i}")
            self.remove_from_state_dict_registry(obj)
            objects.append(obj)
        self.obj = Actor.merge(objects, name = "cube")
        self.add_to_state_dict_registry(self.obj)

         # Randomize object physical and collision properties
        for i, obj in enumerate(self.obj._objs):
            # modify the i-th object which is in parallel environment i

            # modifying physical properties e.g. randomizing mass from 0.1 to 1kg
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

        # we also add in red/white sphere target to visualize where we want the cube to be picked to
        # we specify add_collisions=False as we only use this as a visual for videos and do not want it to affect the actual physics
        # we finally specify the body_type to be "kinematic" so that the object stays in place
        # goal_thresh = goal_radius

        self.goal_region = actors.build_sphere(
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
        self._hidden_objects.append(self.goal_region)

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

            # here we write some randomization code that randomizes the x, y position of the cube we are picking in the desired range
            xyz = torch.zeros((b, 3))
            block_gen_range = torch.tensor(self.block_gen_range)
            block_offset = torch.tensor(self.block_offset)
            xyz[..., :2] = torch.rand((b, 2)) * block_gen_range + (block_offset - block_gen_range/2)
            xyz[..., 2] = self.cube_half_size
            q = [1, 0, 0, 0]
            # we can then create a pose object using Pose.create_from_pq to then set the cube pose with. Note that even though our quaternion
            # is not batched, Pose.create_from_pq will automatically batch p or q accordingly
            # furthermore, notice how here we do not even use env_idx as a variable to say set the pose for objects in desired
            # environments. This is because internally any calls to set data on the GPU buffer (e.g. set_pose, set_linear_velocity etc.)
            # automatically are masked so that you can only set data on objects in environments that are meant to be initialized
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            # here we set the location of that red/white target (the goal region). In particular here, we set the position to be a desired given position
            # and we further rotate 90 degrees on the y-axis to make the target object face up
            target_region_xyz = xyz.clone()
            target_region_xyz[..., :3] += torch.tensor(self.target_offset)
            # set random z that should be within 3d range
            target_region_xyz[..., 2] = torch.rand(b) * block_gen_range
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )
        # set the keyframe for the robot
        self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)

    def _get_obs_extra(self, info: Dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.p,
        )
        if self.obs_mode_struct.use_state:
            # if the observation mode requests to use state, we provide ground truth information about where the cube is.
            # for visual observation modes one should rely on the sensed visual data to determine where the cube is
            obs.update(
                goal_pos=self.goal_region.pose.p,
                obj_pose=self.obj.pose.raw_pose,
            )
        return obs

    def _get_obs_state_dict(self, info: Dict):
        """Get (ground-truth) state-based observations."""
        return dict(
            # agent=self._get_obs_agent(),
            extra=self._get_obs_extra(info),
        )

    def evaluate(self):
        info = super().evaluate()
        if "success" in info:
            info["_success"] = info.pop("success")
            info["terminated"] = False
        return info
    
    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # pose to pick cube
        tcp_to_pick_pose = self.obj.pose.p - self.agent.tcp.pose.p
        tcp_to_pick_pose_dist = torch.linalg.norm(tcp_to_pick_pose, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_pick_pose_dist)
        reward = reaching_reward

        # compute a placement reward to encourage robot to move the cube to the center of the goal region
        # we further multiply the place_reward by a mask reached so we only add the place reward if the robot has reached the desired pick pose
        # This reward design helps train RL agents faster by staging the reward out.
        reached = tcp_to_pick_pose_dist < 0.01
        obj_to_goal_dist = torch.linalg.norm(
            self.obj.pose.p - self.goal_region.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * reached

        # assign rewards to parallel environments that achieved success to the maximum of 3.
        reward[info["_success"]] = 4
        return reward
    
    def render(self):
        if self.render_mode == "rgb_array":
            # Return the rendered image from the first environment in the batch
            return super().render()[0].cpu().numpy()
        else:
            return super().render()
