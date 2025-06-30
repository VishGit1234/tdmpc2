from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, DictControllerConfig, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.controllers.base_controller import ControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.sensors.camera import CameraConfig

from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.structs import Pose
from transforms3d.euler import euler2quat
from mani_skill.utils.structs.types import Array


@register_agent()
class KinovaGen3(BaseAgent):
  uid = "kinova_gen3"
  urdf_path = "/home/vishal/Documents/tdmpc2/tdmpc2/envs/kinova_gen3/Gen3-with-gripper.urdf"
  disable_self_collisions = True
  urdf_config = dict(
    _materials=dict(
      gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
    ),
    link=dict(
      left_inner_finger_pad=dict(
        material="gripper", patch_radius=0.1, min_patch_radius=0.1
      ),
      right_inner_finger_pad=dict(
        material="gripper", patch_radius=0.1, min_patch_radius=0.1
      ),
    ),
  )
  # List of real joint names from the combined URDF (arm + gripper)
  arm_joint_names = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
  ]
  gripper_joint_names = [
    "left_outer_knuckle_joint",
    "right_outer_knuckle_joint",
    "left_inner_knuckle_joint",
    "right_inner_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
  ]
  ee_link_name = "end_effector_link"

  arm_stiffness = 1e3
  arm_damping = 1e2
  arm_force_limit = 100

  gripper_stiffness = 1e3
  gripper_damping = 1e2
  gripper_force_limit = 100
  
  keyframes = dict(
    rest=Keyframe(
      qpos=np.array(
        [0.6961, 1.1129, 1.7474, -2.2817, 1.3084, -1.1489, 4.7124, 0.8210, 0.8210, 0.8210, 0.8210, -0.8210, -0.8210]
      ),
      pose=sapien.Pose(),
    )
  )

  @property
  def _controller_configs(
    self,
  ) -> Dict[str, Union[ControllerConfig, DictControllerConfig]]:
    # Arm joint position controller
    arm_pd_joint_pos = PDJointPosControllerConfig(
      joint_names=self.arm_joint_names,
      lower=None,
      upper=None,
      stiffness=self.arm_stiffness,
      damping=self.arm_damping,
      force_limit=self.arm_force_limit,
      normalize_action=False,
    )
    # Arm end effector delta position controller
    arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
      joint_names=self.arm_joint_names,
      pos_lower=-0.1,
      pos_upper=0.1,
      rot_lower=-0.1,
      rot_upper=0.1,
      stiffness=self.arm_stiffness,
      damping=self.arm_damping,
      force_limit=self.arm_force_limit,
      ee_link=self.ee_link_name,
      urdf_path=self.urdf_path,
    )
    # define a passive controller config to simply "turn off" other joints from being controlled and set their properties (damping/friction) to 0.
    # these joints are controlled passively by the mimic controller later on.
    passive_finger_joint_names = [
      "left_inner_knuckle_joint",
      "right_inner_knuckle_joint",
      "left_inner_finger_joint",
      "right_inner_finger_joint",
    ]
    passive_finger_joints = PassiveControllerConfig(
      joint_names=passive_finger_joint_names,
      damping=0,
      friction=0,
    )

    finger_joint_names = ["left_outer_knuckle_joint", "right_outer_knuckle_joint"]
    # use a mimic controller config to define one action to control both fingers
    finger_mimic_pd_joint_pos = PDJointPosMimicControllerConfig(
      joint_names=finger_joint_names,
      lower=None,
      upper=None,
      stiffness=1e5,
      damping=1e3,
      force_limit=0.1,
      friction=0.05,
      normalize_action=False,
    )
    finger_mimic_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
      joint_names=finger_joint_names,
      lower=-0.1,
      upper=0.1,
      stiffness=1e5,
      damping=1e3,
      force_limit=0.1,
      normalize_action=True,
      friction=0.05,
      use_delta=True,
    )
    return dict(
      pd_joint_pos=dict(
        arm=arm_pd_joint_pos,
        finger=finger_mimic_pd_joint_pos,
        passive_finger_joints=passive_finger_joints,
      ),
      pd_ee_delta_pose=dict(
        arm=arm_pd_ee_delta_pose,
        finger=finger_mimic_pd_joint_delta_pos,
        passive_finger_joints=passive_finger_joints,
      ),
    )

  def _after_loading_articulation(self):
    outer_finger = self.robot.active_joints_map["right_inner_finger_joint"]
    inner_knuckle = self.robot.active_joints_map["right_inner_knuckle_joint"]
    pad = outer_finger.get_child_link()
    lif = inner_knuckle.get_child_link()

    # the next 4 magic arrays come from https://github.com/haosulab/cvpr-tutorial-2022/blob/master/debug/robotiq.py which was
    # used to precompute these poses for drive creation
    p_f_right = [-1.6048949e-08, 3.7600022e-02, 4.3000020e-02]
    p_p_right = [1.3578170e-09, -1.7901104e-02, 6.5159947e-03]
    p_f_left = [-1.8080145e-08, 3.7600014e-02, 4.2999994e-02]
    p_p_left = [-1.4041154e-08, -1.7901093e-02, 6.5159872e-03]

    right_drive = self.scene.create_drive(
      lif, sapien.Pose(p_f_right), pad, sapien.Pose(p_p_right)
    )
    right_drive.set_limit_x(0, 0)
    right_drive.set_limit_y(0, 0)
    right_drive.set_limit_z(0, 0)

    outer_finger = self.robot.active_joints_map["left_inner_finger_joint"]
    inner_knuckle = self.robot.active_joints_map["left_inner_knuckle_joint"]
    pad = outer_finger.get_child_link()
    lif = inner_knuckle.get_child_link()

    left_drive = self.scene.create_drive(
      lif, sapien.Pose(p_f_left), pad, sapien.Pose(p_p_left)
    )
    left_drive.set_limit_x(0, 0)
    left_drive.set_limit_y(0, 0)
    left_drive.set_limit_z(0, 0)

  def _after_init(self):
    self.finger1_link = sapien_utils.get_obj_by_name(
      self.robot.get_links(), "left_inner_finger"
    )
    self.finger2_link = sapien_utils.get_obj_by_name(
      self.robot.get_links(), "right_inner_finger"
    )
    self.finger1pad_link = sapien_utils.get_obj_by_name(
      self.robot.get_links(), "left_inner_finger_pad"
    )
    self.finger2pad_link = sapien_utils.get_obj_by_name(
      self.robot.get_links(), "right_inner_finger_pad"
    )
    self.tcp = sapien_utils.get_obj_by_name(
      self.robot.get_links(), self.ee_link_name
    )
  
  def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
    """Check if the robot is grasping an object

    Args:
      object (Actor): The object to check if the robot is grasping
      min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
      max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
    """
    l_contact_forces = self.scene.get_pairwise_contact_forces(
      self.finger1_link, object
    )
    r_contact_forces = self.scene.get_pairwise_contact_forces(
      self.finger2_link, object
    )
    lforce = torch.linalg.norm(l_contact_forces, axis=1)
    rforce = torch.linalg.norm(r_contact_forces, axis=1)

    # direction to open the gripper
    ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
    rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
    langle = common.compute_angle_between(ldirection, l_contact_forces)
    rangle = common.compute_angle_between(rdirection, r_contact_forces)
    lflag = torch.logical_and(
      lforce >= min_force, torch.rad2deg(langle) <= max_angle
    )
    rflag = torch.logical_and(
      rforce >= min_force, torch.rad2deg(rangle) <= max_angle
    )
    return torch.logical_and(lflag, rflag)

  def is_static(self, threshold: float = 0.2):
    qvel = self.robot.get_qvel()[..., :-2]
    return torch.max(torch.abs(qvel), 1)[0] <= threshold
  
  @property
  def tcp_pos(self):
    return self.tcp.pose.p

  @property
  def tcp_pose(self):
    return self.tcp.pose

@register_env("KinovaPushCube", max_episode_steps=50)
class KinovaPushCubeEnv(PushCubeEnv):
  SUPPORTED_ROBOTS = [
    "kinova_gen3",
  ]

  def __init__(self, *args, robot_uids="kinova_gen3", **kwargs):
    self.block_offset = kwargs["block_offset"]
    self.block_gen_range = kwargs["block_gen_range"]
    self.target_offset = kwargs["target_offset"]
    self.goal_radius = kwargs["goal_radius"]
    
    del kwargs["block_offset"]
    del kwargs["block_gen_range"]
    del kwargs["target_offset"]
    del kwargs["goal_radius"]
    super().__init__(*args, robot_uids=robot_uids, **kwargs)

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
      target_region_xyz[..., :2] += torch.tensor(self.target_offset)
      # set a little bit above 0 so the target is sitting on the table
      target_region_xyz[..., 2] = 1e-3
      self.goal_region.set_pose(
        Pose.create_from_pq(
          p=target_region_xyz,
          q=euler2quat(0, np.pi / 2, 0),
        )
      )
      # set the keyframe for the robot
      self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)

  def evaluate(self):
    info = super().evaluate()
    if "success" in info:
      info["_success"] = info.pop("success")
    info["terminated"] = False
    return info

  def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
    info["success"] = torch.zeros(self.num_envs, dtype=bool, device=self.device)
    reward = super().compute_dense_reward(obs, action, info)
    info.pop("success")
    reward[info["_success"]] = 4
    return reward

  def render(self):
    # Return the rendered image from the first environment in the batch
    return super().render()[0].cpu().numpy()
