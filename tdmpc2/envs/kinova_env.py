import torch
import math
import genesis as gs

class KinovaEnv:
    def __init__(self, num_envs, env_cfg, show_viewer=False):
        """
        Initializes the KinovaEnv environment.

        Args:
            num_envs (int): Number of parallel environments.
            env_cfg (dict): Configuration dictionary for environment settings.  Contains the following keys:
                - "episode_length_s" (int): Duration of each episode in seconds.
                - "init_box_pos" (array): Initial position of the box.
                - "box_size" (tuple): Size of the box.
                - "init_joint_angles" (array):  Initial joint angles of the robot.
                - "bracelet_link_height" (float): Height to keep bracelet link at
                - "init_quat" (array):          Initial orientation (quaternion) of the end-effector.
                - "clip_actions" (float):     Maximum absolute value for actions.
                - "termination_if_cube_goal_dist_less_than" (float): Distance threshold for episode termination.
                - "cube_goal_dist_rew_scale" (float): Scaling factor for the cube-goal distance reward term.
                - "cube_arm_dist_rew_scale"  (float): Scaling factor for the cube-arm distance reward term.
                - "success_reward" (int):     Reward given when the task is successful.
                - "target_displacement" (float): Amount to move box in y-axis 
                - "action_scale" (float): Scaling factor for actions.
            show_viewer (bool, optional): Whether to display a viewer. Defaults to False.
        """

        self.num_envs = num_envs
        self.obs_per_step = 10 # no. of dimensions in the observation space per scene step
            
        self.num_actions = 2 # no. of dims in action space

        self.device = gs.device

        self.env_cfg = env_cfg

        self.dt = 0.01
        self.max_episode_length = math.ceil(env_cfg.episode_length_s / self.dt)

        self.obs_buffer_size = env_cfg.prev_obs_buffer # Stores the `obs_buffer_size` previous observations (including current obs)
        self.num_obs = self.obs_buffer_size * self.obs_per_step # total no. of dimensions in the obs space.

        self.prev_obs = []
        for i in range(self.obs_buffer_size):
            self.prev_obs.append(torch.zeros((self.num_envs,  self.obs_per_step), device=gs.device, dtype=gs.tc_float))

        self.scene = gs.Scene(
          sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
          viewer_options=gs.options.ViewerOptions(
              max_FPS=30
          ),
          vis_options=gs.options.VisOptions(
            rendered_envs_idx=list(range(1))
          ), # Only render one environment
          rigid_options=gs.options.RigidOptions(
              dt=self.dt,
          ),
          show_viewer=show_viewer,
        )

        self.cam = self.scene.add_camera(
              res=(320, 240),
              pos=(1.5, 0.5, 1.0),
              lookat=(0, 0, 0.2),
              up=(0, 0, 1),
              fov=40,
        )

        # add plane
        self.scene.add_entity(
            gs.morphs.Plane(),
        )
        # add box
        self.init_box_pos = torch.tensor(
            list(self.env_cfg.init_box_pos),
            device=gs.device,
            dtype=gs.tc_float
        )
        self.goal = torch.clone(self.init_box_pos)
        self.goal[1] += self.env_cfg.target_displacement
        self.box = self.scene.add_entity(
            gs.morphs.Box(
                pos=self.env_cfg.init_box_pos, # (0.2, 0.2, 0.02)
                size=self.env_cfg.box_size # (0.08, 0.08, 0.02)
            ),
            gs.materials.Rigid(
                rho=400, # 400 kg/m^3 -> density of some types of wood
                friction=None
            ), # The params here can be used for domain randomization
            gs.surfaces.Default(
                color=(196/255, 30/255, 58/255) # make block red
            )
        )

        # add robot
        self.init_joint_angles = torch.tensor(
            self.env_cfg.init_joint_angles,
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file='/root/tdmpc2/tdmpc2/envs/kinova_gen3/gen3.xml'),           
        )

        # get ee link
        self.bracelet_link = self.robot.get_link('bracelet_link')
        self.ee_init_quat = torch.tensor(
            self.env_cfg.init_quat,
            device=gs.device,
            dtype=gs.tc_float
        )

        # build
        self.scene.build(n_envs=num_envs)

        # buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_sums = {
            "cube_goal_dist_rew" : torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float),
            "cube_arm_dist_rew" : torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float),
            "success_rew" : torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float),
        }
        self.info = dict()
        self.info["observations"] = dict() # Only for PPO library
        self.info["success"] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # set to initial state
        self._reset_idx(torch.arange(self.num_envs, device=gs.device))
        self.scene.step()

        # tanh layer
        self.tanh = torch.nn.Tanh()
        self.tanh.to(device=gs.device)

    def step(self, actions):
        # Clamp action between bounds
        clipped_actions = torch.clip(self.env_cfg.action_scale*actions, -self.env_cfg.clip_actions, self.env_cfg.clip_actions)
        new_pos = self.bracelet_link.get_pos()
        new_pos[:, 0] += clipped_actions[:, 0]
        new_pos[:, 1] += clipped_actions[:, 1]
        new_pos[:, 2] = self.env_cfg.bracelet_link_height

        # Clip new pos to be within bounds of robot
        clipped_new_pos = new_pos.clone()
        clipped_new_pos[:, 0] = torch.clip(new_pos[:, 0], -0.5, 0.5)
        clipped_new_pos[:, 1] = torch.clip(new_pos[:, 1], -0.5, 0.5)
        
        # Compute inverse kinematics after applying delta ee pos
        qpos = self.robot.inverse_kinematics(
            link = self.bracelet_link,
            pos  = clipped_new_pos,
            quat = self.ee_init_quat.unsqueeze(dim=0).repeat(self.num_envs, 1),
        )
        qpos[:,-6:] = 0.822 # Keep gripper closed
        self.robot.control_dofs_position(qpos)
        self.scene.step()

        # qpos = self.robot.get_qpos().clone()
        # qpos[:, :6] += self.env_cfg.action_scale*actions[:, :6] # Only control the first 6 dofs
        # qpos[:, -6:] = self.env_cfg.action_scale*actions[:, -1].unsqueeze(dim=1) # Control gripper with last action dim
        # self.robot.control_dofs_position(qpos)
        # self.scene.step()

        # increment episode length
        self.episode_length_buf += 1

        # compute reward
        self.rew_buf[:] = 0.0
        rew_terms = self._get_reward()
        for key in self.episode_sums.keys():
            self.rew_buf += rew_terms[key]
            self.episode_sums[key] += rew_terms[key]

        # # check termination and reset
        # # terminate if box is at goal
        # self.reset_buf = torch.norm(self.goal[:2] - self.box.get_pos()[:, :2], dim=1) < self.env_cfg.termination_if_cube_goal_dist_less_than
        # self.info["success"][:] = self.reset_buf.int()
        # # terminate if episode length is reached
        # self.reset_buf |= self.episode_length_buf > self.max_episode_length
        # # terminate if robot is out of bounds
        # self.reset_buf |= (torch.abs(new_pos[:, 0]) > 0.5)
        # self.reset_buf |= (torch.abs(new_pos[:, 1]) > 0.5)
        # # terminate if action is out of bounds
        # self.reset_buf |= (torch.abs(clipped_actions[:, 0]) > self.env_cfg.clip_actions)
        # self.reset_buf |= (torch.abs(clipped_actions[:, 1]) > self.env_cfg.clip_actions)

        # for tdmpc2 all envs must terminate at same time 
        self.reset_buf = self.episode_length_buf > self.max_episode_length

        time_out_idx = torch.flatten(torch.nonzero(self.episode_length_buf > self.max_episode_length, as_tuple=False).nonzero(as_tuple=False))
        self.info["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.info["time_outs"][time_out_idx] = 1.0
        
        # Reset environments that have terminated
        self._reset_idx(torch.flatten(torch.nonzero(self.reset_buf, as_tuple=False)))

        # compute observations
        self.obs_buf = self._get_observation()

        # set info termination (only for tdmpc2)
        self.info['terminated'] = torch.tensor(0.0, device=gs.device, dtype=gs.tc_float)

        return self.obs_buf, self.rew_buf, self.reset_buf, self.info
    
    def get_observations(self):
        """
        Only for getting observation shape
        Used by PPO Library
        """
        return self.obs_buf, self.info
    
    def get_privileged_observations(self):
        """
        Only used by PPO library
        """
        return None

    def _get_observation(self):
        ee_pos_2d = self.bracelet_link.get_pos()[:, :2]
        curr_obs = torch.cat(
            [
                self.goal[:2] - ee_pos_2d, # 2
                self.box.get_pos()[:, :2] - ee_pos_2d, # 2
                self.box.get_vel()[:, :2], # 2
                self.box.get_quat(), # 4
            ],
            dim=1,
        ) # 10 dims total

        # Add newest obs to the `prev_obs` queue and pop the last obs
        self.prev_obs.insert(0, curr_obs)
        self.prev_obs.pop()

        return torch.cat(
            [obs for obs in self.prev_obs],
            dim = 1
        )

    
    def _get_reward(self):
        cube_goal_dist = torch.norm(self.goal[:2] - self.box.get_pos()[:, :2], dim=1)
        # Note: cube-arm dist doesn't account for width of box yet
        w, z = self.box.get_quat()[:, 0], self.box.get_quat()[:, 3]
        cube_back_pos = self.box.get_pos()[:, :2] + (self.env_cfg.box_size[1] / 2)*torch.stack([2*w*z, 2*z**2 - 1], dim=1)
        cube_arm_dist = torch.norm(self.bracelet_link.get_pos()[:, :2] - cube_back_pos, dim=1)

        # success = (cube_goal_dist < self.env_cfg.termination_if_cube_goal_dist_less_than).int()      
        success = (torch.abs(self.goal[1] - self.box.get_pos()[:, 1]) < self.env_cfg.termination_if_cube_goal_dist_less_than)
        success &= (torch.abs(self.goal[0] - self.box.get_pos()[:, 0]) < self.env_cfg.termination_if_cube_goal_dist_less_than*10)
        self.info["success"][:] = success.int()

        # reward terms
        INIT_CUBE_ARM_DIST = 0.2
        rew_terms = {
            "cube_goal_dist_rew": torch.where(
                cube_goal_dist > 0.1,
                self.env_cfg.target_displacement*self.env_cfg.cube_goal_dist_rew_scale - self.env_cfg.cube_goal_dist_rew_scale*cube_goal_dist,
                (self.env_cfg.target_displacement*self.env_cfg.cube_goal_dist_rew_scale - self.env_cfg.cube_goal_dist_rew_scale*cube_goal_dist)/10
            ),
            "cube_arm_dist_rew": torch.where(
                cube_arm_dist > INIT_CUBE_ARM_DIST,
                INIT_CUBE_ARM_DIST*self.env_cfg.cube_arm_dist_rew_scale - self.env_cfg.cube_arm_dist_rew_scale*cube_arm_dist,
                (INIT_CUBE_ARM_DIST*self.env_cfg.cube_arm_dist_rew_scale - self.env_cfg.cube_arm_dist_rew_scale*cube_arm_dist)/10
            ),
            "success_rew" : self.env_cfg.success_reward*success.int(),
        }

        return rew_terms

    def _reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.robot.set_dofs_position(
            position=self.init_joint_angles.unsqueeze(dim=0).repeat(len(envs_idx), 1),
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset box
        self.box.set_pos(
            pos=self.init_box_pos.unsqueeze(dim=0).repeat(len(envs_idx), 1),
            zero_velocity=True,
            envs_idx=envs_idx
        )

        self.box.set_quat(
            quat=torch.tensor([1,0,0,0]).unsqueeze(dim=0).repeat(len(envs_idx), 1),
            zero_velocity=True,
            envs_idx=envs_idx
        )

        # reset buffers
        self.episode_length_buf[envs_idx] = 0
        # self.reset_buf[envs_idx] = True

        # fill info
        self.info["episode"] = {}
        for key in self.episode_sums.keys():
            self.info["episode"][key] = torch.mean(self.episode_sums[key][envs_idx]).item() # / self.env_cfg["episode_length_s"]
            self.episode_sums[key][envs_idx] = 0.0
        self.info["episode"]["success_pct"] = 100*torch.mean(self.info["success"][envs_idx]).item()

    def reset(self):
        self._reset_idx(torch.arange(self.num_envs, device=gs.device))
        self.obs_buf = self._get_observation()
        return self.obs_buf

    def rand_act(self):
        return 2*torch.rand((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float) - 1

    def render(self):
        return self.cam.render()[0]
