import gymnasium as gym

class KinovaMultitaskEnv(gym.Wrapper):

	def __init__(self, num_envs, *args, **kwargs):
		self.push_cube_env = gym.make("KinovaPushCube", num_envs = num_envs, **kwargs["push_cube_kwargs"])
		self.pick_cube_env = gym.make("KinovaPickCube", num_envs = num_envs, **kwargs["pick_cube_kwargs"])
		self.stack_cube_env = gym.make("KinovaStackCube", num_envs = num_envs, **kwargs["stack_cube_kwargs"])
		self.envs = [
			("kinova_push_cube", self.push_cube_env),
			("kinova_pick_cube", self.pick_cube_env),
			("kinova_stack_cube", self.stack_cube_env)
		]
		super().__init__(self.push_cube_env)
		assert self.push_cube_env.observation_space == self.pick_cube_env.observation_space == self.stack_cube_env.observation_space, "Observation spaces must be equal"
		assert self.push_cube_env.action_space == self.pick_cube_env.action_space == self.stack_cube_env.action_space, "Action spaces must be equal"

		self.observation_space = self.push_cube_env.observation_space
		self.action_space = self.push_cube_env.action_space

		self._task_idx = 0
		self._task = self.envs[self._task_idx][0]

	def print_sim_details(self):
		print(f"PushCube Env: ")
		self.push_cube_env.print_sim_details()
		print(f"PickCube Env: ")
		self.pick_cube_env.print_sim_details()
		print(f"StackCube Env: ")
		self.stack_cube_env.print_sim_details()

	@property
	def task(self):
		return self._task
	
	@property
	def task_idx(self):
		return self._task_idx

	@property
	def _env(self):
		return self.envs[self._task_idx][1]
	
	def get_task(self, task_idx):
		return self.envs[task_idx][0]

	def reset(self, task_idx = None, seed = None, options = None):
		if task_idx is None:
			task_idx = (self._task_idx + 1) % len(self.envs)
		self._task_idx = task_idx
		self._task = self.envs[task_idx][0]
		self.env = self._env
		# print("Current task:", self._task)
		return self.env.reset()

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		return obs, reward, terminated, truncated, info

	def render(self):
		if self.render_mode == "rgb_array":
			# Return the rendered image from the first environment in the batch
			return self._env.render()
		else:
			raise NotImplementedError("Only rgb_array render mode is supported")