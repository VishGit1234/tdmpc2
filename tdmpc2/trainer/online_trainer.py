from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes, ep_lengths = [], [], []
		for i in range(self.cfg.eval_episodes // self.cfg.num_envs):
			obs, _ = self.env.reset()
			done = torch.tensor(False)
			ep_reward = torch.zeros(self.cfg.num_envs, device=self.env.get_wrapper_attr('device'))
			t = 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done.any():
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs.to(self.cfg.cuda_device), t0=t==0, eval_mode=True)
				obs, reward, terminated, truncated, info = self.env.step(action)
				done = terminated | truncated
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			assert done.all(), 'Vectorized environments must reset all environments at once.'
			ep_rewards.append(ep_reward)
			ep_successes.append(info['_success'])
			ep_lengths.append(t)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=torch.cat(ep_rewards).mean().cpu(),
			episode_success=100*torch.cat(ep_successes).float().mean().cpu(),
			episode_length=np.nanmean(ep_lengths),
		)
	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0)
		if action is None:
			action = torch.full_like(torch.from_numpy(self.env.action_space.sample()), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan')).repeat(self.cfg.num_envs)
		if terminated is None:
			terminated = torch.tensor(float('nan')).repeat(self.cfg.num_envs)
		td = TensorDict(dict(
			obs=obs.cpu(),
			action=action.unsqueeze(0).cpu(),
			reward=reward.unsqueeze(0).cpu(),
			terminated=terminated.unsqueeze(0).cpu(),
		), batch_size=(1, self.cfg.num_envs,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, torch.tensor(True), False
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0 and self._step > 0:
				eval_next = True

			# Reset environment
			if done.any():
				assert done.all(), 'Vectorized environments must reset all environments at once.'
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError('Termination detected but you are not in episodic mode. ' \
						'Set `episodic=true` to enable support for terminations.')
					train_metrics.update(
						episode_reward=torch.cat([td['reward'] for td in self._tds[1:]]).sum(0).mean(),
						episode_success=100*info['_success'].float().nanmean(),
						episode_length=len(self._tds),
						episode_terminated=info['terminated'])
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs, _ = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs.to(self.cfg.cuda_device), t0=len(self._tds)==1)
			else:
				action = torch.from_numpy(self.env.action_space.sample()).to(self.cfg.cuda_device)
			obs, reward, terminated, truncated, info = self.env.step(action)
			done = terminated | truncated
			self._tds.append(self.to_td(obs, action, reward))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = int(self.cfg.seed_steps / self.cfg.steps_per_update)
					print('Pretraining agent on seed data...')
				else:
					num_updates = max(1, int(self.cfg.num_envs / self.cfg.steps_per_update))
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += self.cfg.num_envs

		self.logger.finish(self.agent)
