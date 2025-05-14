import numpy as np
import torch
import random

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), gamma=0.99, prevent_replacement=False):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.gamma = gamma
		self.prevent_replacement = prevent_replacement

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.next_action = np.zeros((max_size, action_dim))
		self.not_done = np.zeros((max_size, 1))
		self.episode_done = np.zeros((max_size, 1))
		self.g_return = np.zeros((max_size, 1))
		self.episode_total = np.zeros((max_size, 1))

		self.pending = []

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def push(self, state, action, next_state, reward, done, truncated):
		step = (state, action, next_state, reward, done)

		self.pending.append(step)

		if done or truncated:
			na = None
			G = 0.
			T = 0.

			for sample in self.pending:
				s, a, ns, r, d = sample
				T += r

			while len(self.pending) > 0:
				s, a, ns, r, d = self.pending.pop()

				if na is None:
					na = a

				G = r + self.gamma * G

				self.add(s, a, ns, r, na, float(d), G, float(done), T)

				na = a

	def add(self, state, action, next_state, reward, next_action, done, g_return, ep_done, T):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.next_action[self.ptr] = next_action
		self.not_done[self.ptr] = 1. - done
		self.episode_done[self.ptr] = float(ep_done)
		self.g_return[self.ptr] = g_return
		self.episode_total[self.ptr] = T

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size, last_eval=None):
		if self.prevent_replacement:
			ind = np.random.permutation(self.size)[:min(self.size, batch_size)]
		else:
			ind = np.random.randint(0, self.size, size=batch_size)

		# Clear low episodes
		ep_done_tensor = torch.FloatTensor(self.episode_done[ind])

		if last_eval is not None:
			ep_tot_tensor = torch.FloatTensor(self.episode_total[ind])
			ep_done_tensor = ep_done_tensor * torch.where(ep_tot_tensor > last_eval, 1., 0.)
			
		ret = (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.next_action[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device),
			ep_done_tensor.to(self.device),
			torch.FloatTensor(self.g_return[ind]).to(self.device)
		)

		return ret