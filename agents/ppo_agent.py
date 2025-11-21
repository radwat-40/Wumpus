import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.ppo_policy import PPONetwork


class PPOAgent:
    def __init__(self, grid_size: int, n_actions: int,
                 gamma: float = 0.99,
                 lr: float = 3e-4,
                 clip_eps: float = 0.2):
        self.gamma = gamma
        self.clip_eps = clip_eps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPONetwork(grid_size, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.reset_buffer()

    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.dones = []

    def choose_action(self, obs):
        # obs: numpy (4, H, W)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.policy(obs_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return int(action.item()), logprob

    def store_transition(self, state, action, reward, logprob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.logprobs.append(logprob)
        self.dones.append(done)

    def _compute_returns(self):
        returns = []
        G = 0.0
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            if d:
                G = 0.0
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def learn(self):
        if not self.states:
            return

        states = torch.from_numpy(np.array(self.states)).float().to(self.device)     # (T,4,H,W)
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)   # (T,)
        old_logprobs = torch.stack(self.logprobs).detach()                           # (T,)
        returns = self._compute_returns()                                            # (T,)

        # Normalize returns als Vorteil
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Neue Policy evaluieren
        probs = self.policy(states)                          # (T, n_actions)
        dist = torch.distributions.Categorical(probs)
        new_logprobs = dist.log_prob(actions)                # (T,)

        ratio = torch.exp(new_logprobs - old_logprobs)       # (T,)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

        loss = -torch.min(unclipped, clipped).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.reset_buffer()
