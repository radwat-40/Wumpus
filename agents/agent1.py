from agents.base_agent import Agent
from environment.actions import Action
import os
import random

# DQN / PyTorch imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.net(x)


class MarcAgent(Agent):
    """DQN-basierter Agent mit kleinem CNN-Encoder.

    Observation format expected: numpy array shape (C, H, W) float32
    Actions: we map network outputs to Action enums in ACTIONS
    """

    ACTIONS = [
        Action.MOVE_UP,
        Action.MOVE_DOWN,
        Action.MOVE_LEFT,
        Action.MOVE_RIGHT,
        Action.GRAB,
        Action.SHOOT_UP,
        Action.SHOOT_DOWN,
        Action.SHOOT_LEFT,
        Action.SHOOT_RIGHT,
    ]

    def __init__(self, x, y, role, device=None):
        super().__init__(x, y, role)
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # DQN hyperparams (tunable)
        self.gamma = 0.99
        self.lr = 1e-3  # Higher learning rate for faster convergence
        self.batch_size = 32  # Smaller batch size for more frequent updates
        self.replay_size = 10000
        self.min_replay_size = 256  # Start learning earlier
        self.target_update_freq = 2000  # Update target net more frequently
        self.epsilon = 1.0
        self.eps_final = 0.05
        self.eps_decay = 10000  # Faster epsilon decay

        self.n_actions = len(self.ACTIONS)
        self.in_channels = 4  # breeze, stench, glitter, agent_pos

        # Networks
        self.policy_net = SimpleCNN(self.in_channels, self.n_actions).to(self.device)
        self.target_net = SimpleCNN(self.in_channels, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Replay buffer
        self.memory = deque(maxlen=self.replay_size)

        # Counters
        self.steps_done = 0
        self.learn_steps = 0

    def action_id_to_enum(self, idx):
        return self.ACTIONS[int(idx)]

    def decide_move(self, observation, grid_size):
        # observation: numpy (C,H,W)
        state = torch.from_numpy(np.array(observation, dtype=np.float32)).unsqueeze(0).to(self.device)

        sample = random.random()
        eps_threshold = self.eps_final + (self.epsilon - self.eps_final) * max(0, (1 - self.steps_done / self.eps_decay))
        
        if sample < eps_threshold:
            # random action
            a = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                qvals = self.policy_net(state)
                a = qvals.argmax(dim=1).item()

        # Increment steps and decay epsilon
        self.steps_done += 1
        self.epsilon = self.eps_final + (1.0 - self.eps_final) * max(0, (1 - self.steps_done / self.eps_decay))

        return self.action_id_to_enum(a)

    def store_transition(self, state, action, reward, next_state, done):
        # Convert action enum to index
        try:
            a_idx = self.ACTIONS.index(action)
        except ValueError:
            a_idx = 0
        self.memory.append(
            Transition(
                np.copy(state),
                a_idx,
                reward,
                np.copy(next_state),
                done
            )
        )


    def sample_batch(self):
        batch = random.sample(self.memory, self.batch_size)
        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.uint8)

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones.astype(np.uint8)).to(self.device)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def learn_step(self):
        # Only learn when buffer has enough
        if len(self.memory) < max(self.batch_size, self.min_replay_size):
            return

        states_t, actions_t, rewards_t, next_states_t, dones_t = self.sample_batch()

        # Compute Q(s,a)
        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # Save / Load for compatibility with training.py expectations
    def save_q_table(self, filename="dqn_marc.pth"):
        torch.save({
            'policy_state': self.policy_net.state_dict(),
            'target_state': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
        }, filename)

    def load_q_table(self, filename="dqn_marc.pth"):
        if os.path.exists(filename):
            data = torch.load(filename, map_location=self.device)
            self.policy_net.load_state_dict(data['policy_state'])
            self.target_net.load_state_dict(data.get('target_state', data['policy_state']))
            if 'optimizer' in data:
                try:
                    self.optimizer.load_state_dict(data['optimizer'])
                except Exception:
                    pass
            self.steps_done = data.get('steps_done', self.steps_done)
            return True
        return False

    def reset_episode(self):
        # nothing per-episode to clear for this DQN agent
        pass

    def reward_from_result(self, result, percepts):
        if result == "WIN":
            return 100.0
        if result == "DIED":
            return -100.0
        
        # shaping (optional aber nützlich)
        reward = 0.0
        if percepts.get("breeze"):
            reward -= 1.0
        if percepts.get("stench"):
            reward -= 1.0
            
        return reward

