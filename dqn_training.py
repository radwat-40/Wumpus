import os
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment.world import World
from environment.actions import Action
from agents.base_agent import Agent





class WumpusSingleAgentEnv:
    """
    Single-Agent-Umgebung für A1 auf Basis von environment.world.World.

    State: 5-dim Vektor wie im Scheduler für A1:
      [x_norm, y_norm, breeze, stench, glitter]

    Aktionen:
      0: MOVE_UP
      1: MOVE_DOWN
      2: MOVE_LEFT
      3: MOVE_RIGHT
    """

    def __init__(
        self,
        grid_size=20,
        num_pits=20,
        num_wumpus=3,
        num_gold=1,
        max_steps=300,
    ):
        self.grid_size = grid_size
        self.num_pits = num_pits
        self.num_wumpus = num_wumpus
        self.num_gold = num_gold
        self.max_steps = max_steps

        # intern erzeugt pro Episode eine neue World
        self.world = None
        self.agent = None
        self.steps = 0

        # einfache Memory-Sets (analog Scheduler)
        self.known = set()
        self.safe = set()
        self.risky = set()

    def reset(self):
        # Neue Welt mit denselben Parametern wie im Spiel
        self.world = World(
            grid_size=self.grid_size,
            num_pits=self.num_pits,
            num_wumpus=self.num_wumpus,
            num_gold=self.num_gold,
        )

        # Agent startet oben links
        self.agent = Agent(0, 0, "A1")
        self.agent.agent_alive = True
        self.agent.agent_won = False

        self.steps = 0

        self.known = set()
        self.safe = set()
        self.risky = set()

        # Start-Percepts für Memory
        p = self.world.get_percepts(self.agent)
        self._update_memory(self.agent.pos(), p)

        return self._build_obs()

    def _build_obs(self):
        """
        Gleiche Observation wie im Scheduler für A1:
        [x_norm, y_norm, breeze, stench, glitter, known_fraction]
        """
        x, y = self.agent.pos()
        N = max(self.grid_size - 1, 1)
        x_norm = x / N
        y_norm = y / N

        p = self.world.get_percepts(self.agent)
        breeze = 1.0 if p["breeze"] else 0.0
        stench = 1.0 if p["stench"] else 0.0
        glitter = 1.0 if p["glitter"] else 0.0

        # Anteil bereits bekannter Felder (0..1)
        total_tiles = float(self.grid_size * self.grid_size)
        known_fraction = len(self.known) / total_tiles if total_tiles > 0 else 0.0

        return np.array(
            [x_norm, y_norm, breeze, stench, glitter, known_fraction],
            dtype=np.float32,
        )

    def _update_memory(self, pos, percepts):
        self.known.add(pos)
        if percepts["breeze"] or percepts["stench"]:
            self.risky.add(pos)
        else:
            self.safe.add(pos)

    def step(self, action_idx: int):
        """
        Ausführung einer Aktion, Reward-Shaping konsistent und ohne Explosions-Farming.
        """
        self.steps += 1

        # Map Index -> Action (nur Bewegungen)
        action_map = [
            Action.MOVE_UP,
            Action.MOVE_DOWN,
            Action.MOVE_LEFT,
            Action.MOVE_RIGHT,
        ]
        action_idx = int(action_idx)
        if action_idx < 0 or action_idx >= len(action_map):
            raise ValueError(f"Ungültiger Action-Index: {action_idx}")

        action = action_map[action_idx]

        old_pos = self.agent.pos()
        old_p = self.world.get_percepts(self.agent)
        old_known = set(self.known)
        old_safe = set(self.safe)
        old_risky = set(self.risky)

        result = self.world.execute(self.agent, action)

        new_pos = self.agent.pos()
        new_p = self.world.get_percepts(self.agent)
        self._update_memory(new_pos, new_p)

        reward = self._compute_reward(
            old_pos,
            new_pos,
            old_p,
            new_p,
            old_known,
            old_safe,
            old_risky,
            result,
        )

        done = (
            result in ("WIN", "DIED")
            or not getattr(self.agent, "agent_alive", True)
            or self.steps >= self.max_steps
        )

        obs = self._build_obs()

        info = {"result": result}
        return obs, reward, done, info

    def _compute_reward(
        self,
        old_pos,
        new_pos,
        old_p,
        new_p,
        old_known,
        old_safe,
        old_risky,
        result,
    ):
        """
        Stabiler, nicht ausnutzbarer Reward:
          +1000  WIN
          -1000  DIED
          +1     neues Feld
          +0.5   safe -> safe
          -1     risky (breeze/stench)
          -0.1   pro Schritt
        """
        r = 0.0

        # Terminale Ereignisse
        if result == "WIN":
            r += 1000.0
        elif result == "DIED":
            r -= 1000.0

        # Neues Feld betreten
        if new_pos not in old_known:
            r += 1.0

        # Safe-to-safe Bewegung
        safe_before = not (old_p["breeze"] or old_p["stench"])
        safe_after = not (new_p["breeze"] or new_p["stench"])
        if safe_before and safe_after:
            r += 0.5

        # Risiko-Strafe (risky oder neue Breeze/Stench)
        if new_pos in old_risky or (new_p["breeze"] or new_p["stench"]):
            r -= 1.0

        # Zeitschritt-Strafe, damit er nicht idlet
        r -= 0.1

        return float(r)




class DQN(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done"),
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*batch))
        return batch

    def __len__(self):
        return len(self.buffer)




def evaluate_policy(policy_net, env_params, n_eval_episodes=20, device=None):
    """
    Bewertet das aktuelle Policy-Netz auf der Zielwelt (20x20).
    Epsilon = 0 (pure Greedy-Policy).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = WumpusSingleAgentEnv(**env_params)

    wins = 0

    policy_net.eval()
    with torch.no_grad():
        for _ in range(n_eval_episodes):
            state = env.reset()
            done = False

            while not done:
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                q_vals = policy_net(s_t)
                action = int(q_vals.argmax(dim=-1).item())

                next_state, reward, done, info = env.step(action)
                state = next_state

            if info.get("result") == "WIN":
                wins += 1

    winrate = wins / n_eval_episodes * 100.0
    return winrate


def train_dqn(
    num_episodes=5000,
    max_steps=300,
    gamma=0.99,
    lr=1e-3,
    batch_size=64,
    replay_capacity=50_000,
    min_replay_size=1_000,
    target_update_freq=500,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=50_000,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Env-Parameter wie im Spiel
    env_params = dict(grid_size=20, num_pits=20, num_wumpus=3, num_gold=1, max_steps=max_steps)
    env = WumpusSingleAgentEnv(**env_params)

    obs_dim = 6
    n_actions = 4  # 4 Moves, kein GRAB

    policy_net = DQN(obs_dim, n_actions).to(device)
    target_net = DQN(obs_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(replay_capacity)

    global_step = 0

    best_eval_winrate = 0.0

    def epsilon_by_step(step):
        if step >= eps_decay_steps:
            return eps_end
        frac = step / eps_decay_steps
        return eps_start + frac * (eps_end - eps_start)

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        episode_return = 0.0

        for t in range(max_steps):
            global_step += 1

            eps = epsilon_by_step(global_step)
            if random.random() < eps:
                action = random.randrange(n_actions)
            else:
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q_vals = policy_net(s_t)
                    action = int(q_vals.argmax(dim=-1).item())

            next_state, reward, done, info = env.step(action)

            replay_buffer.push(
                state.astype(np.float32),
                action,
                float(reward),
                next_state.astype(np.float32),
                float(done),
            )

            state = next_state
            episode_return += reward

            # Lernen
            if len(replay_buffer) >= min_replay_size:
                batch = replay_buffer.sample(batch_size)

                states = torch.tensor(
                    np.array(batch.state, dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                )

                actions = torch.tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(-1)

                rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(-1)

                next_states = torch.tensor(
                    np.array(batch.next_state, dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                )

                dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(-1)


                # Q(s,a)
                q_values = policy_net(states).gather(1, actions)

                # Target Q-Werte: r + gamma * max_a' Q_target(s', a')
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(dim=1, keepdim=True)[0]
                    target_q = rewards + gamma * (1.0 - dones) * next_q_values

                loss = nn.MSELoss()(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
                optimizer.step()

                # Target-Net periodisch updaten
                if global_step % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        # Logging
        if ep % 50 == 0:    
            print(
                f"Ep {ep}/{num_episodes} | Return: {episode_return:.2f} | "
                f"Eps: {epsilon_by_step(global_step):.3f}"
            )
        # Alle 200 Episoden: Evaluation auf Zielwelt
        if ep % 200 == 0:
            eval_winrate = evaluate_policy(policy_net, env_params, n_eval_episodes=20, device=device)
            print(f"[EVAL] Ep {ep}: 20x20 Winrate = {eval_winrate:.2f}%")

            if eval_winrate > best_eval_winrate:
                best_eval_winrate = eval_winrate
                torch.save(policy_net.state_dict(), "dqn_agent_best.pth")
                print(
                    f"[EVAL] Neues Bestmodell gespeichert: dqn_agent_best.pth "
                    f"(Winrate {best_eval_winrate:.2f}%)"
                )

    # letztes Modell auch speichern
    torch.save(policy_net.state_dict(), "dqn_agent_last.pth")
    print("DQN-Training abgeschlossen. Letztes Modell gespeichert: dqn_agent_last.pth")


if __name__ == "__main__":
    train_dqn()
