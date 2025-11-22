import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment.world import World
from environment.actions import Action
from agents.base_agent import Agent


# ===========================
#   Multi-Agent Wumpus Env
# ===========================

class MultiAgentWumpusEnv:
    """
    Vereinfachte Multi-Agent-Trainingsumgebung für QMIX.
    - 3 Agenten (Team)
    - Gemeinsamer Team-Reward mit Shaping
    - Beobachtung pro Agent: [x_norm, y_norm, breeze, stench, glitter]
    """

    def __init__(self, grid_size=8, num_pits=3, num_wumpus=1, num_gold=1, max_steps=100):
        self.grid_size = grid_size
        self.num_pits = num_pits
        self.num_wumpus = num_wumpus
        self.num_gold = num_gold
        self.max_steps = max_steps

        self.world = World(grid_size, num_pits, num_wumpus, num_gold)
        self.n_agents = 3

        # Agenten-Objekte, Positionen werden gesetzt
        self.agents = [
            Agent(0, 0, "A1"),
            Agent(1, 0, "A2"),
            Agent(2, 0, "A3"),
        ]

        # Besuchte Felder pro Agent (für Reward-Shaping)
        self.visited = [set() for _ in range(self.n_agents)]

        self.steps = 0

    def reset(self):
        self.world.reset()
        self.steps = 0

        # Startpositionen
        starts = [(0, 0), (1, 0), (2, 0)]
        self.visited = [set() for _ in range(self.n_agents)]

        for idx, (a, (x, y)) in enumerate(zip(self.agents, starts)):
            a.x, a.y = x, y
            a.agent_alive = True
            a.agent_won = False
            self.visited[idx].add((x, y))

        obs_n = [self._get_obs(a) for a in self.agents]
        state = self._build_global_state(obs_n)
        return obs_n, state

    def _get_obs(self, agent):
        x, y = agent.pos()
        x_norm = x / (self.grid_size - 1) if self.grid_size > 1 else 0.0
        y_norm = y / (self.grid_size - 1) if self.grid_size > 1 else 0.0

        p = self.world.get_percepts(agent)
        breeze = 1.0 if p["breeze"] else 0.0
        stench = 1.0 if p["stench"] else 0.0
        glitter = 1.0 if p["glitter"] else 0.0

        return np.array([x_norm, y_norm, breeze, stench, glitter], dtype=np.float32)

    def _build_global_state(self, obs_n):
        # globaler State: concat aller Agent-Observations
        return np.concatenate(obs_n, axis=0).astype(np.float32)

    @property
    def obs_dim(self):
        # pro Agent
        return 5

    @property
    def state_dim(self):
        return self.obs_dim * self.n_agents

    def step(self, actions_n):
        """
        actions_n: list[int] Länge n_agents, jeweils 0..n_actions-1
        Mapped auf MOVE_* Aktionen.
        Reward-Shaping:
          +5  neues Feld (von irgendeinem Agenten besucht)
          +5  neues sicheres Feld (kein breeze/stench)
          -1  wenn aktuelles Feld risky ist
          -0.02 pro Schritt (global)
          +200 WIN, -100 DIED
        """

        self.steps += 1
        team_reward = 0.0
        done = False
        result_flags = []

        # Schritt-Strafe (einmal pro Env-Step)
        team_reward -= 0.02

        for idx, (agent, a_idx) in enumerate(zip(self.agents, actions_n)):
            if not getattr(agent, "agent_alive", True):
                result_flags.append("DEAD")
                continue

            act = self._map_action(a_idx)
            old_pos = agent.pos()

            result = self.world.execute(agent, act)
            result_flags.append(result)

            new_pos = agent.pos()
            # Percepts nach der Aktion
            p_after = self.world.get_percepts(agent)
            risky_now = p_after["breeze"] or p_after["stench"]

            # Terminal-Reward
            if result == "WIN":
                team_reward += 200.0
                agent.agent_won = True
                done = True
            elif result == "DIED":
                team_reward -= 100.0
                agent.agent_alive = False
            else:
                # Shaping: Exploration / Risiko
                if new_pos not in self.visited[idx]:
                    # Neues Feld
                    team_reward += 5.0
                    self.visited[idx].add(new_pos)

                    # Neues sicheres Feld
                    if not risky_now:
                        team_reward += 5.0

                # Risiko-Penalty
                if risky_now:
                    team_reward -= 1.0

        # Wenn alle tot → Ende
        if not any(getattr(a, "agent_alive", True) for a in self.agents):
            done = True

        # Schrittlimit?
        if self.steps >= self.max_steps:
            done = True

        obs_n = [self._get_obs(a) for a in self.agents]
        state = self._build_global_state(obs_n)
        return obs_n, state, team_reward, done, {"results": result_flags}

    def _map_action(self, a_idx):
        # 0: up, 1: down, 2: left, 3: right
        if a_idx == 0:
            return Action.MOVE_UP
        elif a_idx == 1:
            return Action.MOVE_DOWN
        elif a_idx == 2:
            return Action.MOVE_LEFT
        elif a_idx == 3:
            return Action.MOVE_RIGHT
        else:
            return random.choice([
                Action.MOVE_UP,
                Action.MOVE_DOWN,
                Action.MOVE_LEFT,
                Action.MOVE_RIGHT,
            ])


# ===========================
#      QMIX Netzwerke
# ===========================

class AgentQNetwork(nn.Module):
    """
    Per-Agent Q-Netzwerk: Q_i(a | o_i)
    o_i: obs_dim
    """

    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, obs):
        # obs: (B, obs_dim)
        return self.net(obs)  # (B, n_actions)


class MixingNetwork(nn.Module):
    """
    QMIX Mixing-Net: Q_tot = f(Q_1,...,Q_n, s)
    Hypernetworks generieren monotone Gewichte.
    """

    def __init__(self, n_agents, state_dim, embed_dim=32, hypernet_embed=64):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Hypernet für erste Schicht
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, n_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # Hypernet für zweite Schicht
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs, state):
        """
        agent_qs: (B, n_agents)
        state:    (B, state_dim)
        """
        bs = agent_qs.size(0)

        # W1, b1
        w1 = torch.abs(self.hyper_w1(state))  # (B, n_agents*embed_dim)
        b1 = self.hyper_b1(state)             # (B, embed_dim)

        w1 = w1.view(bs, self.n_agents, self.embed_dim)  # (B, n_agents, embed_dim)
        agent_qs = agent_qs.view(bs, 1, self.n_agents)   # (B,1,n_agents)

        # hidden = Q * W1 + b1
        hidden = torch.bmm(agent_qs, w1).squeeze(1) + b1  # (B, embed_dim)
        hidden = torch.relu(hidden)

        # W2, b2
        w2 = torch.abs(self.hyper_w2(state))  # (B, embed_dim)
        w2 = w2.view(bs, self.embed_dim, 1)   # (B, embed_dim,1)
        b2 = self.hyper_b2(state)             # (B,1)

        # Q_tot
        q_tot = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2  # (B,1)
        return q_tot.squeeze(-1)  # (B,)


# ===========================
#       Replay Buffer
# ===========================

Transition = namedtuple(
    "Transition",
    ("obs_n", "state", "actions_n", "reward", "next_obs_n", "next_state", "done"),
)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


# ===========================
#    Curriculum Definition
# ===========================

def get_curriculum_stage(ep: int):
    """
    Einfaches Curriculum:
      0- 999: 4x4, keine Gefahren
    1000-1999: 4x4, leichte Gefahren
    2000-2999: 5x5
    3000-3999: 6x6
    4000+   : 8x8
    """
    if ep < 1000:
        return dict(grid=4, pits=0, wumpus=0, gold=1, steps=20)
    elif ep < 2000:
        return dict(grid=4, pits=1, wumpus=0, gold=1, steps=25)
    elif ep < 3000:
        return dict(grid=5, pits=2, wumpus=1, gold=1, steps=35)
    elif ep < 4000:
        return dict(grid=6, pits=3, wumpus=1, gold=1, steps=45)
    else:
        return dict(grid=8, pits=5, wumpus=2, gold=1, steps=70)


# ===========================
#        QMIX Trainer
# ===========================

def train_qmix(
    num_episodes=5000,
    n_agents=3,
    n_actions=4,
    gamma=0.97,
    lr=5e-4,
    batch_size=64,
    buffer_capacity=100_000,
    start_learning=2000,
    target_update_interval=1000,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=50_000,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy-Env für Dimensionen (erste Curriculum-Stufe)
    first_params = get_curriculum_stage(0)
    tmp_env = MultiAgentWumpusEnv(
        grid_size=first_params["grid"],
        num_pits=first_params["pits"],
        num_wumpus=first_params["wumpus"],
        num_gold=first_params["gold"],
        max_steps=first_params["steps"],
    )

    obs_dim = tmp_env.obs_dim
    state_dim = tmp_env.state_dim

    # Per-Agent Q-Network (shared weights für alle Agenten)
    agent_q_net = AgentQNetwork(obs_dim, n_actions).to(device)
    target_agent_q_net = AgentQNetwork(obs_dim, n_actions).to(device)
    target_agent_q_net.load_state_dict(agent_q_net.state_dict())
    target_agent_q_net.eval()

    mixer = MixingNetwork(n_agents, state_dim).to(device)
    target_mixer = MixingNetwork(n_agents, state_dim).to(device)
    target_mixer.load_state_dict(mixer.state_dict())
    target_mixer.eval()

    optimizer = optim.Adam(
        list(agent_q_net.parameters()) + list(mixer.parameters()),
        lr=lr
    )
    replay = ReplayBuffer(buffer_capacity)

    steps_done = 0
    wins = 0
    deaths = 0

    for ep in range(1, num_episodes + 1):

        params = get_curriculum_stage(ep)
        env = MultiAgentWumpusEnv(
            grid_size=params["grid"],
            num_pits=params["pits"],
            num_wumpus=params["wumpus"],
            num_gold=params["gold"],
            max_steps=params["steps"],
        )

        obs_n, state = env.reset()
        done = False
        episode_return = 0.0
        last_info = {"results": []}

        while not done:
            # Epsilon-greedy
            eps = eps_end + (eps_start - eps_end) * np.exp(-1.0 * steps_done / eps_decay)
            eps = max(eps_end, min(eps_start, eps))
            steps_done += 1

            actions_n = []
            for o in obs_n:
                if random.random() < eps:
                    a = random.randrange(n_actions)
                else:
                    o_t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                    q_vals = agent_q_net(o_t)  # (1, n_actions)
                    a = int(q_vals.argmax(dim=-1).item())
                actions_n.append(a)

            next_obs_n, next_state, reward, done, info = env.step(actions_n)
            episode_return += reward
            last_info = info

            replay.push(
                [o.copy() for o in obs_n],
                state.copy(),
                actions_n[:],
                float(reward),
                [o.copy() for o in next_obs_n],
                next_state.copy(),
                done,
            )

            obs_n = next_obs_n
            state = next_state

            # Learning
            if len(replay) >= max(batch_size, start_learning):
                batch = replay.sample(batch_size)

                obs_batch = np.array(batch.obs_n, dtype=np.float32)        # (B, n_agents, obs_dim)
                next_obs_batch = np.array(batch.next_obs_n, dtype=np.float32)
                state_batch = np.array(batch.state, dtype=np.float32)      # (B, state_dim)
                next_state_batch = np.array(batch.next_state, dtype=np.float32)
                actions_batch = np.array(batch.actions_n, dtype=np.int64)  # (B, n_agents)
                rewards_batch = np.array(batch.reward, dtype=np.float32)   # (B,)
                done_batch = np.array(batch.done, dtype=np.uint8)          # (B,)

                B = obs_batch.shape[0]

                obs_t = torch.tensor(obs_batch, device=device)            # (B, n_agents, obs_dim)
                next_obs_t = torch.tensor(next_obs_batch, device=device)
                state_t = torch.tensor(state_batch, device=device)        # (B, state_dim)
                next_state_t = torch.tensor(next_state_batch, device=device)
                actions_t = torch.tensor(actions_batch, device=device)    # (B, n_agents)
                rewards_t = torch.tensor(rewards_batch, device=device)    # (B,)
                done_t = torch.tensor(done_batch, device=device, dtype=torch.float32)

                # Q_i(s,a)
                obs_flat = obs_t.view(B * n_agents, obs_dim)
                q_all = agent_q_net(obs_flat)  # (B*n_agents, n_actions)

                a_flat = actions_t.view(B * n_agents, 1)
                q_chosen = q_all.gather(1, a_flat).view(B, n_agents)  # (B, n_agents)

                # Target-Q_i
                next_obs_flat = next_obs_t.view(B * n_agents, obs_dim)
                with torch.no_grad():
                    q_next_all = target_agent_q_net(next_obs_flat)
                    q_next_max = q_next_all.max(dim=1)[0].view(B, n_agents)

                # QMIX
                q_tot = mixer(q_chosen, state_t)       # (B,)
                with torch.no_grad():
                    q_tot_next = target_mixer(q_next_max, next_state_t)  # (B,)

                target = rewards_t + gamma * (1.0 - done_t) * q_tot_next
                loss = nn.MSELoss()(q_tot, target)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(agent_q_net.parameters()) + list(mixer.parameters()),
                    max_norm=10.0
                )
                optimizer.step()

                # Target-Update
                if steps_done % target_update_interval == 0:
                    target_agent_q_net.load_state_dict(agent_q_net.state_dict())
                    target_mixer.load_state_dict(mixer.state_dict())

        # Stats
        if any(r == "WIN" for r in last_info["results"]):
            wins += 1
        if not any(getattr(a, "agent_alive", True) for a in env.agents):
            deaths += 1

        if ep % 100 == 0:
            winrate = wins / ep * 100.0
            print(
                f"Ep {ep}/{num_episodes} | "
                f"Wins: {wins} | Deaths: {deaths} | "
                f"Winrate: {winrate:.2f}% | "
                f"Return(last): {episode_return:.2f} | "
                f"Eps: {eps:.3f} | "
                f"Grid={params['grid']} Pits={params['pits']} Wump={params['wumpus']}"
            )

    # Modelle speichern
    torch.save(agent_q_net.state_dict(), "qmix_agent_q_net.pth")
    torch.save(mixer.state_dict(), "qmix_mixer.pth")
    print("QMIX-Training abgeschlossen. Modelle gespeichert: qmix_agent_q_net.pth, qmix_mixer.pth")


if __name__ == "__main__":
    train_qmix(
        num_episodes=5000,
        n_agents=3,
        n_actions=4,
    )
