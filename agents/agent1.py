import os
import numpy as np
import torch
import torch.nn as nn

from agents.base_agent import Agent
from environment.actions import Action


class AgentQNetwork(nn.Module):
    """
    Per-Agent Q-Netzwerk: Q_i(a | o_i)
    Beobachtung o_i: 5 Features
    """

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class MarcAgent(Agent):
    """
    QMIX-Inference-Agent (A1).
    Erwartet Observation:
      [x_norm, y_norm, breeze, stench, glitter]
    """

    ACTIONS = [
        Action.MOVE_UP,
        Action.MOVE_DOWN,
        Action.MOVE_LEFT,
        Action.MOVE_RIGHT,
    ]

    def __init__(
        self,
        x: int,
        y: int,
        role: str,
        grid_size: int = 20,
        model_path: str = "qmix_agent_q_net.pth",
        obs_dim: int = 5,
    ):
        super().__init__(x, y, role)

        self.grid_size = grid_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = obs_dim
        self.n_actions = len(self.ACTIONS)

        # Q-Netzwerk laden
        self.q_net = AgentQNetwork(self.obs_dim, self.n_actions).to(self.device)

        if os.path.exists(model_path):
            try:
                self.q_net.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"[QMIX MarcAgent] Modell geladen: {model_path}")
            except Exception as e:
                print(f"[QMIX MarcAgent] Fehler beim Laden: {e}")
                print("[QMIX MarcAgent] Fallback auf Zufallsaktionen.")
        else:
            print(f"[QMIX MarcAgent] WARNUNG: {model_path} nicht gefunden – Fallback auf Zufallsaktionen.")

        self.q_net.eval()

    def decide_move(self, observation, grid_size: int):
        """
        observation: np.array shape (5,) = [x_norm, y_norm, breeze, stench, glitter]
        """

        obs = np.asarray(observation, dtype=np.float32).reshape(-1)

        if obs.shape[0] != self.obs_dim:
            print(f"[QMIX MarcAgent] Falsches Observation-Format {obs.shape} → Zufall")
            return np.random.choice(self.ACTIONS)

        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_vals = self.q_net(obs_t)

        if torch.isnan(q_vals).any():
            print("[QMIX MarcAgent] NaN in Q-Werten → Zufall")
            a_idx = np.random.randint(self.n_actions)
        else:
            a_idx = int(torch.argmax(q_vals, dim=-1).item())

        return self.ACTIONS[a_idx]
