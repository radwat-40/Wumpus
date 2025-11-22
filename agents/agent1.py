import os
import numpy as np
import torch
import torch.nn as nn

from agents.base_agent import Agent
from environment.actions import Action


class AgentQNetwork(nn.Module):
    """
    Per-Agent Q-Netzwerk: Q_i(a | o_i)
    o_i hat obs_dim Features (hier: 5)
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
        # obs: (B, obs_dim)
        return self.net(obs)


class MarcAgent(Agent):
    """
    QMIX-Inference-Agent für A1.

    Erwartet als Observation einen Vektor:
      [x_norm, y_norm, breeze, stench, glitter]  (Shape: (5,))

    Aktionen: nur Bewegung (UP/DOWN/LEFT/RIGHT).
    Das entspricht dem Setup aus qmix_training.py.
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

        self.n_actions = len(self.ACTIONS)
        self.obs_dim = obs_dim

        # Q-Netzwerk (gleiche Architektur wie im Training)
        self.q_net = AgentQNetwork(self.obs_dim, self.n_actions).to(self.device)

        self.agent_alive = True
        self.agent_won = False

        # Modell laden
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.q_net.load_state_dict(state_dict)
                print(f"[QMIX MarcAgent] Modell geladen: {model_path}")
            except Exception as e:
                print(f"[QMIX MarcAgent] FEHLER beim Laden von {model_path}: {e}")
                print("[QMIX MarcAgent] Fallback: zufällige Aktionen.")
        else:
            print(f"[QMIX MarcAgent] WARNUNG: {model_path} nicht gefunden – Fallback: zufällige Aktionen.")

        self.q_net.eval()

    def pos(self):
        return (self.x, self.y)

    def decide_move(self, observation, grid_size: int):
        """
        Wird vom Scheduler aufgerufen.
        observation: np.array shape (5,) = [x_norm, y_norm, breeze, stench, glitter]
        grid_size: aktuell ungenutzt, nur für Kompatibilität.
        """

        # Safety: Observation in numpy float32 bringen
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)

        if obs.shape[0] != self.obs_dim:
            # Fallback falls irgendwas schief läuft
            print(f"[QMIX MarcAgent] Ungültige Obs-Shape {obs.shape}, erwarte ({self.obs_dim},). → random move")
            return np.random.choice(self.ACTIONS)

        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # (1, obs_dim)

        with torch.no_grad():
            q_vals = self.q_net(obs_t)  # (1, n_actions)

        if torch.isnan(q_vals).any():
            print("[QMIX MarcAgent] NaN in Q-Werten → random move")
            a_idx = np.random.randint(self.n_actions)
        else:
            a_idx = int(torch.argmax(q_vals, dim=-1).item())

        return self.ACTIONS[a_idx]
