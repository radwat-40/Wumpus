import os
import random
import numpy as np
import torch
import torch.nn as nn

from agents.base_agent import Agent
from environment.actions import Action


class DQN(nn.Module):
    """
    Gleiches Netz wie im Training (qmix_training.py).
    Input: 5-dim State
    Output: Q-Werte für 5 Aktionen
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

    def forward(self, x):
        return self.net(x)


class MarcAgent(Agent):
    """
    A1: DQN-basiert, lädt dqn_agent_best.pth und nutzt
    die 5D-Observation aus dem Scheduler:

      [x_norm, y_norm, breeze, stench, glitter]
    """

    def __init__(
        self,
        x: int,
        y: int,
        role: str,
        model_path: str = "dqn_agent_best.pth",
        obs_dim: int = 5,
    ):
        super().__init__(x, y, role)

        self.n_actions = 5  # 4 Moves + GRAB
        self.obs_dim = obs_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DQN(self.obs_dim, self.n_actions).to(self.device)

        self.model_loaded = False

        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.q_net.load_state_dict(state_dict)
                self.q_net.eval()
                self.model_loaded = True
                print(f"[DQN MarcAgent] Modell geladen: {model_path}")
            except Exception as e:
                print(f"[DQN MarcAgent] Fehler beim Laden von {model_path}: {e}")
                print("[DQN MarcAgent] Fallback auf Zufallsaktionen.")
        else:
            print(f"[DQN MarcAgent] WARNUNG: {model_path} nicht gefunden – Fallback auf Zufallsaktionen.")

    def _obs_to_tensor(self, observation):
        """
        Erwartet denselben 5D-Vektor, den der Scheduler für A1 baut.
        """
        obs = np.array(observation, dtype=np.float32)
        t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return t

    def _idx_to_action(self, idx: int) -> Action:
        """
        Map Netzwerk-Action-Index -> World-Action.
        0..3: Bewegungen, 4: GRAB.
        """
        mapping = [
            Action.MOVE_UP,
            Action.MOVE_DOWN,
            Action.MOVE_LEFT,
            Action.MOVE_RIGHT,
            Action.GRAB,
        ]
        idx = int(idx)
        if idx < 0 or idx >= len(mapping):
            # Sicherheitsfallback
            return random.choice(mapping[:4])
        return mapping[idx]

    def decide_move(self, observation, grid_size: int):
        """
        Wird vom Scheduler aufgerufen.
        observation: 5D-Vektor [x_norm, y_norm, breeze, stench, glitter].
        """
        # Wenn kein Modell geladen werden konnte: pure Random-Policy
        if not self.model_loaded:
            # leichte Exploration auch im Fallback
            return self._idx_to_action(random.randrange(self.n_actions))

        with torch.no_grad():
            obs_t = self._obs_to_tensor(observation)
            q_vals = self.q_net(obs_t)
            action_idx = int(q_vals.argmax(dim=-1).item())

        return self._idx_to_action(action_idx)
