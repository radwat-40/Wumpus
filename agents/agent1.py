import os
import random
import numpy as np
import torch
import torch.nn as nn

from agents.base_agent import Agent
from environment.actions import Action


# ---------------------------------------------------------
#  DQN-Netz
# ---------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
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


# ---------------------------------------------------------
#  A1 = MarcAgent (Hybrid: Regeln + DQN-Fallback)
# ---------------------------------------------------------
class MarcAgent(Agent):

    def __init__(
        self,
        x: int,
        y: int,
        role: str,
        model_path: str = "dqn_agent_best.pth",
        obs_dim: int = 6,
    ):
        super().__init__(x, y, role)

        self.n_actions = 4  # UP, DOWN, LEFT, RIGHT
        self.obs_dim = obs_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DQN(self.obs_dim, self.n_actions).to(self.device)
        self.model_loaded = False

        # Knowledge vom Scheduler
        self.known = set()
        self.safe = set()
        self.risky = set()
        self.grid_size = None
        self.last_pos = None

        # DQN laden (falls vorhanden)
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.q_net.load_state_dict(state_dict)
                self.q_net.eval()
                self.model_loaded = True
                print(f"[A1] Modell geladen: {model_path}")
            except Exception as e:
                print(f"[A1] Fehler beim Laden: {e}")
                print("[A1] --> Fallback: Random")
        else:
            print(f"[A1] WARNUNG: Modell '{model_path}' nicht gefunden. Fallback auf Random.")


    # -----------------------------------------------------
    #  Scheduler setzt Knowledge
    # -----------------------------------------------------
    def set_memory(self, known, safe, risky, grid_size):
        self.known = set(known)
        self.safe = set(safe)
        self.risky = set(risky)
        self.grid_size = grid_size


    # -----------------------------------------------------
    #  Observation → Tensor
    # -----------------------------------------------------
    def _obs_to_tensor(self, observation):
        arr = np.array(observation, dtype=np.float32)
        return torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)


    # -----------------------------------------------------
    #  Hybrid-Entscheidung
    # -----------------------------------------------------
    def decide_move(self, observation, grid_size):
        if self.grid_size is None:
            self.grid_size = grid_size

        # 1. Versuche Regel-basiert
        rule_action = self._rule_based_action(observation)
        if rule_action is not None:
            return rule_action

        # 2. DQN oder Random (Fallback)
        if not self.model_loaded:
            return self._idx_to_action(random.randrange(self.n_actions))

        with torch.no_grad():
            obs_t = self._obs_to_tensor(observation)
            q_vals = self.q_net(obs_t)
            action_idx = int(q_vals.argmax(dim=-1).item())

        return self._idx_to_action(action_idx)


    # -----------------------------------------------------
    # Mapping Index → Action
    # -----------------------------------------------------
    def _idx_to_action(self, idx: int) -> Action:
        mapping = [
            Action.MOVE_UP,
            Action.MOVE_DOWN,
            Action.MOVE_LEFT,
            Action.MOVE_RIGHT,
            Action.GRAB,
        ]
        idx = int(idx)
        if idx < 0 or idx >= len(mapping):
            return random.choice(mapping[:4])
        return mapping[idx]


    # -----------------------------------------------------
    #  Regelbasierte Policy (Safe-Explorer)
    # -----------------------------------------------------
    def _rule_based_action(self, observation):
        if self.grid_size is None:
            return None

        if len(observation) < 5:
            return None

        x_norm, y_norm, breeze, stench, glitter = observation[:5]

        # Wenn Gold glitzert → GRAB
        if glitter >= 0.5:
            return Action.GRAB

        x, y = self.pos()

        neighbors = {
            Action.MOVE_UP:    (x, y - 1),
            Action.MOVE_DOWN:  (x, y + 1),
            Action.MOVE_LEFT:  (x - 1, y),
            Action.MOVE_RIGHT: (x + 1, y),
        }

        # gültige Moves behalten
        valid = {
            act: (nx, ny)
            for act, (nx, ny) in neighbors.items()
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size
        }

        # **FLUCHT-REGEL**: Wenn aktuell auf gefährlichem Feld (Breeze/Stench) → zurück zu SAFE
        if (breeze >= 0.5 or stench >= 0.5) and len(self.safe) > 0:
            safe_moves = [act for act, pos in valid.items() if pos in self.safe]
            if safe_moves:
                chosen = random.choice(safe_moves)
                self.last_pos = self.pos()
                return chosen

        best_score = -1e9
        best_actions = []

        for act, pos in valid.items():
            score = 0.0

            # risky vermeiden (wenn wir davon wissen)
            if pos in self.risky:
                score -= 50.0  # Sehr hohe Strafe für bekannt gefährlich!

            # safe bevorzugen (Rückzug)
            if pos in self.safe:
                score += 10.0

            # unknown STARK erkunden (nur wenn NICHT auf Breeze/Stench!)
            if pos not in self.known:
                score += 25.0

            # kein Hin-und-Her
            if self.last_pos is not None and pos == self.last_pos:
                score -= 5.0

            if score > best_score:
                best_score = score
                best_actions = [act]
            elif score == best_score:
                best_actions.append(act)

        # Wenn ALLE schlecht sind → DQN entscheiden lassen
        if best_score <= -40:
            return None

        # Wenn mehrere gleich gut → wähle zufällig
        if len(best_actions) > 1:
            chosen = random.choice(best_actions)
            self.last_pos = self.pos()
            return chosen

        # Eine beste Action
        chosen = best_actions[0]
        self.last_pos = self.pos()
        return chosen
