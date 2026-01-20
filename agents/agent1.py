import os
import random
import numpy as np
import torch
import torch.nn as nn
import logging

from agents.base_agent import Agent
from environment.actions import Action


logger = logging.getLogger("A1")

# DQN Modell
class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),    # Input Layer
            nn.ReLU(),                 # Aktivierung
            nn.Linear(64, 64),         # Hidden Layer
            nn.ReLU(),                 # Aktivierung
            nn.Linear(64, n_actions),  # Output Layer
        )

    def forward(self, x):
        return self.net(x)


# Hybrid Agent
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

        self.n_actions = 4  # Aktionsanzahl
        self.obs_dim = obs_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DQN(self.obs_dim, self.n_actions).to(self.device)
        self.model_loaded = False

        # Wissensmengen
        self.known = set()
        self.safe = set()
        self.risky = set()
        self.grid_size = None
        self.last_pos = None

        # Team Verdacht
        self.pit_suspects = {}     # Pit Zähler
        self.wumpus_suspects = {}  # Wumpus Zähler

        # Modell laden
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.q_net.load_state_dict(state_dict)
                self.q_net.eval()
                self.model_loaded = True
                logger.info(f"Modell geladen: {model_path}")
            except Exception as e:
                logger.error(f"Fehler beim Laden von {model_path}: {e}")
                logger.warning("Random Fallback")
        else:
            logger.warning(f"Modell nicht gefunden: {model_path}")

    def set_memory(self, known, safe, risky, grid_size):
        self.known = set(known)
        self.safe = set(safe)

        # Verdacht entfernen
        for p in list(self.pit_suspects.keys()):
            if p in self.safe:
                del self.pit_suspects[p]
        for p in list(self.wumpus_suspects.keys()):
            if p in self.safe:
                del self.wumpus_suspects[p]

        self.risky = set(risky)
        self.grid_size = grid_size

    def _neighbors4(self, x, y):
        # Nachbarfelder
        g = getattr(self, "grid_size", None)
        cand = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
        if not g:
            return cand
        return [(nx, ny) for (nx, ny) in cand if 0 <= nx < g and 0 <= ny < g]

    def receive_messages(self, msgs):
        # Team Infos
        for m in msgs:
            topic = getattr(m, "topic", None) or (m.get("topic") if isinstance(m, dict) else None)
            payload = getattr(m, "payload", None) or (m.get("payload") if isinstance(m, dict) else None) or {}
            pos = payload.get("pos", None)

            if not topic or not pos:
                continue

            x, y = pos
            t = topic.lower()

            if "breeze_detected" in t:
                for nb in self._neighbors4(x, y):
                    if nb in self.safe:
                        continue
                    self.pit_suspects[nb] = self.pit_suspects.get(nb, 0) + 1

            elif "stench_detected" in t:
                for nb in self._neighbors4(x, y):
                    if nb in self.safe:
                        continue
                    self.wumpus_suspects[nb] = self.wumpus_suspects.get(nb, 0) + 1

    def _obs_to_tensor(self, observation):
        # Tensor bauen
        arr = np.array(observation, dtype=np.float32)
        return torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)

    def decide_move(self, observation, grid_size):
        if self.grid_size is None:
            self.grid_size = grid_size

        # Regel prüfen
        rule_action = self._rule_based_action(observation)
        if rule_action is not None:
            return rule_action

        # Random Fallback
        if not self.model_loaded:
            a_idx = random.randrange(self.n_actions)
            return self._idx_to_action(a_idx)

        # DQN Entscheidung
        with torch.no_grad():
            obs_t = self._obs_to_tensor(observation)
            q_vals = self.q_net(obs_t)
            action_idx = int(q_vals.argmax(dim=-1).item())
            return self._idx_to_action(action_idx)

    def _idx_to_action(self, idx: int) -> Action:
        # Index Mapping
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

    def _rule_based_action(self, observation):
        # Regel Policy
        if self.grid_size is None or len(observation) < 5:
            return None

        x_norm, y_norm, breeze, stench, glitter = observation[:5]
        x, y = self.pos()

        # Gold greifen
        if glitter >= 0.5:
            return Action.GRAB

        neighbors = {
            Action.MOVE_UP:    (x, y - 1),
            Action.MOVE_DOWN:  (x, y + 1),
            Action.MOVE_LEFT:  (x - 1, y),
            Action.MOVE_RIGHT: (x + 1, y),
        }

        # Gültige Züge
        valid = {
            act: (nx, ny)
            for act, (nx, ny) in neighbors.items()
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size
        }

        # Fluchtregel
        if (breeze >= 0.5 or stench >= 0.5) and len(self.safe) > 0:
            safe_moves = [act for act, pos in valid.items() if pos in self.safe]
            if safe_moves:
                self.last_pos = self.pos()
                return random.choice(safe_moves)

        best_score = -1e9
        best_actions = []

        for act, pos in valid.items():
            score = 0.0

            # Risiko meiden
            if pos in self.risky:
                score -= 50.0

            # Sicherheit bevorzugen
            if pos in self.safe:
                score += 10.0

            # Neues erkunden
            if pos not in self.known:
                score += 25.0

            # Zurück vermeiden
            if self.last_pos is not None and pos == self.last_pos:
                score -= 5.0

            # Verdacht einbeziehen
            score -= 15.0 * self.pit_suspects.get(pos, 0)
            score -= 25.0 * self.wumpus_suspects.get(pos, 0)

            if score > best_score:
                best_score = score
                best_actions = [act]
            elif score == best_score:
                best_actions.append(act)

        # RL Fallback
        if best_score <= -40:
            return None

        # Zufall bei Gleichstand
        if len(best_actions) > 1:
            self.last_pos = self.pos()
            return random.choice(best_actions)

        # Beste Aktion
        self.last_pos = self.pos()
        return best_actions[0]
