import os
import random
import numpy as np
import torch
import torch.nn as nn
import logging

from agents.base_agent import Agent
from environment.actions import Action


logger = logging.getLogger("A1")

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

        #  Kommunikation: Verdachtskarten aus Team-Wahrnehmungen 
        self.pit_suspects = {}     # Dict[(x,y)] = count
        self.wumpus_suspects = {}  # Dict[(x,y)] = count


        # DQN laden (falls vorhanden)
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.q_net.load_state_dict(state_dict)
                self.q_net.eval()
                self.model_loaded = True
                logger.info(f"Modell geladen: {model_path}")
            except Exception as e:
                logger.error(f"Fehler beim Laden von {model_path}: {e}")
                logger.warning("Fallback auf Random-Aktionen.")
        else:
            logger.warning(f"Modell '{model_path}' nicht gefunden. Fallback auf Random.")



    # -----------------------------------------------------
    #  Scheduler setzt Knowledge
    # -----------------------------------------------------
    def set_memory(self, known, safe, risky, grid_size):
        self.known = set(known)
        self.safe = set(safe)
        # Wenn ein Feld safe ist, kann es weder Pit noch Wumpus sein -> Verdacht löschen
        for p in list(self.pit_suspects.keys()):
            if p in self.safe:
                del self.pit_suspects[p]
        for p in list(self.wumpus_suspects.keys()):
            if p in self.safe:
                del self.wumpus_suspects[p]

        self.risky = set(risky)
        self.grid_size = grid_size

        logger.debug(
            f"set_memory: grid_size={grid_size}, "
            f"known={len(self.known)}, safe={len(self.safe)}, risky={len(self.risky)}"
        )

    def _neighbors4(self, x, y):
        # grid_size kommt über set_memory rein
        g = getattr(self, "grid_size", None)
        cand = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
        if not g:
            return cand
        return [(nx, ny) for (nx, ny) in cand if 0 <= nx < g and 0 <= ny < g]


    def receive_messages(self, msgs):
    
        for m in msgs:
            # kompatibel mit Message-Objekt oder dict
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

        logger.debug(
            f"decide_move: pos={self.pos()}, obs={observation}, grid_size={grid_size}"
        )

        # 1. Versuche Regel-basiert
        rule_action = self._rule_based_action(observation)
        if rule_action is not None:
            logger.debug(f"decide_move -> RULE action={rule_action}")
            return rule_action

        # 2. DQN oder Random (Fallback)
        if not self.model_loaded:
            a_idx = random.randrange(self.n_actions)
            action = self._idx_to_action(a_idx)
            logger.debug(f"decide_move -> RANDOM (kein Modell) idx={a_idx}, action={action}")
            return action

        with torch.no_grad():
            obs_t = self._obs_to_tensor(observation)
            q_vals = self.q_net(obs_t)
            action_idx = int(q_vals.argmax(dim=-1).item())
            action = self._idx_to_action(action_idx)

        logger.debug(
            f"decide_move -> RL action_idx={action_idx}, action={action}, "
            f"q_vals={q_vals.cpu().numpy().round(3).tolist()}"
        )
        return action



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
            logger.warning(f"_idx_to_action: ungültiger idx={idx}, fallback random move")
            return random.choice(mapping[:4])
        return mapping[idx]


    # -----------------------------------------------------
    #  Regelbasierte Policy (Safe-Explorer)
    # -----------------------------------------------------
    def _rule_based_action(self, observation):
        if self.grid_size is None or len(observation) < 5:
            return None

        x_norm, y_norm, breeze, stench, glitter = observation[:5]
        x, y = self.pos()

        logger.debug(
            f"_rule_based_action: pos={self.pos()}, "
            f"breeze={breeze}, stench={stench}, glitter={glitter}, "
            f"known={len(self.known)}, safe={len(self.safe)}, risky={len(self.risky)}"
        )

        # Wenn Gold glitzert → GRAB
        if glitter >= 0.5:
            logger.debug("_rule_based_action: glitter detected -> GRAB")
            return Action.GRAB

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

        # FLUCHT-REGEL: Feld ist gefährlich -> versuche zu SAFE zu flüchten
        if (breeze >= 0.5 or stench >= 0.5) and len(self.safe) > 0:
            safe_moves = [act for act, pos in valid.items() if pos in self.safe]
            if safe_moves:
                chosen = random.choice(safe_moves)
                self.last_pos = self.pos()
                logger.debug(
                    f"_rule_based_action: DANGER-FIELD -> FLIGHT to safe via {chosen}"
                )
                return chosen

        best_score = -1e9
        best_actions = []

        for act, pos in valid.items():
            score = 0.0

            # risky vermeiden (wenn wir davon wissen)
            if pos in self.risky:
                score -= 50.0

            # safe bevorzugen
            if pos in self.safe:
                score += 10.0

            # unknown stark erkunden
            if pos not in self.known:
                score += 25.0

            # kein Hin-und-Her
            if self.last_pos is not None and pos == self.last_pos:
                score -= 5.0

            # Kommunikation: Verdachtsfelder meiden (soft, nicht absolut)
            pit_c = self.pit_suspects.get(pos, 0)
            wum_c = self.wumpus_suspects.get(pos, 0)

            score -= 15.0 * pit_c
            score -= 25.0 * wum_c


            logger.debug(
                f"_rule_based_action: neighbor={pos}, act={act}, score={score}, "
                f"is_safe={pos in self.safe}, is_risky={pos in self.risky}, "
                f"is_known={pos in self.known}"
                f"pit_sus={pit_c}, wum_sus={wum_c}"
            )

            if score > best_score:
                best_score = score
                best_actions = [act]
            elif score == best_score:
                best_actions.append(act)

        # Wenn ALLE schlecht sind → DQN entscheiden lassen
        if best_score <= -40:
            logger.debug("_rule_based_action: all scores very bad -> fallback RL")
            return None

        # Wenn mehrere gleich gut → zufällig
        if len(best_actions) > 1:
            chosen = random.choice(best_actions)
            self.last_pos = self.pos()
            logger.debug(
                f"_rule_based_action: tie between {best_actions} -> random chosen {chosen}"
            )
            return chosen

        # Eine beste Action
        chosen = best_actions[0]
        self.last_pos = self.pos()
        logger.debug(
            f"_rule_based_action: best_action={chosen} with score={best_score}"
        )
        return chosen

