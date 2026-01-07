from agents.base_agent import Agent
from environment.actions import Action
import random
import logging
from typing import Tuple, List, Dict, Optional
import os
import pickle
from collections import defaultdict, deque
import math

logger = logging.getLogger("A2")


class YahiaAgent(Agent):
    def __init__(self, x: int, y: int, role: str):
        super().__init__(x, y, role)

        # exploration / bookkeeping
        self.visited = set()
        self.visit_counts = defaultdict(int)
        self.recent_positions = deque(maxlen=12)  # last K positions to discourage cycles
        self.last_action: Optional[Action] = None
        self.last_pos: Optional[Tuple[int, int]] = None

        # world/grid maps
        self.grid_size: Optional[int] = None
        self.breeze_map: Optional[List[List[int]]] = None
        self.stench_map: Optional[List[List[int]]] = None

        # message inbox
        self.received_messages = []

        # Q-learning per component
        self.components = ["gold", "pit", "wumpus"]
        # Q: Dict[component -> Dict[(state, action) -> value]]
        self.Q: Dict[str, Dict[Tuple[Tuple[int, int], Action], float]] = {c: {} for c in self.components}

        # RL hyperparams
        self.alpha = 0.3
        self.gamma = 0.99
        self.epsilon = 0.9
        self.epsilon_decay = 0.9995
        self.min_epsilon = 0.1
        self.init_q = 1.0

        self.actions: List[Action] = [Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT]

        # Anti-oscillation / visitation penalties
        self.visit_penalty_beta = 0.8
        self.cycle_penalty = 5.0

        # Softmax / Boltzmann sampling
        self.use_softmax = True
        self.temperature = 1.0
        self.temperature_decay = 0.9999
        self.min_temperature = 0.1

        # Percept-based soft penalties (breeze/stench)
        self.breeze_penalty = 1.0
        self.stench_penalty = 1.0

        # Inference for pits/wumpus (pragmatic count-based)
        self.pit_suspect_counts = defaultdict(int)
        self.wumpus_suspect_counts = defaultdict(int)
        self.confirmed_pits = set()
        self.confirmed_wumpus = set()
        self.safe_cells = set()
        self.confirm_threshold = 1  # threshold to confirm a suspect

    # -------------------------
    # Map initialization
    # -------------------------
    def init_maps(self, grid_size: int):
        self.grid_size = grid_size
        self.breeze_map = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        self.stench_map = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        logger.info(f"[A2] initialized maps size = {grid_size}x{grid_size}")

    # -------------------------
    # Map setters (row=y,col=x)
    # -------------------------
    def set_breeze_at(self, pos: Tuple[int, int]):
        if pos is None or self.breeze_map is None or self.grid_size is None:
            return
        x, y = pos
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            # indexing is row=y, col=x
            self.breeze_map[y][x] = 1
            logger.debug(f"[A2] breeze_map set 1 at {(x, y)}")

    def set_stench_at(self, pos: Tuple[int, int]):
        if pos is None or self.stench_map is None or self.grid_size is None:
            return
        x, y = pos
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.stench_map[y][x] = 1
            logger.debug(f"[A2] stench_map set 1 at {(x, y)}")


    # -------------------------
    # Message handling
    # -------------------------
    def receive_messages(self, messages):
        if not messages:
            return
        update = False
        for m in messages:
            topic = (getattr(m, "topic", "") or "").lower()
            payload = getattr(m, "payload", {}) or {}
            sender = getattr(m, "sender", None)

            pos = payload.get("pos")
            if pos is not None:
                try:
                    pos = tuple(pos)
                except Exception:
                    pos = None

            logger.info(f"[A2] receive_messages: from={sender} topic={topic} pos={pos}")

            if topic.startswith("breeze"):
                self.set_breeze_at(pos)
                update = True
            elif topic.startswith("stench"):
                self.set_stench_at(pos)
                update = True
            elif topic == "agent_died":
                dead_pos = payload.get("pos") or payload.get("position")
                if dead_pos is not None:
                    try:
                        dead_pos = tuple(dead_pos)
                    except Exception:
                        dead_pos = None
                cause = (payload.get("cause", "") or "").lower()
                if dead_pos is not None:
                    if "pit" in cause:
                        self.confirmed_pits.add(dead_pos)
                    elif "wumpus" in cause:
                        self.confirmed_wumpus.add(dead_pos)
                    else:
                        self.confirmed_pits.add(dead_pos)
                    logger.info(f"[A2] receive_messages: confirmed danger at {dead_pos} cause={cause}")
            else:
                logger.debug(f"[A2] receive_messages: unhandled topic={topic}")

        # after processing all messages
        if update:
            # recompute suspects from the full maps (deduplicates implicit duplicate broadcasts)
            self.recompute_suspects_from_maps()
            self.print_maps_console()

    # -------------------------
    # Helpers for inference
    # -------------------------
    def _neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        if pos is None or self.grid_size is None:
            return []
        x, y = pos
        res = []
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                res.append((nx, ny))
        return res

    def _mark_suspects_from_breeze(self, pos: Tuple[int, int]):
        if pos is None:
            return
        for n in self._neighbors(pos):
            if n in self.safe_cells:
                continue
            self.pit_suspect_counts[n] += 1
            if self.pit_suspect_counts[n] >= self.confirm_threshold:
                if n not in self.confirmed_pits:
                    logger.info(f"[A2] confirmed PIT at {n} (from breeze evidence)")
                self.confirmed_pits.add(n)

    def _mark_suspects_from_stench(self, pos: Tuple[int, int]):
        if pos is None:
            return
        for n in self._neighbors(pos):
            if n in self.safe_cells:
                continue
            self.wumpus_suspect_counts[n] += 1
            if self.wumpus_suspect_counts[n] >= self.confirm_threshold:
                if n not in self.confirmed_wumpus:
                    logger.info(f"[A2] confirmed WUMPUS at {n} (from stench evidence)")
                self.confirmed_wumpus.add(n)

    def _process_no_breeze_no_stench(self, pos: Tuple[int, int]):
        if pos is None:
            return
        for n in self._neighbors(pos):
            if n not in self.safe_cells:
                self.safe_cells.add(n)
            # clear suspects for neighbors
            self.pit_suspect_counts.pop(n, None)
            self.wumpus_suspect_counts.pop(n, None)
            self.confirmed_pits.discard(n)
            self.confirmed_wumpus.discard(n)
        if pos in self.wumpus_suspect_counts:
            self.wumpus_suspect_counts.pop(pos, None)


    def is_confirmed_danger(self, pos: Tuple[int, int]) -> bool:
        return pos in self.confirmed_pits or pos in self.confirmed_wumpus

    # -------------------------
    # Q helpers
    # -------------------------
    def state_repr(self) -> Tuple[int, int]:
        return self.pos()

    def q_get(self, comp: str, state: Tuple[int, int], action: Action) -> float:
        return self.Q[comp].get((state, action), getattr(self, "init_q", 1.0))

    def q_set(self, comp: str, state: Tuple[int, int], action: Action, value: float):
        self.Q[comp][(state, action)] = value

    def normalize_list(self, vals: List[float]) -> List[float]:
        if not vals:
            return vals
        max_abs = max(abs(v) for v in vals)
        if max_abs == 0:
            return [0.0 for _ in vals]
        return [v / max_abs for v in vals]

    def combined_q_values(self, state: Tuple[int, int], valid_actions: List[Action]) -> Dict[Action, float]:
        comp_vals: Dict[str, List[float]] = {}
        for comp in self.components:
            comp_vals[comp] = [self.q_get(comp, state, a) for a in valid_actions]

        comp_norm: Dict[str, List[float]] = {
            comp: self.normalize_list(vals) for comp, vals in comp_vals.items()
        }

        q_total: Dict[Action, float] = {}
        for i, a in enumerate(valid_actions):
            total = 0.0
            for comp in self.components:
                total += self.weights.get(comp, 1.0) * comp_norm[comp][i]
            q_total[a] = total
        return q_total

    # -------------------------
    # Action selection: softmax + penalties
    # -------------------------
    def _softmax_sample(self, scores: List[float], actions: List[Action], temp: float) -> Action:
        if not scores:
            return random.choice(actions)
        max_s = max(scores)
        exps = [math.exp((s - max_s) / max(temp, 1e-6)) for s in scores]
        ssum = sum(exps)
        if ssum == 0:
            return random.choice(actions)
        probs = [e / ssum for e in exps]
        r = random.random()
        cum = 0.0
        for p, a in zip(probs, actions):
            cum += p
            if r <= cum:
                return a
        return actions[-1]

    def _risk_penalty_for(self, pos: Tuple[int, int]) -> float:
        if pos is None or self.breeze_map is None or self.stench_map is None or self.grid_size is None or pos in self.visited:
            return 0.0
        x, y = pos
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return 0.0
        try:
            b = 1 if self.breeze_map[y][x] else 0
            s = 1 if self.stench_map[y][x] else 0
            return b * self.breeze_penalty + s * self.stench_penalty
        except Exception:
            return 0.0

    def detect_deadlock(self):
        if len(self.recent_positions) < 6:
            return False
        return len(set(self.recent_positions)) <= 2

    def select_action(self, state: Tuple[int, int], valid_actions: List[Action]) -> Action:
        x, y = state
        dxdy = {
                Action.MOVE_UP: (0, -1),
                Action.MOVE_DOWN: (0, 1),
                Action.MOVE_LEFT: (-1, 0),
                Action.MOVE_RIGHT: (1, 0)
            }
        
        if self.detect_deadlock():
            self.epsilon = 1.0
            self.temperature = 2.0

        # exploration: epsilon-greedy prefers least-visited neighbouring cells
        if random.random() < self.epsilon:
            min_vis = None
            candidates = []
            for a in valid_actions:
                dx, dy = dxdy[a]
                ns = (x + dx, y + dy)
                cnt = self.visit_counts.get(ns, 0)
                if min_vis is None or cnt < min_vis:
                    min_vis = cnt
                    candidates = [a]
                elif cnt == min_vis:
                    candidates.append(a)
            return random.choice(candidates)

        # exploitation: build combined Q and apply penalties
        q_total = self.combined_q_values(state, valid_actions)

        dxdy = {
            Action.MOVE_UP: (0, -1),
            Action.MOVE_DOWN: (0, 1),
            Action.MOVE_LEFT: (-1, 0),
            Action.MOVE_RIGHT: (1, 0)
        }

        scores = []
        actions_order = []
        for a in valid_actions:
            dx, dy = dxdy[a]
            ns = (x + dx, y + dy)

            visit_pen = self.visit_penalty_beta * self.visit_counts.get(ns, 0)
            cyc_pen = self.cycle_penalty if ns in self.recent_positions else 0.0
            # larger penalty for confirmed danger (hard avoidance is implemented in decide_move,
            # but also penalize here to prefer other options when unavoidable)
            confirmed_pen = 9999.0 if self.is_confirmed_danger(ns) else 0.0
            risk_pen = self._risk_penalty_for(ns)
            adjusted = q_total.get(a, 0.0) - visit_pen - cyc_pen - risk_pen - confirmed_pen
            scores.append(adjusted)
            actions_order.append(a)

        if self.use_softmax:
            act = self._softmax_sample(scores, actions_order, self.temperature)
            self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)
            return act

        # deterministic fallback
        best_val = max(scores)
        best_actions = [a for a, s in zip(actions_order, scores) if s == best_val]
        if len(best_actions) == 1:
            return best_actions[0]

        # tie-break by least visited next state
        min_vis = None
        candidates = []
        for a in best_actions:
            dx, dy = dxdy[a]
            ns = (x + dx, y + dy)
            cnt = self.visit_counts.get(ns, 0)
            if min_vis is None or cnt < min_vis:
                min_vis = cnt
                candidates = [a]
            elif cnt == min_vis:
                candidates.append(a)
        return random.choice(candidates)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    # -------------------------
    # Q update
    # -------------------------
    def update_component_qs(
        self,
        state: Tuple[int, int],
        action: Action,
        next_state: Tuple[int, int],
        rewards: Dict[str, float]
    ):
        # small exploration bonus for never-visited states
        exploration_bonus = 1.0 if next_state not in self.visited else 0.0

        for comp in self.components:
            r_k = rewards.get(comp, 0.0)
            old = self.q_get(comp, state, action)
            next_vals = [self.q_get(comp, next_state, a2) for a2 in self.actions]
            next_max = max(next_vals) if next_vals else 0.0
            target = (r_k + exploration_bonus) + self.gamma * next_max
            new_val = old + self.alpha * (target - old)
            self.q_set(comp, state, action, new_val)

    # -------------------------
    # Persistence
    # -------------------------
    def save_q_tables(self, path: str):
        try:
            dirp = os.path.dirname(path)
            if dirp and not os.path.exists(dirp):
                os.makedirs(dirp, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({
                    "Q": self.Q,
                    "weights": getattr(self, "weights", {"gold": 1.0, "pit": 1.0, "wumpus": 1.0}),
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    # do NOT persist confirmed_pits/confirmed_wumpus (recompute from maps each run)
                }, f)
            logger.info(f"[A2] Q-tables saved to {path}")
        except Exception:
            logger.exception("[A2] Failed to save Q-tables")

    def load_q_tables(self, path: str) -> bool:
        if not os.path.exists(path):
            logger.debug(f"[A2] Q file not found: {path}")
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.Q = data.get("Q", self.Q)
            self.weights = data.get("weights", getattr(self, "weights", {"gold": 1.0, "pit": 1.0, "wumpus": 1.0}))
            self.alpha = data.get("alpha", self.alpha)
            self.gamma = data.get("gamma", self.gamma)
            self.epsilon = data.get("epsilon", self.epsilon)
            logger.info(f"[A2] Q-tables loaded from {path}")
            return True
        except Exception:
            logger.exception("[A2] Failed to load Q-tables")
            return False

    # -------------------------
    # Decision / move
    # -------------------------
    def decide_move(self, percep, grid_size):
        x, y = self.pos()
        self.visited.add((x, y))
        self.visit_counts[(x, y)] += 1

        # if percep is dict, apply safe inference when no breeze/stench present
        if isinstance(percep, dict):
            breeze_flag = bool(percep.get("breeze", 0) or percep.get("brezze", 0))
            stench_flag = bool(percep.get("stench", 0))
            if not breeze_flag and not stench_flag:
                # mark neighbors safe
                self._process_no_breeze_no_stench((x, y))

        # deliver any buffered messages first
        if self.received_messages:
            self.receive_messages(self.received_messages)

        # neighbors map
        moves_map = {
            Action.MOVE_UP:    (x, y - 1),
            Action.MOVE_DOWN:  (x, y + 1),
            Action.MOVE_LEFT:  (x - 1, y),
            Action.MOVE_RIGHT: (x + 1, y)
        }

        # valid actions within grid
        valid_actions = [a for a, p in moves_map.items() if 0 <= p[0] < grid_size and 0 <= p[1] < grid_size]

        if not valid_actions:
            return random.choice(list(moves_map.keys()))

        # HARD avoid: remove confirmed pit/wumpus moves if alternatives exist
        safe_actions = [a for a in valid_actions if not self.is_confirmed_danger(moves_map[a])]
        if safe_actions:
            valid_actions = safe_actions

        # compute state and select action
        state = self.state_repr()
        selected_action = self.select_action(state, valid_actions)

        # update last_action / recent positions
        self.last_action = selected_action
        self.last_pos = state
        dxdy = {
            Action.MOVE_UP:    (0, -1),
            Action.MOVE_DOWN:  (0,  1),
            Action.MOVE_LEFT:  (-1, 0),
            Action.MOVE_RIGHT: (1,  0),
        }
        dx, dy = dxdy.get(selected_action, (0, 0))
        next_state = (state[0] + dx, state[1] + dy)
        self.recent_positions.append(next_state)

        self.decay_epsilon()
        return selected_action

    def recompute_suspects_from_maps(self):
        """Recompute pit/wumpus suspect counts and confirmed sets from breeze/stench maps."""
        if self.grid_size is None or self.breeze_map is None or self.stench_map is None:
            return

        from collections import defaultdict
        pit_scores = defaultdict(int)
        wumpus_scores = defaultdict(int)

        # accumulate evidence from all breeze / stench cells
        stench_cells = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.breeze_map[y][x]:
                    for n in self._neighbors((x, y)):
                        if n in self.safe_cells:
                            continue
                        pit_scores[n] += 1
                if self.stench_map[y][x]:
                    stench_cells.append((x, y))
                    for n in self._neighbors((x, y)):
                        if n in self.safe_cells:
                            continue
                        wumpus_scores[n] += 1

        # replace suspect counts (fresh computation avoids duplicate counting issues)
        self.pit_suspect_counts = pit_scores
        self.wumpus_suspect_counts = wumpus_scores

        # confirm pits by threshold
        self.confirmed_pits = {pos for pos, c in pit_scores.items() if c >= self.confirm_threshold}

        # ---- WUMPUS logic ----
        # If multiple stench cells and world likely has single wumpus, intersection is strong evidence.
        # CONFIRM WUMPUS — multi Wumpus safe
        self.confirmed_wumpus = {
            pos for pos, c in wumpus_scores.items()
            if c >= self.confirm_threshold
        }

        logger.debug(
            f"[A2] recomputed suspects: pit_scores={dict(list(pit_scores.items())[:8])} "
            f"wumpus_scores={dict(list(wumpus_scores.items())[:8])} stench_cells={stench_cells}"
        )
        logger.info(f"[A2] confirmed_pits={self.confirmed_pits} confirmed_wumpus={self.confirmed_wumpus}")

    def print_maps_console(self):
        """Print simple breeze/stench maps for debugging (row=y, col=x)."""
        if self.breeze_map is None or self.stench_map is None:
            print("[A2] maps not initialized")
            return
        print("[A2] Breeze map:")
        for row in self.breeze_map:
            print(''.join(str(v) for v in row))
        print("[A2] Stench map:")
        for row in self.stench_map:
            print(''.join(str(v) for v in row))

