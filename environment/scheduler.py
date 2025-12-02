import numpy as np


class Scheduler:
    def __init__(self, agents, world):
        self.agents = agents
        self.world = world
        self.turn = 0

        # MemoryGrid für A1 (QMIX)
        self.known = set()
        self.safe = set()
        self.risky = set()

    def reset_memory(self, grid_size: int):
        # Für Restart
        self.known = set()
        self.safe = set()
        self.risky = set()

    def step(self):
        # Falls alle tot
        if not any(getattr(a, "agent_alive", True) for a in self.agents):
            return "ALL_DEAD"

        start = self.turn
        while not getattr(self.agents[self.turn], "agent_alive", True):
            self.turn = (self.turn + 1) % len(self.agents)
            if self.turn == start:
                return "ALL_DEAD"

        agent = self.agents[self.turn]

        if not getattr(agent, "agent_alive", True):
            self.turn = (self.turn + 1) % len(self.agents)
            return "CONTINUE"

        percepts = self.world.get_percepts(agent)

        # ----- OBSERVATION -----

        if agent.role == "A1":
            observation = self._build_qmix_observation(agent, percepts)
        else:
            observation = self._build_patch_observation(agent)

        # ----- ACTION -----

        try:
            action = agent.decide_move(observation, self.world.grid_size)
        except TypeError:
            action = agent.decide_move(percepts, self.world.grid_size)

        # ----- MEMORYGRID UPDATE BEFORE MOVE -----

        old_pos = agent.pos()
        self._update_memory_before_move(agent, percepts)

        # ----- APPLY MOVE -----

        result = self.world.execute(agent, action)
        new_pos = agent.pos()
        new_percepts = self.world.get_percepts(agent)

        # ----- REWARD SHAPING (nutzt Wissensstand VOR dem Update nach dem Zug) -----

        self._apply_reward_shaping(agent, old_pos, new_pos, percepts, new_percepts)

        # ----- MEMORYGRID UPDATE AFTER MOVE -----

        self._update_memory_after_move(agent, new_percepts)

        # ----- NEXT AGENT -----

        self.turn = (self.turn + 1) % len(self.agents)
        return result

    #                         QMIX OBSERVATION (A1)

    def _build_qmix_observation(self, agent, percepts):
        x, y = agent.pos()
        N = self.world.grid_size - 1

        x_norm = x / N if N > 0 else 0.0
        y_norm = y / N if N > 0 else 0.0

        breeze = 1.0 if percepts["breeze"] else 0.0
        stench = 1.0 if percepts["stench"] else 0.0
        glitter = 1.0 if percepts["glitter"] else 0.0

        return np.array([x_norm, y_norm, breeze, stench, glitter], dtype=np.float32)

    #                         MEMORY GRID UPDATE

    def _update_memory_before_move(self, agent, percepts):
        pos = agent.pos()
        self.known.add(pos)

        if percepts["breeze"] or percepts["stench"]:
            self.risky.add(pos)
        else:
            self.safe.add(pos)

    def _update_memory_after_move(self, agent, percepts):
        pos = agent.pos()
        self.known.add(pos)

        if percepts["breeze"] or percepts["stench"]:
            self.risky.add(pos)
        else:
            self.safe.add(pos)

    def _apply_reward_shaping(self, agent, old_pos, new_pos, old_p, new_p):
        if agent.role != "A1":
            return

        # Neues Feld?
        if new_pos not in self.known:
            agent.reward = getattr(agent, "reward", 0) + 5

        # Safe Feld?
        safe_before = not (old_p["breeze"] or old_p["stench"])
        safe_after = not (new_p["breeze"] or new_p["stench"])
        if safe_before and safe_after:
            agent.reward = getattr(agent, "reward", 0) + 5

        # Risiko Strafe
        if new_pos in self.risky:
            agent.reward = getattr(agent, "reward", 0) - 1

        # Schritt Strafe
        agent.reward = getattr(agent, "reward", 0) - 0.02

    #                 PATCH OBSERVATION (A2 / A3)

    def _build_patch_observation(self, agent, patch_size=5):
        half = patch_size // 2
        gx, gy = agent.pos()
        grid = self.world

        obs = []
        for channel in ("breeze_tiles", "stench_tiles", "gold"):
            layer = [[0 for _ in range(patch_size)] for __ in range(patch_size)]
            for dx in range(-half, half + 1):
                for dy in range(-half, half + 1):
                    x = gx + dx
                    y = gy + dy
                    if not grid.in_bounds(x, y):
                        continue

                    if channel == "breeze_tiles" and (x, y) in grid.breeze_tiles:
                        ix = dx + half
                        iy = dy + half
                        layer[iy][ix] = 1
                    elif channel == "stench_tiles" and (x, y) in grid.stench_tiles:
                        ix = dx + half
                        iy = dy + half
                        layer[iy][ix] = 1
                    elif channel == "gold" and (x, y) in grid.gold:
                        ix = dx + half
                        iy = dy + half
                        layer[iy][ix] = 1
            obs.append(layer)

        pos_layer = [[0 for _ in range(patch_size)] for __ in range(patch_size)]
        pos_layer[half][half] = 1
        obs.append(pos_layer)

        return np.array(obs, dtype=np.float32)
