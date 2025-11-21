"""Scheduler Klasse für die Verwaltung der Agenten. Bestimmt wer welcher Agent an der Reihe ist,
 was er wahrnimmt und welche Aktion er ausführt.
"""

class Scheduler:
    def __init__(self, agents, world):
        self.agents = agents
        self.world = world
        self.turn = 0

        # Memory-Grids für A1 (PPO)
        self.known = None
        self.safe  = None
        self.risky = None

    # ------------------------------------------------------
    # MemoryGrid initialisieren
    # ------------------------------------------------------
    def reset_memory(self, grid_size):
        import numpy as np
        self.known = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.safe  = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.risky = np.zeros((grid_size, grid_size), dtype=np.float32)

    # ------------------------------------------------------
    # Memory-Observation für PPO (4, H, W)
    # ------------------------------------------------------
    def _build_memory_observation(self, agent):
        import numpy as np

        ax, ay = agent.x, agent.y

        agent_layer = np.zeros_like(self.safe)
        agent_layer[ay, ax] = 1

        return np.stack([
            1 - self.known,
            self.safe,
            self.risky,
            agent_layer
        ], axis=0)

    # ------------------------------------------------------
    # Patch Observation für A2 / A3
    # ------------------------------------------------------
    def _build_observation(self, agent, patch_size=5):
        half = patch_size // 2
        gx = agent.x
        gy = agent.y
        grid = self.world

        obs = []
        for channel in ('breeze_tiles', 'stench_tiles', 'gold'):
            layer = [[0 for _ in range(patch_size)] for __ in range(patch_size)]
            for dx in range(-half, half + 1):
                for dy in range(-half, half + 1):
                    x = gx + dx
                    y = gy + dy
                    ix = dx + half
                    iy = dy + half

                    if 0 <= x < grid.grid_size and 0 <= y < grid.grid_size:
                        if channel == 'breeze_tiles' and (x, y) in grid.breeze_tiles:
                            layer[iy][ix] = 1
                        if channel == 'stench_tiles' and (x, y) in grid.stench_tiles:
                            layer[iy][ix] = 1
                        if channel == 'gold' and (x, y) in grid.gold:
                            layer[iy][ix] = 1

            obs.append(layer)

        # Agent position channel
        pos_layer = [[0 for _ in range(patch_size)] for __ in range(patch_size)]
        pos_layer[half][half] = 1
        obs.append(pos_layer)

        import numpy as _np
        return _np.array(obs, dtype=_np.float32)

    # ------------------------------------------------------
    # MAIN STEP
    # ------------------------------------------------------
    def step(self):
        # Falls alle Agenten tot
        if not any(getattr(a, 'agent_alive', True) for a in self.agents):
            return "ALL_DEAD"

        # Finde den nächsten lebenden Agenten
        start = self.turn
        while not getattr(self.agents[self.turn], 'agent_alive', True):
            self.turn = (self.turn + 1) % len(self.agents)
            if self.turn == start:
                return "ALL_DEAD"

        agent = self.agents[self.turn]

        if not agent.agent_alive:
            self.turn = (self.turn + 1) % len(self.agents)
            return "CONTINUE"

        percepts = self.world.get_percepts(agent)
        old_pos = agent.pos()

        # ------------------------------------------------------
        # A1 (PPO) → MemoryGrid (4,H,W)
        # Andere → Patch
        # ------------------------------------------------------
        if agent.role == "A1":
            observation = self._build_memory_observation(agent)
        else:
            observation = self._build_observation(agent, patch_size=5)

        # Agent entscheidet
        try:
            action = agent.decide_move(observation, self.world.grid_size)
        except TypeError:
            action = agent.decide_move(percepts, self.world.grid_size)

        # ------------------------------------------------------
        # Memory aktualisieren (nur A1)
        # ------------------------------------------------------
        if agent.role == "A1":
            ax, ay = agent.pos()
            self.known[ay, ax] = 1

            p = self.world.get_percepts(agent)
            if p["breeze"] or p["stench"]:
                self.risky[ay, ax] = 1
            else:
                self.safe[ay, ax] = 1

        # Welt führt aus
        result = self.world.execute(agent, action)

        # Neue Percepts
        next_percepts = self.world.get_percepts(agent)
        new_pos = agent.pos()

        # Next Observation
        if agent.role == "A1":
            next_observation = self._build_memory_observation(agent)
        else:
            next_observation = self._build_observation(agent, patch_size=5)

        # DQN-Update entfällt → PPO macht kein learn_step im Spiel

        # Nächster Agent
        self.turn = (self.turn + 1) % len(self.agents)

        return result
