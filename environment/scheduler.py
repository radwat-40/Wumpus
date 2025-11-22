"""Scheduler Klasse für die Verwaltung der Agenten.
Bestimmt, welcher Agent an der Reihe ist, was er wahrnimmt und welche Aktion er ausführt.
"""

import numpy as np


class Scheduler:
    def __init__(self, agents, world):
        self.agents = agents
        self.world = world
        self.turn = 0

    def reset_memory(self, grid_size: int):
        """
        Platzhalter, damit main.py nicht crasht.
        Früher für PPO-Memory genutzt, jetzt für QMIX-Inference nicht mehr notwendig.
        """
        pass

    def step(self):
        # Falls alle Agenten tot sind
        if not any(getattr(a, 'agent_alive', True) for a in self.agents):
            return "ALL_DEAD"

        # Nächsten lebenden Agenten finden
        start = self.turn
        while not getattr(self.agents[self.turn], 'agent_alive', True):
            self.turn = (self.turn + 1) % len(self.agents)
            if self.turn == start:
                return "ALL_DEAD"

        agent = self.agents[self.turn]

        if not getattr(agent, "agent_alive", True):
            self.turn = (self.turn + 1) % len(self.agents)
            return "CONTINUE"

        # Percepts (für QMIX-Obs und ggf. alte Agents)
        percepts = self.world.get_percepts(agent)

        # Observation bauen
        if getattr(agent, "role", None) == "A1":
            # QMIX-Inference-Agent: 5-d Vektor
            observation = self._build_qmix_observation(agent, percepts)
        else:
            # Alte Agents: Patch-Observation
            observation = self._build_observation(agent, patch_size=5)

        # Agent entscheidet Aktion
        try:
            action = agent.decide_move(observation, self.world.grid_size)
        except TypeError:
            # Fallback für ältere Agents, die (percepts, grid_size) erwarten
            action = agent.decide_move(percepts, self.world.grid_size)

        # Welt führt Aktion aus
        result = self.world.execute(agent, action)

        # Nächster Agent
        self.turn = (self.turn + 1) % len(self.agents)

        return result

    # ------------------------------
    # QMIX-Observation für A1
    # ------------------------------
    def _build_qmix_observation(self, agent, percepts):
        """
        Baut eine 5-d Observation wie im QMIX-Training:

          [x_norm, y_norm, breeze, stench, glitter]
        """

        x, y = agent.pos()
        grid_size = self.world.grid_size

        x_norm = x / (grid_size - 1) if grid_size > 1 else 0.0
        y_norm = y / (grid_size - 1) if grid_size > 1 else 0.0

        breeze = 1.0 if percepts.get("breeze", False) else 0.0
        stench = 1.0 if percepts.get("stench", False) else 0.0
        glitter = 1.0 if percepts.get("glitter", False) else 0.0

        return np.array([x_norm, y_norm, breeze, stench, glitter], dtype=np.float32)

    # ------------------------------
    # Alte Patch-Observation für Random-Agents
    # ------------------------------
    def _build_observation(self, agent, patch_size=5):
        """
        Returns a tensor-like numpy array with channels [breeze, stench, glitter]
        plus Agent-Position im Patch, wie vorher.
        """
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

        # Agent-Position im Patch
        pos_layer = [[0 for _ in range(patch_size)] for __ in range(patch_size)]
        pos_layer[half][half] = 1
        obs.append(pos_layer)

        return np.array(obs, dtype=np.float32)
