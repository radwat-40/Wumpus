"""Scheduler Klasse für die Verwaltung der Agenten. Bestimmt werlcher Agent an der Reihee ist,
 was er wahrnimmt und welche Atkion er ausführt."""

class Scheduler:
    def __init__(self, agents, world):
        self.agents = agents
        self.world = world
        self.turn = 0

    def step(self):
        # Falls alle Agenten tot sind
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


        #   1. Erzeuge lokale Beobachtung (Patch um Agent)
        percepts = self.world.get_percepts(agent)
        observation = self._build_observation(agent, patch_size=5)

        #   2. Agent entscheidet Aktion  

        try:
            action = agent.decide_move(observation, self.world.grid_size)
        except TypeError:
            # Fallback für ältere Agents: akzeptieren nur percepts
            action = agent.decide_move(percepts, self.world.grid_size)

        #   3. Welt führt aus  
        result = self.world.execute(agent, action)

        #   4. Neue Percepts / Observation nach der Aktion  
        next_observation = self._build_observation(agent, patch_size=5)

        #   5. RL-Update  
        if hasattr(agent, 'store_transition') and agent.role == "A1":
            done = result in ("WIN", "DIED")
            reward = agent.reward_from_result(result, percepts)

            agent.store_transition(observation, action, reward, next_observation, done)
            agent.learn_step()

        #   6. Nächster Agent  
        self.turn = (self.turn + 1) % len(self.agents)

        return result

    def _build_observation(self, agent, patch_size=5):
        # Returns a tensor-like numpy array with channels [breeze, stench, glitter]
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

        # agent position channel
        pos_layer = [[0 for _ in range(patch_size)] for __ in range(patch_size)]
        pos_layer[half][half] = 1
        obs.append(pos_layer)

        # Convert to numpy array: shape (channels, H, W)
        import numpy as _np
        return _np.array(obs, dtype=_np.float32)
