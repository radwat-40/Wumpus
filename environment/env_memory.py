import numpy as np
from environment.actions import Action
from environment.world import World

class WumpusMemoryEnv:

    def __init__(self, agent, grid_size=20, num_pits=20, num_wumpus=3, num_gold=1, max_steps=500):
        self.agent = agent
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.world = World(grid_size, num_pits, num_wumpus, num_gold)

        # Memory-Grids
        self.known = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.safe = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.risky = np.zeros((grid_size, grid_size), dtype=np.float32)

        self.steps = 0

    def reset(self):
        self.world.reset()
        self.steps = 0

        # Memory-Maps löschen
        self.known[:] = 0
        self.safe[:] = 0
        self.risky[:] = 0

        # Startposition setzen
        self.agent.x, self.agent.y = 0, 0
        self.update_memory()

        return self.get_state()

    def update_memory(self):
        x, y = self.agent.x, self.agent.y
        self.known[y, x] = 1

        percepts = self.world.get_percepts(self.agent)

        if percepts["breeze"] or percepts["stench"]:
            self.risky[y, x] = 1
        else:
            self.safe[y, x] = 1

    def get_state(self):
        ax, ay = self.agent.x, self.agent.y
        agent_map = np.zeros_like(self.safe)
        agent_map[ay, ax] = 1

        return np.stack([1 - self.known, self.safe, self.risky, agent_map], axis=0)

    def step(self, action):
        self.steps += 1

        old_known = self.known.copy()
        old_safe = self.safe.copy()

        result = self.world.execute(self.agent, action)

        self.update_memory()

        reward = self.compute_reward(action, result, old_known, old_safe)
        done = result in ("WIN", "DIED") or self.steps >= self.max_steps

        return self.get_state(), reward, done, {"result": result}

    def compute_reward(self, action, result, old_known, old_safe):
        r = 0

        if result == "WIN": return 100
        if result == "DIED": return -100

        # neues Feld?
        newly_known = np.sum(self.known - old_known)
        r += newly_known * 1.0

        # neue Safe-Area?
        newly_safe = np.sum(self.safe - old_safe)
        r += newly_safe * 2.0

        # risk penalty
        x, y = self.agent.x, self.agent.y
        if self.risky[y, x] == 1:
            r -= 1.0

        # time penalty
        r -= 0.05

        return r
