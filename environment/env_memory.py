import numpy as np
from environment.actions import Action
from environment.world import World


class WumpusMemoryEnv:
    """
    Single-Agent Wumpus-Env mit Memory-Grid:
    Kanäle:
      0: unknown = 1 / known = 0
      1: safe    = 1
      2: risky   = 1 (breeze/stench)
      3: agent position = 1
    """

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
        self.known[:] = 0.0
        self.safe[:] = 0.0
        self.risky[:] = 0.0

        # Startposition setzen
        self.agent.x, self.agent.y = 0, 0
        self.update_memory()

        return self.get_state()

    def update_memory(self):
        x, y = self.agent.x, self.agent.y
        self.known[y, x] = 1.0

        percepts = self.world.get_percepts(self.agent)

        if percepts["breeze"] or percepts["stench"]:
            self.risky[y, x] = 1.0
        else:
            self.safe[y, x] = 1.0

    def get_state(self):
        ax, ay = self.agent.x, self.agent.y
        agent_map = np.zeros_like(self.safe, dtype=np.float32)
        agent_map[ay, ax] = 1.0

        # Channel 0: unknown = 1 - known
        unknown = 1.0 - self.known

        return np.stack([unknown, self.safe, self.risky, agent_map], axis=0)

    def step(self, action):
        self.steps += 1

        # kopiere alten Memory-Zustand für Reward-Berechnung
        old_known = self.known.copy()
        old_safe = self.safe.copy()
        old_risky = self.risky.copy()

        result = self.world.execute(self.agent, action)

        # Memory nach Aktion aktualisieren
        self.update_memory()

        reward = self.compute_reward(action, result, old_known, old_safe, old_risky)
        done = result in ("WIN", "DIED") or self.steps >= self.max_steps

        return self.get_state(), reward, done, {"result": result}

    def compute_reward(self, action, result, old_known, old_safe, old_risky):
        """
        Stabileres Reward-Shaping für PPO / Q-Learning.

        Terminal:
          +200  bei WIN
          -100  bei DIED

        Shaping:
          +5   pro neu bekanntem Feld
          +10  pro neu als 'safe' markiertem Feld
          -2   wenn aktuelles Feld 'risky' ist
          -0.01 pro Schritt (leichte Zeitstrafe)

        Alles geclamped, um Ausreißer zu vermeiden.
        """
        # Terminal-Rewards
        if result == "WIN":
            return 200.0
        if result == "DIED":
            return -100.0

        r = 0.0

        # neu bekannte Felder
        new_known_mask = (self.known > 0.5) & (old_known <= 0.5)
        newly_known = np.count_nonzero(new_known_mask)
        r += 5.0 * float(newly_known)

        # neu als sicher markierte Felder
        new_safe_mask = (self.safe > 0.5) & (old_safe <= 0.5)
        newly_safe = np.count_nonzero(new_safe_mask)
        r += 10.0 * float(newly_safe)

        # Risiko-Penalty am aktuellen Feld
        x, y = self.agent.x, self.agent.y
        if self.risky[y, x] > 0.5:
            r -= 2.0

        # leichte Schritt-Strafe, damit er nicht idlet
        r -= 0.01

        # Sicherheit: Reward clampen
        if r > 200.0:
            r = 200.0
        if r < -200.0:
            r = -200.0

        return float(r)
