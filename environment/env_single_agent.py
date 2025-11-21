import numpy as np
from environment.world import World

class WumpusSingleAgentEnv:
    """
    Minimal Single-Agent-Environment für DQN-Training.
    Kein Scheduler, keine anderen Agenten.
    step() gibt (next_obs, reward, done, info) zurück.
    Observation = Feature-Vektor, kein CNN-Patch.
    """

    def __init__(
        self,
        agent,
        grid_size=20,
        num_pits=10,
        num_wumpus=2,
        num_gold=1,
        max_steps=300,
    ):
        self.agent = agent
        self.world = World(grid_size=grid_size,
                           num_pits=num_pits,
                           num_wumpus=num_wumpus,
                           num_gold=num_gold)
        self.max_steps = max_steps
        self.steps = 0

    def reset(self):
        """Welt und Agent für eine neue Episode zurücksetzen."""
        self.world.reset()
        # Startposition wie im Spiel
        self.agent.x, self.agent.y = 0, 0

        # Agent-internes Episoden-Reset
        if hasattr(self.agent, "reset_episode"):
            self.agent.reset_episode()

        # Besuchte Felder zurücksetzen
        if hasattr(self.agent, "visited"):
            self.agent.visited.clear()
            self.agent.visited.add((self.agent.x, self.agent.y))

        self.steps = 0
        return self._build_observation()

    def _build_observation(self):
        """
        Feature-Vektor:
        [ x_norm, y_norm, breeze, stench, glitter, visited, step_frac ]
        """
        x, y = self.agent.pos()
        g = self.world.grid_size

        percepts = self.world.get_percepts(self.agent)
        breeze = 1.0 if percepts.get("breeze", False) else 0.0
        stench = 1.0 if percepts.get("stench", False) else 0.0
        glitter = 1.0 if percepts.get("glitter", False) else 0.0

        visited_flag = 1.0 if (hasattr(self.agent, "visited") and (x, y) in self.agent.visited) else 0.0
        step_frac = float(self.steps) / float(self.max_steps) if self.max_steps > 0 else 0.0

        obs = np.array([
            x / (g - 1),
            y / (g - 1),
            breeze,
            stench,
            glitter,
            visited_flag,
            step_frac,
        ], dtype=np.float32)

        return obs

    def step(self, action):
        """
        Einen Schritt ausführen:
        - Action auf Agent anwenden
        - Reward über MarcAgent.reward_from_result berechnen
        - done, wenn WIN/DIED oder max_steps erreicht
        """
        old_pos = self.agent.pos()
        percepts = self.world.get_percepts(self.agent)

        result = self.world.execute(self.agent, action)

        self.steps += 1
        next_obs = self._build_observation()
        next_percepts = self.world.get_percepts(self.agent)
        new_pos = self.agent.pos()

        done = result in ("WIN", "DIED") or self.steps >= self.max_steps

        if hasattr(self.agent, "reward_from_result"):
            reward = self.agent.reward_from_result(
                result=result,
                percepts=percepts,
                next_percepts=next_percepts,
                old_pos=old_pos,
                new_pos=new_pos,
            )
        else:
            reward = 0.0

        info = {"result": result}
        return next_obs, reward, done, info
