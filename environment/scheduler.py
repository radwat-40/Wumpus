"""Scheduler Klasse für die Verwaltung der Agenten. Bestimmt werlcher Agent an der Reihee ist,
 was er wahrnimmt und welche Atkion er ausführt."""

class Scheduler:
    def __init__(self, agents, world):
        self.agents = agents
        self.world = world
        self.turn = 0

    def step(self):
        # Falls alle Agenten tot sind, signalisiere das
        if not any(getattr(a, 'agent_alive', True) for a in self.agents):
            return "ALL_DEAD"

        # Finde den nächsten lebenden Agenten
        start = self.turn
        while not getattr(self.agents[self.turn], 'agent_alive', True):
            self.turn = (self.turn + 1) % len(self.agents)
            if self.turn == start:
                return "ALL_DEAD"

        agent = self.agents[self.turn]
        percepts = self.world.get_percepts(agent)
        action = agent.decide_move(percepts, self.world.grid_size)
        result = self.world.execute(agent, action)

        # Q-Learning Update (falls Agent es unterstützt)
        if hasattr(agent, 'learn') and hasattr(agent, 'reward_from_result'):
            reward = agent.reward_from_result(result, percepts)
            next_state = agent.get_state_id()
            agent.learn(reward, next_state)

        # Nächster Agent ist dran
        self.turn = (self.turn + 1) % len(self.agents)

        return result