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

        #   1. Percepts vor Aktion  
        percepts = self.world.get_percepts(agent)

        #   2. Agent entscheidet Aktion  
        action = agent.decide_move(percepts, self.world.grid_size)

        #   3. Welt führt aus  
        result = self.world.execute(agent, action)

        #   4. Neue Percepts nach der Aktion  
        next_percepts = self.world.get_percepts(agent)

        #   5. RL-Update  
        if hasattr(agent, 'learn') and hasattr(agent, 'reward_from_result'):
            reward = agent.reward_from_result(result, percepts)

            # Knowledge updaten (Agent hat Zustand gewechselt)
            agent.update_knowledge(next_percepts)

            # Lernen (next percepts mitgeben, damit Agent next_state berechnen kann)
            agent.learn(reward, next_percepts)

        #   6. Nächster Agent  
        self.turn = (self.turn + 1) % len(self.agents)

        return result
