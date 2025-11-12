"""Scheduler Klasse für die Verwaltung der Agenten. Bestimmt werlcher Agent an der Reihee ist,
 was er wahrnimmt und welche Atkion er ausführt."""

class Scheduler:
    def __init__(self, agents, world):
        self.agents = agents
        self.world = world
        self.turn = 0

    def step(self):
            #Ein schritt im Scheduler
        agent = self.agents[self.turn]

        percepts = self.world.get_percepts(agent)   # z.B. {"breeze": True, "stench": False, "glitter": True}

            #Agent entscheidet sich für eine Aktion
        action = agent.decide_move(percepts, self.world.grid_size)

            #Welt führt die Aktion aus
        result = self.world.execute(agent, action)

            #Nächster Agent ist dran
        self.turn = (self.turn + 1) % len(self.agents)

        return result