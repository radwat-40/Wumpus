class Scheduler:
    def __init__(self, agents, world):
        self.agents = agents
        self.world = world
        self.turn = 0

    def step(self):
            #Ein schritt im Scheduler
        agent = self.agents[self.turn]

        percepts = {
            "breeze": agent.pos() in self.world.breeze_tiles,
            "stench": agent.pos() in self.world.stench_tiles,
            "glitter": agent.pos() in self.world.gold
            }

            #Agent entscheidet sich für eine Aktion
        action = agent.decide_move(percepts, self.world.grid_size)

            #Welt führt die Aktion aus
        result = self.world.execute(agent, action)

            #Nächster Agent ist dran
        self.turn = (self.turn + 1) % len(self.agents)

        return result