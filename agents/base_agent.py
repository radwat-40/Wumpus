class Agent:
    def __init__(self, x, y, role):
        self.x = x
        self.y = y
        self.role = role
        self.direction = ['N','E','S','W']
        self.visited = set()
        self.memory = {}
        self.agent_alive = True
        self.agent_won = False
        self.color = (0, 0, 0)

    def pos(self):
        return (self.x, self.y)
    
