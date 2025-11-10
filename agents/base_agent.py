class Agent:
    def __init__(self, x, y, role):
        self.x = x
        self.y = y
        self.role = role
        self.direction = ['N','E','S','W']
        self.visited = set()
        self.memory = {}

    def pos(self):
        return (self.x, self.y)