from agents.base_agent import Agent
from environment.actions import Action
import random

class SimpleAgent(Agent):
    def __init__(self, x, y, role):
        super().__init__(x, y, role)
        self.visited = set()

    def decide_move(self, percep, grid_size):
        x, y = self.pos()
        self.visited.add((x, y))

        # Mögliche Bewegungen
        moves = {
            "up":    (x, y - 1),
            "down":  (x, y + 1),
            "left":  (x - 1, y),
            "right": (x + 1, y)
        }

        # Filtriere gültige Bewegungen
        valid_moves = {m: p for m, p in moves.items() if 0 <= p[0] < grid_size and 0 <= p[1] < grid_size}

        # Priorität: unbesuchte Zellen bevorzugen
        unvisited_moves = [m for m, p in valid_moves.items() if p not in self.visited]
        if unvisited_moves:
            move = random.choice(unvisited_moves)
        else:
            move = random.choice(list(valid_moves.keys()))

        # Konvertiere zu Action Enum
        if move == "up":
            return Action.MOVE_UP
        elif move == "down":
            return Action.MOVE_DOWN
        elif move == "left":
            return Action.MOVE_LEFT
        elif move == "right":
            return Action.MOVE_RIGHT