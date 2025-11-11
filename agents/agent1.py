from agents.base_agent import Agent
import random

class SimpleAgent(Agent):
    def __init__(self, x, y, role):
        super().__init__(x, y, role)
        self.visited = set()

    def decide_move(self, percep, grid_size):
        x, y = self.pos()
        self.visited.add((x, y))

        # Vecinii posibili
        moves = {
            "up":    (x, y - 1),
            "down":  (x, y + 1),
            "left":  (x - 1, y),
            "right": (x + 1, y)
        }

        # Filtrăm mișcările valide
        moves = {m: p for m, p in moves.items() if 0 <= p[0] < grid_size and 0 <= p[1] < grid_size}

        # Prioritate: mergi într-o celulă nevizitată
        unvisited_moves = [m for m, p in moves.items() if p not in self.visited]
        if unvisited_moves:
            return random.choice(unvisited_moves)

        # Dacă toate sunt vizitate → mișcare aleatoare validă
        return random.choice(list(moves.keys()))
