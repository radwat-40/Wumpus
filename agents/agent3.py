from agents.base_agent import Agent
from environment.actions import Action
import random


class HenrikAgent(Agent):
    def __init__(self, x, y, role):
        super().__init__(x, y, role)
        self.visited = set()
        self.visited_dangerous = set()
        self.visited_safe = set()
        self.previous_safe = None

    def decide_move(self, percep, grid_size):
        x, y = self.pos()
        self.visited.add((x, y))

        if isinstance(percep, dict):
            breeze = percep.get("breeze", False)
            stench = percep.get("stench", False)
        else:
            breeze = getattr(percep, "breeze", False)
            stench = getattr(percep, "stench", False)

        is_unsafe = bool(breeze or stench)

        if is_unsafe:
            self.visited_dangerous.add((x, y))

            if self.previous_safe is not None:
                px, py = self.previous_safe
                dx = px - x
                dy = py - y

                if abs(dx) + abs(dy) == 1:
                    if dx == 1:
                        return Action.MOVE_RIGHT
                    if dx == -1:
                        return Action.MOVE_LEFT
                    if dy == 1:
                        return Action.MOVE_DOWN
                    if dy == -1:
                        return Action.MOVE_UP

            valid_moves = self.get_valid_moves(x, y, grid_size)
            if not valid_moves:
                # absolute Notlösung (sollte quasi nie passieren)
                return Action.MOVE_RIGHT
            return random.choice(valid_moves)[0]

        # safe
        self.visited_safe.add((x, y))
        self.previous_safe = (x, y)

        valid_moves = self.get_valid_moves(x, y, grid_size)
        if not valid_moves:
            return Action.MOVE_RIGHT  # Notlösung

        scored_moves = []
        for action, (nx, ny) in valid_moves:
            if (nx, ny) in self.visited_dangerous:
                score = 10
            elif (nx, ny) not in self.visited:
                score = 150
            elif (nx, ny) in self.visited_safe:
                score = 100
            else:
                score = 50
            scored_moves.append((score, action))

        max_score = max(score for score, _ in scored_moves)
        best_actions = [a for score, a in scored_moves if score == max_score]
        return random.choice(best_actions)

    def get_valid_moves(self, x, y, grid_size):
        moves = []
        directions = [
            (Action.MOVE_UP,    (x, y - 1)),
            (Action.MOVE_DOWN,  (x, y + 1)),
            (Action.MOVE_RIGHT, (x + 1, y)),
            (Action.MOVE_LEFT,  (x - 1, y)),
        ]
        for action, (nx, ny) in directions:
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                moves.append((action, (nx, ny)))
        return moves
