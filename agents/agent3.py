from agents.base_agent import Agent
from environment.actions import Action
import random
import logging

logger = logging.getLogger("A3")


class HenrikAgent(Agent):
    def __init__(self, x, y, role):
        super().__init__(x, y, role)
        self.visited = set()
        self.visited_dangerous = set()
        self.visited_safe = set()
        self.previous_safe = None
        self.confirmed_pits = set()
        self.confirmed_wumpus = set()
        self.breeze_positions = set()
        self.stench_positions = set()
        self.suspects_pits = set()
        self.suspects_wumpus = set()
        self.visit_count = {}

    def set_breeze_at(self, pos):
        if pos:
            self.breeze_positions.add(pos)

    def set_stench_at(self, pos):
        if pos:
            self.stench_positions.add(pos)

    def _mark_suspects_from_breeze(self, pos):
        if pos:
            x, y = pos
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in self.confirmed_pits and (nx, ny) not in self.confirmed_wumpus:
                    self.suspects_pits.add((nx, ny))

    def _mark_suspects_from_stench(self, pos):
        if pos:
            x, y = pos
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in self.confirmed_pits and (nx, ny) not in self.confirmed_wumpus:
                    self.suspects_wumpus.add((nx, ny))

    def find_next_unexplored_adjacent_to_safe(self, grid_size):
        """Finde das nächste unentdeckte Feld, das neben einem sicheren entdeckten Feld liegt."""
        current_x, current_y = self.pos()
        candidates = []
        
        for nx in range(grid_size):
            for ny in range(grid_size):
                cell = (nx, ny)
                if cell in self.visited:
                    continue  # Schon entdeckt
                
                # Prüfe ob neben einem sicheren Feld
                adjacent_safe = False
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    adj_x, adj_y = nx + dx, ny + dy
                    if 0 <= adj_x < grid_size and 0 <= adj_y < grid_size:
                        adj_cell = (adj_x, adj_y)
                        if adj_cell in self.visited_safe or (adj_cell in self.visited and adj_cell not in self.visited_dangerous):
                            adjacent_safe = True
                            break
                
                if adjacent_safe:
                    # Berechne distanzur aktuellen Position
                    dist = abs(nx - current_x) + abs(ny - current_y)
                    candidates.append((dist, cell))
        
        if candidates:
            # Wähle das nächste kleinste Distanz
            candidates.sort()
            return candidates[0][1]  # Die Position
        return None

    def decide_move(self, percep, grid_size):
        x, y = self.pos()
        self.visited.add((x, y))
        self.visit_count[(x, y)] = self.visit_count.get((x, y), 0) + 1

        # Finde nächste unentdeckte Feld 
        next_target = self.find_next_unexplored_adjacent_to_safe(grid_size)

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
                # wenn sonst nichts geht
                return Action.MOVE_RIGHT
            return random.choice(valid_moves)[0]

        # safe
        self.visited_safe.add((x, y))
        self.previous_safe = (x, y)

        valid_moves = self.get_valid_moves(x, y, grid_size)
        if not valid_moves:
            return Action.MOVE_RIGHT  # Notlösung

        # Filtere bestätigte Gefahren heraus
        safe_moves = [
            (action, (nx, ny)) for action, (nx, ny) in valid_moves
            if (nx, ny) not in self.confirmed_pits and (nx, ny) not in self.confirmed_wumpus
        ]
        if not safe_moves:
            #  Wenn keine sicheren Moves dann zufällig
            return random.choice(valid_moves)[0]

        scored_moves = []
        for action, (nx, ny) in safe_moves:
            score = 100  # Basis score
            if (nx, ny) in self.visited_dangerous:
                score = 10
            elif (nx, ny) not in self.visited:
                score = 150
            elif (nx, ny) in self.visited_safe:
                score = 120
            # minus score für Verdächtige
            if (nx, ny) in self.suspects_pits or (nx, ny) in self.suspects_wumpus:
                score -= 50  # Mache sie weniger attraktiv
            # - scorefür häufiges Besuchen
            visit_penalty = self.visit_count.get((nx, ny), 0) * 10
            score -= visit_penalty
            # + score nächstes unentdecktes Ziel
            if next_target and (nx, ny) == next_target:
                score += 50
            scored_moves.append((score, action))

        max_score = max(score for score, _ in scored_moves)
        best_actions = [a for score, a in scored_moves if score == max_score]
        
       
        best_targets = [(action, (nx, ny)) for action, (nx, ny) in safe_moves if action in best_actions]
        if all((nx, ny) in self.visited for _, (nx, ny) in best_targets):
            next_target = self.find_next_unexplored_adjacent_to_safe(grid_size)
            if next_target:
                target_x, target_y = next_target
                dx = target_x - x
                dy = target_y - y
                if abs(dx) + abs(dy) == 1: 
                    if dx == 1:
                        return Action.MOVE_RIGHT
                    elif dx == -1:
                        return Action.MOVE_LEFT
                    elif dy == 1:
                        return Action.MOVE_DOWN
                    elif dy == -1:
                        return Action.MOVE_UP
        
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



    def receive_messages(self, messages):
        if not messages:
            return
        for m in messages:
            topic = (getattr(m, "topic", "") or "").lower()
            payload = getattr(m, "payload", {}) or {}
            sender = getattr(m, "sender", None)

            pos = payload.get("pos")
            if pos is not None:
                try:
                    pos = tuple(pos)
                except Exception:
                    pos = None

            logger.info(f"[A3] receive_messages: from={sender} topic={topic} pos={pos}")

            
            if topic.startswith("breeze"):
                self.set_breeze_at(pos)
                self._mark_suspects_from_breeze(pos)
            elif topic.startswith("stench"):
                self.set_stench_at(pos)
                self._mark_suspects_from_stench(pos)


            elif topic == "AGENT_DIED" or topic == "agENT_DIED".lower():
              
                dead_pos = payload.get("pos") or payload.get("position")
                if dead_pos is not None:
                    try:
                        dead_pos = tuple(dead_pos)
                    except Exception:
                        dead_pos = None
                cause = payload.get("cause", "").lower() if payload else ""
                if dead_pos is not None:
                    if "pit" in cause:
                        self.confirmed_pits.add(dead_pos)
                    elif "wumpus" in cause:
                        self.confirmed_wumpus.add(dead_pos)
                    else:
                        
                        self.confirmed_pits.add(dead_pos)
                    logger.info(f"[A3] receive_messages: confirmed danger at {dead_pos} cause={cause}")
            else:
                logger.debug(f"[A3] receive_messages: unhandled topic={topic}")
