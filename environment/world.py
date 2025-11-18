import random
from environment.actions import Action

#World Klasse für die Verwaltung der Welt, Platzierung von Gegenständen und Ausführung von Aktionen.
class World:
    def __init__(self, grid_size=20, num_pits=20, num_wumpus = 3, num_gold =1):
        self.grid_size = grid_size
        self.num_pits = num_pits
        self.num_wumpus = num_wumpus
        self.num_gold = num_gold
        self.pits = set()
        self.wumpus = set()
        self.gold = set()
        self.breeze_tiles = set()
        self.stench_tiles = set()

        self.place_random_items()


    #ist in Welt?
    def in_bounds(self, x, y):
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    #gibt nachbarblöcke zurück
    def get_neighbors(self, x, y):
        return [
            (nx, ny)
            for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
            if self.in_bounds(nx, ny)]
    
    #platziert zufällige gegenstände und felder
    def place_random_items(self):
        self.pits.clear()
        self.wumpus.clear()
        self.gold.clear()
        self.breeze_tiles.clear()
        self.stench_tiles.clear()

        forbidden = {(0, 0),(1, 0), (2, 0)} 
    
        all_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if (x, y) not in forbidden]

        self.pits.update(random.sample(all_cells, self.num_pits))
        available = [c for c in all_cells if c not in self.pits]
        self.wumpus.update(random.sample(available, self.num_wumpus))
        available = [c for c in available if c not in self.wumpus]
        self.gold.update(random.sample(available, self.num_gold))
        for px, py in self.pits:
            for n in self.get_neighbors(px, py):
                self.breeze_tiles.add(n)

        for wx, wy in self.wumpus:
            for n in self.get_neighbors(wx, wy):
                self.stench_tiles.add(n)

    #Setzt die Welt vollständig zurück und generiert alle Elemente neu
    def reset(self):
        self.pits.clear()
        self.wumpus.clear()
        self.gold.clear()
        self.breeze_tiles.clear()
        self.stench_tiles.clear()
        self.place_random_items()

    #behandelt Schussaktionen
    def handle_shoot(self, agent, action):
        x, y = agent.pos()
         
        if action == Action.SHOOT_UP:
            target_pos = (x, y - 1)
        elif action == Action.SHOOT_DOWN:
                target_pos = (x, y + 1)
        elif action == Action.SHOOT_LEFT:
                target_pos = (x - 1, y)
        elif action == Action.SHOOT_RIGHT:
                target_pos = (x + 1, y)
        else:
             return False
        
        if target_pos in self.wumpus:
            self.wumpus.remove(target_pos)
            self.update_stench_tiles()
            return True
        
        return False

    #führt die Aktion des Agenten aus und gibt das Ergebnis zurück
    def execute(self, agent, action):
        x, y = agent.pos()

        if action == Action.SHOOT_UP:
            target_pos = (x, y - 1)
            if target_pos in self.wumpus:
                self.wumpus.remove(target_pos)
                self.update_stench_tiles()
            return "CONTINUE"
        
        if action == Action.SHOOT_DOWN:
            target_pos = (x, y + 1)
            if target_pos in self.wumpus:
                self.wumpus.remove(target_pos)
                self.update_stench_tiles()
            return "CONTINUE"
        if action == Action.SHOOT_LEFT:
            target_pos = (x - 1, y)
            if target_pos in self.wumpus:
                self.wumpus.remove(target_pos)
                self.update_stench_tiles()
            return "CONTINUE"
        if action == Action.SHOOT_RIGHT:
            target_pos = (x + 1, y)
            if target_pos in self.wumpus:
                self.wumpus.remove(target_pos)
                self.update_stench_tiles()
            return "CONTINUE"
        
        if action == Action.GRAB:
            if agent.pos() in self.gold:
                self.gold.remove(agent.pos())
                return "WIN"
            return "CONTINUE"

        if action == Action.MOVE_UP and y > 0:
            agent.y -= 1
        elif action == Action.MOVE_DOWN and y < self.grid_size - 1:
            agent.y += 1
        elif action == Action.MOVE_LEFT and x > 0:
            agent.x -= 1
        elif action == Action.MOVE_RIGHT and x < self.grid_size - 1:
            agent.x += 1

        if agent.pos() in self.pits or agent.pos() in self.wumpus:
            # markiere Agent als tot, aber beende nicht das gesamte Spiel
            try:
                agent.agent_alive = False
            except Exception:
                # falls Agent kein flag hat, setze ein Attribut
                setattr(agent, 'agent_alive', False)
            return "DIED"

        elif agent.pos() in self.gold:
            return "WIN"
        return "CONTINUE"
    
    #gibt die Wahrnehmungen des Agenten basierend auf seiner Position zurück
    def get_percepts(self, agent):
            x, y = agent.pos()
            return {
                "breeze": (x, y) in self.breeze_tiles,
                "stench": (x, y) in self.stench_tiles,
                "glitter": (x, y) in self.gold
            }
    
    def update_stench_tiles(self):
        self.stench_tiles.clear()
        for wx, wy in self.wumpus:
            for n in self.get_neighbors(wx, wy):
                self.stench_tiles.add(n)