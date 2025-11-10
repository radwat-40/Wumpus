import random

# Welt-Funktionen und Daten
GRID_SIZE = 20
NUM_PITS = 20
NUM_WUMPUS = 3
NUM_GOLD = 1

pits = set()
wumpus = set()
gold = set()
breeze_tiles = set()
stench_tiles = set()

def in_bounds(x, y):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def get_neighbors(x, y):
    return [(nx, ny) for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if in_bounds(nx, ny)]

def place_random_items():
    global pits, wumpus, gold, breeze_tiles, stench_tiles
    pits.clear()
    wumpus.clear()
    gold.clear()
    breeze_tiles.clear()
    stench_tiles.clear()

    forbidden = {(0, 0)}
    all_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE) if (x, y) not in forbidden]

    pits.update(random.sample(all_cells, NUM_PITS))
    available = [c for c in all_cells if c not in pits]
    wumpus.update(random.sample(available, NUM_WUMPUS))
    available = [c for c in available if c not in wumpus]
    gold.update(random.sample(available, NUM_GOLD))

    for px, py in pits:
        for n in get_neighbors(px, py):
            breeze_tiles.add(n)

    for wx, wy in wumpus:
        for n in get_neighbors(wx, wy):
            stench_tiles.add(n)
