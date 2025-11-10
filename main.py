# test 3 yahia

import pygame
import sys
import random

# Game config
GRID_SIZE = 20
TILE_SIZE = 32
WINDOW_SIZE = GRID_SIZE * TILE_SIZE

# Game element counts
NUM_PITS = 20
NUM_WUMPUS = 3
NUM_GOLD = 1

# Colors
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLACK = (0, 0, 0)
UNVISITED_COLOR = (230, 230, 230)
AGENT_COLOR = (0, 100, 255)
WUMPUS_COLOR = (150, 0, 0)
GOLD_COLOR = (255, 215, 0)
PIT_COLOR = (50, 50, 50)
BREEZE_COLOR = (173, 216, 230)
STENCH_COLOR = (255, 182, 193)

# Initialize pygame
pygame.init()
LEGEND_WIDTH = 250
screen = pygame.display.set_mode((WINDOW_SIZE + LEGEND_WIDTH, WINDOW_SIZE))
pygame.display.set_caption("Wumpus World 20x20")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# Agent class
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

# Create the agent
agent = Agent(0, 0, "player")

# World data
visited = set()
pits = set()
wumpus = set()
gold = set()
breeze_tiles = set()
stench_tiles = set()
game_over = False
win = False


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


def draw_grid():
    for x in range(0, WINDOW_SIZE, TILE_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, WINDOW_SIZE))
    for y in range(0, WINDOW_SIZE, TILE_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (WINDOW_SIZE, y))


def draw_world():
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            cell = (x, y)

            if cell in visited:
                if cell in breeze_tiles:
                    pygame.draw.rect(screen, BREEZE_COLOR, rect)
                elif cell in stench_tiles:
                    pygame.draw.rect(screen, STENCH_COLOR, rect)
                else:
                    pygame.draw.rect(screen, WHITE, rect)

                if cell in pits:
                    pygame.draw.rect(screen, PIT_COLOR, rect)
                if cell in wumpus:
                    pygame.draw.rect(screen, WUMPUS_COLOR, rect)
                if cell in gold:
                    pygame.draw.rect(screen, GOLD_COLOR, rect)

            else:
                pygame.draw.rect(screen, UNVISITED_COLOR, rect)

            pygame.draw.rect(screen, GRAY, rect, 1)


def draw_agent():
    x, y = agent.pos()
    pygame.draw.rect(screen, AGENT_COLOR, (x*TILE_SIZE+4, y*TILE_SIZE+4, TILE_SIZE-8, TILE_SIZE-8))
    label = font.render("A", True, BLACK)
    screen.blit(label, (x*TILE_SIZE + TILE_SIZE//4, y*TILE_SIZE + TILE_SIZE//4))


def draw_legend():
    items = [
        ("Agent", AGENT_COLOR, "A"),
        ("Gold", GOLD_COLOR, ""),
        ("Pit", PIT_COLOR, ""),
        ("Wumpus", WUMPUS_COLOR, ""),
        ("Breeze", BREEZE_COLOR, ""),
        ("Stench", STENCH_COLOR, ""),
        ("Visited", WHITE, ""),
        ("Unvisited", UNVISITED_COLOR, ""),
    ]
    x = WINDOW_SIZE + 20
    y = 20

    for text, color, symbol in items:
        pygame.draw.rect(screen, color, (x, y, 30, 30))
        pygame.draw.rect(screen, GRAY, (x, y, 30, 30), 1)
        if symbol:
            screen.blit(font.render(symbol, True, BLACK), (x+8, y+4))
        screen.blit(font.render(text, True, BLACK), (x+40, y+5))
        y += 40


def show_message(text, color):
    screen.blit(font.render(text, True, color), (10, 10))


def game_loop():
    global game_over, win

    # Mark start cell as visited
    visited.add(agent.pos())

    running = True
    while running:
        screen.fill(WHITE)
        draw_world()
        draw_grid()
        draw_agent()
        draw_legend()

        if game_over:
            show_message("Game Over! Press R to Restart", (255, 0, 0))
        elif win:
            show_message("You found the gold! Press R to Restart", (0, 180, 0))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_game()

                if game_over or win:
                    continue

                x, y = agent.pos()
                if event.key == pygame.K_UP and y > 0: y -= 1
                elif event.key == pygame.K_DOWN and y < GRID_SIZE-1: y += 1
                elif event.key == pygame.K_LEFT and x > 0: x -= 1
                elif event.key == pygame.K_RIGHT and x < GRID_SIZE-1: x += 1

                agent.x, agent.y = x, y
                visited.add(agent.pos())

                if agent.pos() in pits or agent.pos() in wumpus:
                    game_over = True
                elif agent.pos() in gold:
                    win = True

        clock.tick(10)

    pygame.quit()
    sys.exit()


def reset_game():
    global agent, visited, pits, wumpus, gold, breeze_tiles, stench_tiles, game_over, win
    agent = Agent(0, 0, "player")
    visited.clear()
    pits.clear()
    wumpus.clear()
    gold.clear()
    breeze_tiles.clear()
    stench_tiles.clear()
    game_over = False
    win = False
    place_random_items()
    visited.add(agent.pos())


if __name__ == "__main__":
    place_random_items()
    game_loop()
