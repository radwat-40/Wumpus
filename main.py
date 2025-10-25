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
LEGEND_WIDTH = 250  # space for legend
screen = pygame.display.set_mode((WINDOW_SIZE + LEGEND_WIDTH, WINDOW_SIZE))
pygame.display.set_caption("Wumpus World 20x20")
clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 24)

# Grid setup
agent_pos = [0, 0]
visited = set()
pits = set()
wumpus = set()
gold = set()
breeze_tiles = set()
stench_tiles = set()
game_over = False
win = False

# Helper functions
def in_bounds(x, y):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def get_neighbors(x, y):
    return [(nx, ny) for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if in_bounds(nx, ny)]

def place_random_items():
    global pits, wumpus, gold, breeze_tiles, stench_tiles

    forbidden = {(0, 0)}
    all_cells = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE) if (x, y) not in forbidden]

    pits = set(random.sample(all_cells, NUM_PITS))
    available = [cell for cell in all_cells if cell not in pits]
    wumpus = set(random.sample(available, NUM_WUMPUS))
    available = [cell for cell in available if cell not in wumpus]
    gold = set(random.sample(available, NUM_GOLD))

    # Calculate percepts
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
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            cell = (x, y)

            if cell in visited:
                # Show percepts if visited
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

            # Grid border
            pygame.draw.rect(screen, GRAY, rect, 1)

def draw_agent():
    x, y = agent_pos
    label = font.render("A", True, BLACK)
    screen.blit(label, (x * TILE_SIZE + TILE_SIZE // 4, y * TILE_SIZE + TILE_SIZE // 4))

def draw_legend():
    legend_items = [
        ("Agent", WHITE, "A"),
        ("Gold", GOLD_COLOR, ""),
        ("Pit", PIT_COLOR, ""),
        ("Wumpus", WUMPUS_COLOR, ""),
        ("Breeze", BREEZE_COLOR, ""),
        ("Stench", STENCH_COLOR, ""),
        ("Visited", WHITE, ""),
        ("Unvisited", UNVISITED_COLOR, ""),
    ]

    start_x = WINDOW_SIZE + 20
    start_y = 20
    spacing = 40

    for i, (label_text, color, symbol) in enumerate(legend_items):
        y = start_y + i * spacing

        pygame.draw.rect(screen, color, (start_x, y, 30, 30))
        pygame.draw.rect(screen, GRAY, (start_x, y, 30, 30), 1)  # border

        if symbol:
            label = font.render(symbol, True, BLACK)
            screen.blit(label, (start_x + 8, y + 4))

        label = font.render(label_text, True, BLACK)
        screen.blit(label, (start_x + 40, y + 5))

def show_message(text, color):
    label = font.render(text, True, color)
    screen.blit(label, (10, 10))

# Place game items
place_random_items()

# Main game loop
def game_loop():
    global agent_pos, game_over, win

    running = True
    while running:
        screen.fill(WHITE)
        draw_world()
        draw_grid()
        draw_agent()
        draw_legend()

        if game_over:
            show_message("Game Over! Press R to Restart.", (255, 0, 0))
        elif win:
            show_message("You found the gold! Press R to Restart.", (0, 180, 0))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_game()
                if game_over or win:
                    continue

                x, y = agent_pos
                if event.key == pygame.K_UP and y > 0:
                    y -= 1
                elif event.key == pygame.K_DOWN and y < GRID_SIZE - 1:
                    y += 1
                elif event.key == pygame.K_LEFT and x > 0:
                    x -= 1
                elif event.key == pygame.K_RIGHT and x < GRID_SIZE - 1:
                    x += 1

                agent_pos = [x, y]
                visited.add((x, y))

                # Check outcomes
                if (x, y) in pits or (x, y) in wumpus:
                    game_over = True
                elif (x, y) in gold:
                    win = True

        clock.tick(10)

    pygame.quit()
    sys.exit()

def reset_game():
    global agent_pos, visited, pits, wumpus, gold, breeze_tiles, stench_tiles, game_over, win
    agent_pos = [0, 0]
    visited = set()
    pits = set()
    wumpus = set()
    gold = set()
    breeze_tiles = set()
    stench_tiles = set()
    game_over = False
    win = False
    place_random_items()

if __name__ == "__main__":
    game_loop()
