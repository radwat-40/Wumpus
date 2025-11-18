import pygame



# Farben
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

def draw_grid(screen, window_size, tile_size):
    for x in range(0, window_size, tile_size):
        pygame.draw.line(screen, GRAY, (x, 0), (x, window_size))
    for y in range(0, window_size, tile_size):
        pygame.draw.line(screen, GRAY, (0, y), (window_size, y))

def draw_world(screen, world, visited, grid_size, tile_size, font):
    for y in range(grid_size):
        for x in range(grid_size):
            rect = pygame.Rect(x*tile_size, y*tile_size, tile_size, tile_size)
            cell = (x, y)
            if cell in visited:
                if cell in world.breeze_tiles:
                    pygame.draw.rect(screen, BREEZE_COLOR, rect)
                elif cell in world.stench_tiles:
                    pygame.draw.rect(screen, STENCH_COLOR, rect)
                else:
                    pygame.draw.rect(screen, WHITE, rect)

                if cell in world.pits:
                    pygame.draw.rect(screen, PIT_COLOR, rect)
                if cell in world.wumpus:
                    pygame.draw.rect(screen, WUMPUS_COLOR, rect)
                if cell in world.gold:
                    pygame.draw.rect(screen, GOLD_COLOR, rect)
            else:
                pygame.draw.rect(screen, UNVISITED_COLOR, rect)
            pygame.draw.rect(screen, GRAY, rect, 1)

def draw_agent(screen, agent, tile_size, font):
    x, y = agent.pos()

    # Farbe abhängig von Rolle (A1, A2, A3, etc.)
    color = AGENT_COLOR
    role = getattr(agent, 'role', None)
    if isinstance(role, str):
        if role.endswith('2'):
            color = (0, 180, 0)
        elif role.endswith('3'):
            color = (200, 100, 0)
        elif role.endswith('1'):
                color = AGENT_COLOR

    alive = getattr(agent, 'agent_alive', True)
    draw_color = color if alive else (150, 150, 150)

    pygame.draw.rect(screen, draw_color, (x*tile_size+4, y*tile_size+4, tile_size-8, tile_size-8))

    # Label mit der Rollenbezeichnung (A1, A2, A3)
    label_text = role if role is not None else 'A'
    label = font.render(label_text, True, BLACK)
    screen.blit(label, (x*tile_size + tile_size//4, y*tile_size + tile_size//4))

    # Wenn tot, zeichne ein X über dem Agentenfeld
    if not alive:
        rect_x = x*tile_size+4
        rect_y = y*tile_size+4
        rect_size = tile_size-8
        pygame.draw.line(screen, BLACK, (rect_x, rect_y), (rect_x+rect_size, rect_y+rect_size), 2)
        pygame.draw.line(screen, BLACK, (rect_x+rect_size, rect_y), (rect_x, rect_y+rect_size), 2)

def draw_legend(screen, font, window_size, legend_width):
    items = [
        ("Gold", GOLD_COLOR, ""),
        ("Pit", PIT_COLOR, ""),
        ("Wumpus", WUMPUS_COLOR, ""),
        ("Breeze", BREEZE_COLOR, ""),
        ("Stench", STENCH_COLOR, ""),
        ("Visited", WHITE, ""),
        ("Unvisited", UNVISITED_COLOR, ""),
    ]
    x = window_size + 20
    y = 20
    # Zeichne spezielle Agenten-Zeile ohne farbigen Hintergrund
    screen.blit(font.render("  A     Agent", True, BLACK), (x, y+5))
    y += 40

    for text, color, symbol in items:
        pygame.draw.rect(screen, color, (x, y, 30, 30))
        pygame.draw.rect(screen, GRAY, (x, y, 30, 30), 1)
        if symbol:
            screen.blit(font.render(symbol, True, BLACK), (x+8, y+4))
        screen.blit(font.render(text, True, BLACK), (x+40, y+5))
        y += 40

def show_message(screen, font, text, color):
    screen.blit(font.render(text, True, color), (10, 10))
