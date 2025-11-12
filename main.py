import pygame
import sys
from agents.base_agent import Agent
from environment.world import World
from environment import drawing
from agents.agent1 import SimpleAgent as Sim

world = World()

# Game config

TILE_SIZE = 32
WINDOW_SIZE = world.grid_size * TILE_SIZE
LEGEND_WIDTH = 250

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE + LEGEND_WIDTH, WINDOW_SIZE))
pygame.display.set_caption("Wumpus World 20x20")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

agent = Sim(0, 0, "player")
visited = set()
game_over = False
win = False

def reset_game():
    global agent, visited, game_over, win
    agent = Sim(0, 0, "player")
    visited.clear()
    world.place_random_items()
    game_over = False
    win = False
    visited.add(agent.pos())

def game_loop():
    global game_over, win
    visited.add(agent.pos())
    world.place_random_items()
    running = True

    while running:
        screen.fill((255, 255, 255))
        drawing.draw_world(screen, visited, world.grid_size, TILE_SIZE, font)
        drawing.draw_grid(screen, WINDOW_SIZE, TILE_SIZE)
        drawing.draw_agent(screen, agent, TILE_SIZE, font)
        drawing.draw_legend(screen, font, WINDOW_SIZE, LEGEND_WIDTH)

        if game_over:
            drawing.show_message(screen, font, "Game Over! Press R to Restart", (255, 0, 0))
        elif win:
            drawing.show_message(screen, font, "You found the gold! Press R to Restart", (0, 180, 0))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_game()
            if not(game_over or win):
                    
                percepts = {
                    "stench": agent.pos() in world.wumpus,
                    "breeze": agent.pos() in world.pits,
                    "glitter": agent.pos() in world.gold
                }

                move = agent.decide_move(percepts, world.grid_size)

                x, y = agent.pos()
                if move == "up" and y > 0: y -= 1
                elif move == "down" and y < world.grid_size-1: y += 1
                elif move == "left" and x > 0: x -= 1
                elif move == "right" and x < world.grid_size-1: x += 1

                agent.x, agent.y = x, y
                visited.add(agent.pos())

                if agent.pos() in world.pits or agent.pos() in world.wumpus:
                    game_over = True
                elif agent.pos() in world.gold:
                    win = True

        clock.tick(10)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    game_loop()
