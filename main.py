import pygame
import sys

from environment.world import World
from environment import drawing
from environment.scheduler import Scheduler

# Agents
from agents.agent1 import MarcAgent          # PPO-Agent
from agents.agent2 import YahiaAgent         
from agents.agent3 import HenrikAgent        


 
# Welt erstellen
 
world = World()

TILE_SIZE = 32
WINDOW_SIZE = world.grid_size * TILE_SIZE
LEGEND_WIDTH = 250

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE + LEGEND_WIDTH, WINDOW_SIZE))
pygame.display.set_caption("Wumpus World 20x20")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)


 
# Agenten erstellen
 
agent1 = MarcAgent(0, 0, "A1", grid_size=world.grid_size, model_path="ppo_model.pth")
agent2 = YahiaAgent(1, 0, "A2")
agent3 = HenrikAgent(2, 0, "A3")

agents = [agent1, agent2, agent3]

scheduler = Scheduler(agents, world)
scheduler.reset_memory(world.grid_size)

visited = set()
game_over = False
win = False


 
# Reset Funktion
 
def reset_game():
    global agents, scheduler, visited, game_over, win

    agent1 = MarcAgent(0, 0, "A1", grid_size=world.grid_size, model_path="ppo_model.pth")
    agent2 = YahiaAgent(1, 0, "A2")
    agent3 = HenrikAgent(2, 0, "A3")

    agents = [agent1, agent2, agent3]

    scheduler = Scheduler(agents, world)
    scheduler.reset_memory(world.grid_size)

    visited.clear()
    world.reset()

    game_over = False
    win = False

    for agent in agents:
        visited.add(agent.pos())


 
# Haupt-Game-Loop
 
def game_loop():
    global game_over, win

    for agent in agents:
        visited.add(agent.pos())

    world.reset()
    running = True

    while running:
        screen.fill((255, 255, 255))

        drawing.draw_world(screen, world, visited, world.grid_size, TILE_SIZE, font)
        drawing.draw_grid(screen, WINDOW_SIZE, TILE_SIZE)

        for agent in agents:
            drawing.draw_agent(screen, agent, TILE_SIZE, font)

        drawing.draw_legend(screen, font, WINDOW_SIZE, LEGEND_WIDTH)

        if game_over:
            drawing.show_message(screen, font, "Game Over! Press R to Restart", (255, 0, 0))
        elif win:
            drawing.show_message(screen, font, "You found the gold! Press R to Restart", (0, 180, 0))

        pygame.display.flip()

        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_game()

        if not (game_over or win):
            result = scheduler.step()

            current_agent = agents[scheduler.turn - 1]
            visited.add(current_agent.pos())

            if result == "ALL_DEAD":
                game_over = True
            elif result == "WIN":
                win = True

        clock.tick(10)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    game_loop()
