import pygame
import sys
import logging
from pathlib import Path

from environment.world import World
from environment import drawing
from environment.scheduler import Scheduler

from agents.agent1 import MarcAgent
from agents.agent2 import YahiaAgent
from agents.agent3 import HenrikAgent

from logger.logger import MessageBus
    
# --- Logging ---
LOG_PATH = Path("wumpus.log")
logging.basicConfig(
    level=logging.DEBUG,  # Bei Bedarf auf INFO ändern
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
    ],
)

# --- Welt + UI Konfiguration ---
world = World()
TILE_SIZE = 32
WINDOW_SIZE = world.grid_size * TILE_SIZE
LEGEND_WIDTH = 250

pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE + LEGEND_WIDTH, WINDOW_SIZE))
pygame.display.set_caption(f"Wumpus World {world.grid_size}x{world.grid_size}")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# --- MessageBus erstellen (Nachrichten-Persistenz optional) ---
bus = MessageBus(persist_file="wumpus_messages.log")

# --- Pfad für A2 Q-Tables ---
QS_PATH = Path("data/agent2_qtables.pkl")
QS_PATH.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Agenten erzeugen: Registrierung auf dem Bus und A2 init_maps
# ------------------------------------------------------------
def create_agents():
    """
    Erstellt Agenten, registriert sie am MessageBus und
    initialisiert für A2 die Karten (breeze/stench).
    """
    agent1 = MarcAgent(0, 0, "A1")
    agent2 = YahiaAgent(1, 0, "A2")
    agent3 = HenrikAgent(2, 0, "A3")

    for a in (agent1, agent2, agent3):
        bus.register(a.role)
        a.bus = bus

    # A2 Karten auf Grid-Größe setzen (falls vorhanden)
    if hasattr(agent2, "init_maps"):
        agent2.init_maps(world.grid_size)

    return [agent1, agent2, agent3]

# initiale Agenten & Scheduler erzeugen
agents = create_agents()

# A2 Q-Tables nur laden, wenn die Datei existiert (nicht überschreiben)
for a in agents:
    if getattr(a, "role", None) == "A2" and hasattr(a, "load_q_tables"):
        if QS_PATH.exists():
            a.load_q_tables(str(QS_PATH))
        else:
            logging.getLogger("Main").info(f"A2 Q-Datei nicht gefunden: {QS_PATH}; starte frisch")

scheduler = Scheduler(agents, world, bus=bus)

visited = set()
game_over = False
win = False

# ------------------------------------------------------------
# Spiel-Reset: Q speichern, Agenten neu erzeugen, Q wieder laden
# ------------------------------------------------------------
def reset_game():
    global agents, scheduler, visited, game_over, win

    # A2 Q-Tables speichern bevor Agenten ersetzt werden
    for a in agents:
        if getattr(a, "role", None) == "A2" and hasattr(a, "save_q_tables"):
            try:
                a.save_q_tables(str(QS_PATH))
            except Exception:
                logging.getLogger("Main").exception("Fehler beim Speichern der A2 Q-Tables vor Reset")

    # Agenten und Scheduler neu erzeugen
    agents = create_agents()

    # Gespeicherte Q in die neue A2 Instanz laden (falls vorhanden)
    for a in agents:
        if getattr(a, "role", None) == "A2" and hasattr(a, "load_q_tables"):
            if QS_PATH.exists():
                a.load_q_tables(str(QS_PATH))

    scheduler = Scheduler(agents, world, bus=bus)

    visited.clear()
    world.reset()
    game_over = False
    win = False

    for agent in agents:
        visited.add(agent.pos())

# ------------------------------------------------------------
# Haupt-Spiel-Schleife (Pygame)
# ------------------------------------------------------------
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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_game()

        if not (game_over or win):
            result = scheduler.step()

            # Agent, der gerade gehandelt hat, ist (turn - 1)
            current_agent = agents[(scheduler.turn - 1) % len(agents)]
            visited.add(current_agent.pos())

            if result == "ALL_DEAD":
                logging.getLogger("Main").info("ALL_DEAD erkannt -> reset")
                game_over = True
            elif result == "WIN":
                logging.getLogger("Main").info("WIN erkannt -> reset")
                win = True

        clock.tick(10)

    pygame.quit()
    sys.exit()

# ------------------------------------------------------------
# Programmstart: game_loop, mit finaler Speicherung von A2 Q
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        game_loop()
    finally:
        # Beim Beenden A2 Q-Tables speichern
        for a in agents:
            if getattr(a, "role", None) == "A2" and hasattr(a, "save_q_tables"):
                try:
                    a.save_q_tables(str(QS_PATH))
                except Exception:
                    logging.getLogger("Main").exception("Fehler beim Speichern der A2 Q-Tables beim Beenden")

def clear_logs():
    """Truncate/clear relevant log files and MessageBus inboxes."""
    try:
        
        Path("wumpus_messages.log").write_text("")
    except Exception:
        logging.getLogger("Main").exception("Failed to clear wumpus_messages.log")

    try:
        
        Path("wumpus.log").write_text("")
    except Exception:
        logging.getLogger("Main").exception("Failed to clear wumpus.log")

    except Exception:
        
        logging.getLogger("Main").exception("Failed to clear MessageBus inboxes")
