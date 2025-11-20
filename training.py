"""
Training-Script für MarcAgent (Q-Learning)
Trainiert den Agenten für N Episoden ohne GUI
Die Q-Table wird nach jedem Spiel gespeichert und kann in main.py geladen werden
"""

import sys
from agents.agent1 import MarcAgent
from environment.world import World
from environment.scheduler import Scheduler

def train(num_episodes=10000):
    """Trainiere MarcAgent für N Episoden ohne GUI"""
    
    world = World()
    agent1 = MarcAgent(0, 0, "A1")
    agent2 = MarcAgent(1, 0, "A2")
    agent3 = MarcAgent(2, 0, "A3")
    
    agents = [agent1, agent2, agent3]
    scheduler = Scheduler(agents, world)
    
    wins = 0
    deaths = 0
    
    for episode in range(num_episodes):
        # Reset Welt und Agenten für neues Spiel
        world.reset()
        for agent in agents:
            agent.agent_alive = True
            agent.x, agent.y = 0, 0  # Reset Position
        scheduler.turn = 0
        
        episode_done = False
        steps = 0
        max_steps = 1000  # Verhindere infinite loops
        
        while not episode_done and steps < max_steps:
            result = scheduler.step()
            steps += 1
            
            if result == "WIN":
                wins += 1
                episode_done = True
            elif result == "ALL_DEAD":
                deaths += 1
                episode_done = True
        
        # Speichere Q-Table nach jedem Spiel
        agent1.save_q_table("q_table_agent1.npy")
        agent2.save_q_table("q_table_agent2.npy")
        agent3.save_q_table("q_table_agent3.npy")
        
        # Progress output alle 500 Episoden
        if (episode + 1) % 500 == 0:
            win_rate = (wins / (episode + 1)) * 100
            print(f"Episode {episode + 1}/{num_episodes} | Wins: {wins} | Deaths: {deaths} | Win Rate: {win_rate:.2f}% | Epsilon A1: {agent1.epsilon:.4f}")
    
    print("\n" + "="*60)
    print(f"Training abgeschlossen!")
    print(f"Gesamt Episoden: {num_episodes}")
    print(f"Wins: {wins} ({(wins/num_episodes)*100:.2f}%)")
    print(f"Deaths: {deaths} ({(deaths/num_episodes)*100:.2f}%)")
    print(f"Q-Tables gespeichert: q_table_agent*.npy")
    print("="*60)

if __name__ == "__main__":
    episodes = 10000
    
    # Optional: Episoden-Anzahl als Argument übergeben
    if len(sys.argv) > 1:
        try:
            episodes = int(sys.argv[1])
        except ValueError:
            print("Ungültige Episode-Anzahl, verwende Standard: 10000")
    
    print(f"Starte Training mit {episodes} Episoden...")
    print("Keine GUI, fortlaufend trainierend...\n")
    
    train(num_episodes=episodes)
