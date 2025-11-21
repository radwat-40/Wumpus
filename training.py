"""
Schnelles Single-Agent-Training für MarcAgent (DQN) ohne GUI und ohne Scheduler.
"""

from agents.ppo_agent import PPOAgent
from agents.agent1 import MarcAgent
from environment.env_memory import WumpusMemoryEnv
from environment.actions import Action
import torch
import numpy as np





def train(num_episodes=10000):
    # Nur ein Agent, der lernt
    agent = MarcAgent(0, 0, "A1")

    # Environment: du kannst num_pits / num_wumpus zum Trainieren reduzieren
    env = WumpusSingleAgentEnv(
    agent=agent,
    grid_size=8,
    num_pits=3,
    num_wumpus=1,
    num_gold=1,
    max_steps=150,
    )

    wins = 0
    deaths = 0

    for episode in range(num_episodes):
        obs = env.reset()
        

        done = False
        last_result = None

        while not done:

            import time
            t0 = time.time()
            # Aktion vom DQN-Agent holen
            action = agent.decide_move(obs, env.world.grid_size)

            # Schritt in der Umgebung ausführen
            next_obs, reward, done, info = env.step(action)
            last_result = info.get("result", None)

            # Transition ins Replay-Buffer schreiben und lernen
            if hasattr(agent, "store_transition"):
                agent.store_transition(obs, action, reward, next_obs, done)
                agent.learn_step()

            obs = next_obs

        
        # Statistik
        if last_result == "WIN":
            wins += 1
        elif last_result == "DIED":
            deaths += 1

        # Fortschritt alle 500 Episoden ausgeben & Checkpoint speichern
        if (episode + 1) % 500 == 0:
            win_rate = (wins / (episode + 1)) * 100
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Wins: {wins} | Deaths: {deaths} | "
                f"Win Rate: {win_rate:.2f}% | Epsilon A1: {agent.epsilon:.4f}"
            )
            agent.save_q_table("dqn_agent1.pth")

    # Abschluss-Checkpoint
    agent.save_q_table("dqn_agent1.pth")

    print("\n" + "=" * 60)
    print(f"Training abgeschlossen!")
    print(f"Gesamt Episoden: {num_episodes}")
    print(f"Wins: {wins} ({(wins/num_episodes)*100:.2f}%)")
    print(f"Deaths: {deaths} ({(deaths/num_episodes)*100:.2f}%)")
    print("Model checkpoints gespeichert: dqn_agent1.pth")
    print("=" * 60)


if __name__ == "__main__":
    episodes = 10000
    if len(sys.argv) > 1:
        try:
            episodes = int(sys.argv[1])
        except ValueError:
            print("Ungültige Episoden-Anzahl, verwende Standard: 10000")

    print(f"Starte Training mit {episodes} Episoden...")
    print("Keine GUI, Single-Agent, fortlaufend trainierend...\n")

    train(num_episodes=episodes)
