"""
Agent Evaluation - 500 Episode Test ohne GUI
Misst Win-Rate und Performance der drei Agenten
"""

import logging
import random
from pathlib import Path
from collections import defaultdict
import time

from environment.world import World
from environment.scheduler import Scheduler
from agents.agent1 import MarcAgent
from agents.agent2 import YahiaAgent
from agents.agent3 import HenrikAgent
from logger.logger import MessageBus


logging.basicConfig(level=logging.ERROR)


QS_PATH = Path("data/agent2_qtables.pkl")
QS_PATH.parent.mkdir(parents=True, exist_ok=True)


def create_agents(world_grid_size):
    """Erstellt frische Agenten für eine Episode"""
    agent1 = MarcAgent(0, 0, "A1")
    agent2 = YahiaAgent(1, 0, "A2")
    agent3 = HenrikAgent(2, 0, "A3")

    bus = MessageBus(persist_file=None)

    for a in (agent1, agent2, agent3):
        bus.register(a.role)
        a.bus = bus

    if hasattr(agent2, "init_maps"):
        agent2.init_maps(world_grid_size)

    if QS_PATH.exists() and hasattr(agent2, "load_q_tables"):
        try:
            agent2.load_q_tables(str(QS_PATH))
        except Exception:
            pass

    return [agent1, agent2, agent3], bus


def run_episode(world, agents, scheduler, max_steps=1000):
    """
    Führt eine Episode aus bis zum Sieg, Tod aller oder Max-Schritte
    
    Returns:
        (winner_role, steps_taken)
        winner_role: "A1", "A2", "A3" oder None (wenn keiner gewonnen hat)
    """
    step_count = 0
    
    while step_count < max_steps:
        step_count += 1
        
        result = scheduler.step()
        
        if result == "WIN":
            winner = agents[(scheduler.turn - 1) % len(agents)]
            return winner.role, step_count
        elif result == "ALL_DEAD":
            return None, step_count
        elif result == "CONTINUE":
            continue

    return None, step_count


def evaluate_agents(num_episodes=500):
    """
    Führt 500 Episoden aus und berechnet Win-Rates
    """
    stats = {
        "A1": {"wins": 0, "total": 0, "avg_steps": 0},
        "A2": {"wins": 0, "total": 0, "avg_steps": 0},
        "A3": {"wins": 0, "total": 0, "avg_steps": 0},
        "none": {"wins": 0, "total": 0},
    }
    
    all_steps = defaultdict(list)
    
    print(f"\n{'='*60}")
    print(f"Starting Agent Evaluation: {num_episodes} Episodes")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        world = World()
        agents, bus = create_agents(world.grid_size)
        scheduler = Scheduler(agents, world, bus=bus)
        
        winner, steps = run_episode(world, agents, scheduler)
        
        if winner:
            stats[winner]["wins"] += 1
            all_steps[winner].append(steps)
        else:
            stats["none"]["wins"] += 1
        
        for role in ["A1", "A2", "A3"]:
            stats[role]["total"] += 1
        
        if episode % 50 == 0:
            elapsed = time.time() - start_time
            rate = episode / elapsed
            eta = (num_episodes - episode) / rate if rate > 0 else 0
            print(f"Episode {episode:3d}/{num_episodes} | {rate:5.1f} ep/s | ETA: {eta:6.1f}s")
        
        if episode % 100 == 0 and hasattr(agents[1], "save_q_tables"):
            try:
                agents[1].save_q_tables(str(QS_PATH))
            except Exception:
                pass
    
    total_time = time.time() - start_time
    
    for role in ["A1", "A2", "A3"]:
        if all_steps[role]:
            stats[role]["avg_steps"] = sum(all_steps[role]) / len(all_steps[role])
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ({total_time:.1f}s)")
    print(f"{'='*60}\n")
    
    for role in ["A1", "A2", "A3"]:
        wins = stats[role]["wins"]
        wr_pct = (wins / num_episodes * 100)
        avg_steps = stats[role]["avg_steps"]
        
        print(f"{role}:")
        print(f"  Win Rate:   {wins:3d}/{num_episodes} ({wr_pct:5.1f}%)")
        print(f"  Avg Steps:  {avg_steps:6.1f}")
        print()
    
    draw_text = stats["none"]["wins"]
    print(f"No Winner:  {draw_text:3d}/{num_episodes} ({draw_text/num_episodes*100:5.1f}%)")
    
    print(f"\n{'='*60}")
    print("RANKING:")
    print(f"{'='*60}\n")
    
    ranking = sorted(
        [(role, stats[role]["wins"]) for role in ["A1", "A2", "A3"]],
        key=lambda x: x[1],
        reverse=True
    )
    
    for idx, (role, wins) in enumerate(ranking, 1):
        wr_pct = (wins / num_episodes * 100)
        print(f"{idx}. {role}: {wr_pct:5.1f}% ({wins} wins)")
    
    print(f"\n{'='*60}\n")
    
    if hasattr(agents[1], "save_q_tables"):
        try:
            agents[1].save_q_tables(str(QS_PATH))
            print(f"✓ A2 Q-Tables gespeichert")
        except Exception as e:
            print(f"✗ Fehler beim Speichern: {e}")
    
    return stats


if __name__ == "__main__":
    random.seed(42)
    stats = evaluate_agents(num_episodes=500)
