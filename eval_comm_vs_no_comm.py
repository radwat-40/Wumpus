#!/usr/bin/env python3
"""
Evaluation script for comparing communication vs no-communication in Hunt-the-Wumpus multi-agent system.
Runs 500 episodes with and without MessageBus communication using the same seeds for fair comparison.
"""

import argparse
import random
import sys
import math
from pathlib import Path
from collections import defaultdict

# Defensive imports with clear error messages
try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed. Please install with: pip install numpy")
    sys.exit(1)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: torch not available, skipping torch.manual_seed")

# Repository imports
try:
    from environment.world import World
    from environment.scheduler import Scheduler
    from agents.agent1 import MarcAgent
    from agents.agent2 import YahiaAgent
    from agents.agent3 import HenrikAgent
    from logger.logger import MessageBus
except ImportError as e:
    print(f"ERROR: Failed to import repository modules: {e}")
    print("Make sure you're running this script from the Wumpus repository root.")
    sys.exit(1)


def set_seed(seed):
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)


def create_agents():
    """Create the three agents A1, A2, A3."""
    agents = []
    # Agent positions: A1 at (0,0), A2 at (1,0), A3 at (2,0) as per world.py
    agents.append(MarcAgent(0, 0, "A1"))
    agents.append(YahiaAgent(1, 0, "A2"))
    agents.append(HenrikAgent(2, 0, "A3"))
    return agents


def run_episode(seed, max_steps, use_bus=True):
    """
    Run a single episode.
    
    Returns:
        outcome: "WIN", "ALL_DEAD", or "TIMEOUT"
        steps: number of steps taken
    """
    set_seed(seed)
    
    # Initialize world with standard parameters
    world = World(grid_size=20, num_pits=20, num_wumpus=3, num_gold=1)
    
    # Create agents
    agents = create_agents()
    
    # Create MessageBus if needed
    bus = MessageBus() if use_bus else None
    
    # Register agents on bus if using communication
    if bus:
        for agent in agents:
            bus.register(agent.role)
    
    # Create scheduler
    scheduler = Scheduler(agents, world, bus)
    
    steps = 0
    outcome = "TIMEOUT"  # default
    
    while steps < max_steps:
        result = scheduler.step()
        steps += 1
        
        if result == "WIN":
            outcome = "WIN"
            break
        elif result == "ALL_DEAD":
            outcome = "ALL_DEAD"
            break
    
    return outcome, steps


def wilson_score_interval(successes, total, confidence=0.95):
    """Calculate Wilson score confidence interval for proportion."""
    if total == 0:
        return 0.0, 0.0
    
    p_hat = successes / total
    z = 1.96  # for 95% confidence
    
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) / denominator
    
    return max(0, center - spread), min(1, center + spread)


def calculate_stats(results):
    """Calculate statistics from results list of (outcome, steps)."""
    total = len(results)
    wins = sum(1 for outcome, _ in results if outcome == "WIN")
    fails = sum(1 for outcome, _ in results if outcome in ("ALL_DEAD", "TIMEOUT"))
    
    win_rate = wins / total * 100.0
    fail_rate = fails / total * 100.0
    
    all_steps = [steps for _, steps in results]
    avg_steps = sum(all_steps) / total
    win_steps = [steps for outcome, steps in results if outcome == "WIN"]
    avg_win_steps = sum(win_steps) / len(win_steps) if win_steps else float('nan')
    
    # 95% confidence interval for win rate
    ci_low, ci_high = wilson_score_interval(wins, total)
    ci_low *= 100.0
    ci_high *= 100.0
    
    return {
        'win_rate': win_rate,
        'fail_rate': fail_rate,
        'avg_steps': avg_steps,
        'avg_win_steps': avg_win_steps,
        'ci_low': ci_low,
        'ci_high': ci_high
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate communication vs no-communication in Wumpus world")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to run (default: 500)")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode (default: 500)")
    parser.add_argument("--silent", action="store_true", help="Suppress progress output")
    
    args = parser.parse_args()
    
    episodes = args.episodes
    max_steps = args.max_steps
    silent = args.silent
    
    print(f"Running evaluation: {episodes} episodes, max {max_steps} steps per episode")
    print("=" * 60)
    
    # Results storage
    results_with_comm = []
    results_without_comm = []
    
    for seed in range(episodes):
        if not silent and (seed + 1) % 50 == 0:
            print(f"Progress: {seed + 1}/{episodes} episodes completed")
        
        # With communication
        outcome, steps = run_episode(seed, max_steps, use_bus=True)
        results_with_comm.append((outcome, steps))
        
        # Without communication
        outcome, steps = run_episode(seed, max_steps, use_bus=False)
        results_without_comm.append((outcome, steps))
    
    # Calculate statistics
    stats_comm = calculate_stats(results_with_comm)
    stats_no_comm = calculate_stats(results_without_comm)
    
    # Output results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nWITH COMMUNICATION:")
    print(f"Win Rate: {stats_comm['win_rate']:.1f}% (95% CI: {stats_comm['ci_low']:.1f}% - {stats_comm['ci_high']:.1f}%)")
    print(f"Fail Rate: {stats_comm['fail_rate']:.1f}%")
    print(f"Average Steps: {stats_comm['avg_steps']:.1f}")
    print(f"Average Win Steps: {stats_comm['avg_win_steps']:.1f}")
    
    print("\nWITHOUT COMMUNICATION:")
    print(f"Win Rate: {stats_no_comm['win_rate']:.1f}% (95% CI: {stats_no_comm['ci_low']:.1f}% - {stats_no_comm['ci_high']:.1f}%)")
    print(f"Fail Rate: {stats_no_comm['fail_rate']:.1f}%")
    print(f"Average Steps: {stats_no_comm['avg_steps']:.1f}")
    print(f"Average Win Steps: {stats_no_comm['avg_win_steps']:.1f}")
    
    print("\nCOMPARISON:")
    win_diff = stats_comm['win_rate'] - stats_no_comm['win_rate']
    steps_diff = stats_comm['avg_steps'] - stats_no_comm['avg_steps']
    print(f"Win Rate Difference: {win_diff:.1f}% (Comm - No Comm)")
    print(f"Average Steps Difference: {steps_diff:.1f} (Comm - No Comm)")


if __name__ == "__main__":
    main()