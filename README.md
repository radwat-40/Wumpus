# Hunt the Wumpus – Multi-Agent System

A multi-agent version of the classic *Hunt the Wumpus* game. Three agents with different decision-making approaches explore a hidden grid world together, share what they perceive through a message bus, and try to find the gold without falling into a pit or walking into a Wumpus.

The game is visualised with **pygame**.

## The World

| Parameter | Value |
|---|---|
| Grid size | 20 × 20 |
| Pits | 20 |
| Wumpus | 3 |
| Gold | 1 |
| Start positions | A1 at (0,0), A2 at (1,0), A3 at (2,0) – these cells are always free |

Pits, Wumpus and gold are placed randomly at the start of each game.

**Percepts** – an agent only perceives its current cell:
- **Breeze** – a pit is on an adjacent cell
- **Stench** – a Wumpus is on an adjacent cell
- **Glitter** – the gold is on the current cell

**Actions** – `MOVE_UP / DOWN / LEFT / RIGHT`, `GRAB`, and `SHOOT_UP / DOWN / LEFT / RIGHT` (shooting is supported by the world, but none of the current agents use it).

**End of the game**
- An agent that enters a pit or a Wumpus cell dies; the others continue.
- The team **wins** as soon as one agent reaches the gold.
- The game is **over** when all agents are dead.

## Architecture

```
main.py                    # Game loop, pygame UI, agent setup
environment/
  world.py                 # World: placement of items, percepts, action execution
  scheduler.py             # Turn order, shared memory, rewards, message broadcasting
  actions.py               # Action enum
  drawing.py               # Rendering of grid, agents and legend
  env_memory.py            # Single-agent environment with memory grids (experimental)
agents/
  base_agent.py            # Common base class
  agent1.py                # A1 – MarcAgent   (rule-based + DQN hybrid)
  agent2.py                # A2 – YahiaAgent  (decomposed Q-learning)
  agent3.py                # A3 – HenrikAgent (rule-based explorer)
logger/
  logger.py                # MessageBus for agent communication
dqn_training.py            # Training script for the DQN of A1
agent_eval.py              # Evaluation of the agents over 500 episodes
eval_comm_vs_no_comm.py    # Comparison: with vs. without communication
tools/                     # Helpers to create and inspect the Q-tables of A2
data/                      # Q-tables of A2 (.pkl) and CSV exports
*.pth                      # Saved neural network weights
```

### Scheduler

The `Scheduler` runs the agents **turn by turn** (A1 → A2 → A3 → …), skipping dead agents. In each step it:

1. delivers new messages from the message bus to the active agent,
2. builds the observation for the agent and asks it for an action,
3. executes the action in the world,
4. updates the **shared team memory** (`known`, `safe`, `risky` cells),
5. applies reward shaping (−1 per step, −10 for entering a risky cell, +5 for a new cell, +100 for glitter),
6. broadcasts `Breeze_Detected` / `Stench_Detected` messages and, if an agent dies, an `AGENT_DIED` message with the position and cause.

### Communication (MessageBus)

All agents are registered on a shared `MessageBus`. Messages are broadcast to every agent's inbox and also written to `wumpus_messages.log`. This lets every agent learn from cells it has never visited itself – e.g. an agent that died reveals a confirmed pit or Wumpus to the rest of the team.

## The Agents

### A1 – MarcAgent: hybrid rule-based + Deep Q-Network

**Agent type:** model-based, utility-based agent with a learned fallback policy.

- Receives the shared team memory (`known`, `safe`, `risky`) from the scheduler and keeps its own suspicion counters for pits and Wumpus, built from the breeze/stench messages of its teammates.
- **Rule-based layer (first priority):**
  - grabs the gold when it perceives glitter,
  - escapes to a known safe neighbour cell when it perceives a breeze or stench,
  - otherwise scores every neighbour cell: penalties for risky cells, suspected pits/Wumpus and moving back; bonuses for safe and unexplored cells.
- **DQN layer (fallback):** if every option scores too low, a Deep Q-Network decides instead. The network (6 → 64 → 64 → 4) gets the observation `[x, y (normalised), breeze, stench, glitter, share of explored cells]` and outputs a Q-value per move.
- The network is trained offline with `dqn_training.py` (experience replay, target network, ε-greedy) and loaded from `dqn_agent_best.pth`. If no model is found, A1 falls back to random moves.

### A2 – YahiaAgent: decomposed Q-learning

**Agent type:** learning agent with an internal world model.

- Learns **three separate Q-tables**, one per goal: `gold`, `pit` and `wumpus`. Each component gets its own reward (+100 for gold, −100 for dying in a pit, −100 for dying to a Wumpus). The normalised Q-values are summed to choose an action.
- Keeps **breeze and stench maps** from the team messages and derives suspected and confirmed dangerous cells. Cells next to a percept-free cell are marked safe.
- **Action selection:**
  - confirmed dangers are avoided whenever an alternative exists,
  - ε-greedy exploration prefers the least-visited neighbour cell,
  - exploitation uses softmax (Boltzmann) sampling over the combined Q-values minus penalties for frequently visited cells, cycles and risky cells,
  - a deadlock detector raises exploration when the agent keeps moving between the same cells.
- The Q-tables persist across games in `data/agent2_qtables.pkl`, so A2 keeps learning over many runs.

### A3 – HenrikAgent: rule-based explorer

**Agent type:** model-based reflex agent with goal-directed exploration.

- Remembers visited, safe and dangerous cells as well as suspected and confirmed pits/Wumpus (from messages).
- When it perceives a breeze or stench, it **steps back** to the last safe cell.
- On safe cells it scores the neighbour cells: unexplored > known safe > dangerous, with penalties for suspected cells and frequently visited cells. Confirmed dangers are excluded.
- It targets the **nearest unexplored cell next to a safe cell** (frontier exploration) to avoid wandering in circles.

## Installation

Requirements: Python 3 and

```sh
pip install pygame numpy torch
```

## Usage

**Run the game**

```sh
python main.py
```

Press **R** to start a new game. A2's Q-tables are saved automatically on reset and when the window is closed.

**Train the DQN of A1**

```sh
python dqn_training.py
```

Saves `dqn_agent_best.pth` (best evaluation win rate) and `dqn_agent_last.pth`.

**Evaluate the agents** (500 episodes, win rate and average steps per agent)

```sh
python agent_eval.py
```

**Compare communication vs. no communication** (same random seeds for both runs, win rate with 95% confidence interval)

```sh
python eval_comm_vs_no_comm.py --episodes 500 --max_steps 500
```

**Inspect A2's Q-tables** (exports them to `data/*_q.csv`)

```sh
python tools/inspect_q.py
```

## Logs

- `wumpus.log` – detailed debug log of every step
- `wumpus_messages.log` – every message sent over the message bus (JSON lines)
