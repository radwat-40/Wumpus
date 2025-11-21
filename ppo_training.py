from agents.ppo_agent import PPOAgent
from agents.agent1 import MarcAgent
from environment.env_memory import WumpusMemoryEnv
import torch
import numpy as np


def train_ppo(num_episodes: int = 500, grid_size: int = 8):
    # Basisagent für die Env (nur Position/Percepts), PPO übernimmt Policy
    base_agent = MarcAgent(0, 0, "A1")

    env = WumpusMemoryEnv(
        agent=base_agent,
        grid_size=grid_size,
        num_pits=3,
        num_wumpus=1,
        num_gold=1,
        max_steps=200,
    )

    ACTIONS = base_agent.ACTIONS
    n_actions = len(ACTIONS)

    ppo_agent = PPOAgent(grid_size=grid_size, n_actions=n_actions)

    wins = 0
    deaths = 0

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        last_result = None

        while not done:
            # Safety: State darf keine NaNs / infs enthalten
            if not np.isfinite(state).all():
                print(f"[WARN] Episode {ep}: invalid state (NaN/inf). Reset episode.")
                break

            # Aktion wählen – falls Policy schon NaN ist, Agent neu initialisieren
            try:
                action_idx, logprob = ppo_agent.choose_action(state)
            except ValueError as e:
                print(f"[WARN] Episode {ep}: invalid action distribution ({e}). "
                      f"Reinitialisiere PPOAgent und breche Episode ab.")
                ppo_agent = PPOAgent(grid_size=grid_size, n_actions=n_actions)
                break

            action_enum = ACTIONS[action_idx]

            next_state, reward, done, info = env.step(action_enum)

            ppo_agent.store_transition(state, action_idx, reward, logprob, done)

            state = next_state
            last_result = info.get("result", None)

        # Episode vorbei → PPO-Update (nur wenn wir überhaupt Daten haben)
        ppo_agent.learn()

        if last_result == "WIN":
            wins += 1
        elif last_result == "DIED":
            deaths += 1

        if ep % 50 == 0:
            winrate = wins / ep * 100.0
            print(
                f"Episode {ep}/{num_episodes} | "
                f"Wins: {wins} | Deaths: {deaths} | Winrate: {winrate:.2f}%"
            )

    print("\n" + "=" * 60)
    print("PPO-Training abgeschlossen")
    print(f"Episoden: {num_episodes}")
    if num_episodes > 0:
        print(f"Wins: {wins} ({wins / num_episodes * 100:.2f}%)")
        print(f"Deaths: {deaths} ({deaths / num_episodes * 100:.2f}%)")
    print("=" * 60)

    torch.save(ppo_agent.policy.state_dict(), "ppo_model.pth")
    print("Modell gespeichert als ppo_model.pth")


if __name__ == "__main__":
    train_ppo(500, grid_size=8)
