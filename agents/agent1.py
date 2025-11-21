import torch
import torch.nn as nn
from agents.ppo_policy import PPONetwork
from environment.actions import Action
import numpy as np
import os


class MarcAgent:
    """
    PPO-Inference-Agent für das Wumpus-Spiel.
    Erwartet als Observation ein MemoryGrid: (4, H, W)
    """

    ACTIONS = [
        Action.MOVE_UP,
        Action.MOVE_DOWN,
        Action.MOVE_LEFT,
        Action.MOVE_RIGHT,
        Action.GRAB,
        Action.SHOOT_UP,
        Action.SHOOT_DOWN,
        Action.SHOOT_LEFT,
        Action.SHOOT_RIGHT,
    ]

    def __init__(self, x, y, role, grid_size=20, model_path="ppo_model.pth"):
        self.x = x
        self.y = y
        self.role = role
        self.agent_alive = True
        self.agent_won = False

        self.grid_size = grid_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # PPO-Policy laden
        self.policy = PPONetwork(grid_size, len(self.ACTIONS)).to(self.device)

        if os.path.exists(model_path):
            try:
                self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"[PPO] Modell geladen: {model_path}")
            except Exception as e:
                print("[PPO] FEHLER beim Laden:", e)
        else:
            print(f"[PPO] WARNUNG: {model_path} existiert nicht – Agent spielt zufällig.")

        self.policy.eval()

    def pos(self):
        return (self.x, self.y)

    # PPO-Action Auswahl
    def decide_move(self, obs, grid_size):
        """
        obs: numpy array (4, H, W)
        gibt Action Enum zurück
        """

        # Safety: convert to tensor
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            probs = self.policy(obs_t)  # (1, n_actions)

        # Fallback wenn das Modell Müll liefert
        if torch.isnan(probs).any():
            print("[PPO] WARNUNG: NaN-Policy → random action")
            action_idx = np.random.randint(len(self.ACTIONS))
        else:
            action_idx = torch.multinomial(probs[0], 1).item()

        return self.ACTIONS[action_idx]
