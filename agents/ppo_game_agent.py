import torch
import torch.nn as nn
from agents.ppo_policy import PPONetwork
from environment.actions import Action
import numpy as np


class PPOGameAgent:
    """PPO-Agent für das laufende Spiel (Inference Only).
    Erwartet MemoryGrid (4, H, W) vom Scheduler.
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PPONetwork(grid_size, len(self.ACTIONS)).to(self.device)
        try:
            self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
            print("[PPOGameAgent] Modell geladen:", model_path)
        except Exception as e:
            print("[PPOGameAgent] WARNUNG: Modell konnte nicht geladen werden:", e)

        self.policy.eval()

    def pos(self):
        return (self.x, self.y)

    def decide_move(self, observation, grid_size):
        """observation kommt als (4, H, W) MemoryGrid vom Scheduler."""
        obs = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            probs = self.policy(obs)          # Softmax-Output
            action_idx = probs.argmax(dim=1).item()

        return self.ACTIONS[action_idx]
