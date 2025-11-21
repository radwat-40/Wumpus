import torch
import torch.nn as nn


class PPONetwork(nn.Module):
    def __init__(self, grid_size: int, n_actions: int):
        super().__init__()
        input_dim = 4 * grid_size * grid_size  # 4 Kanäle, grid_size x grid_size

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x erwartet: (B, 4, H, W)
        x = x.view(x.size(0), -1)
        return self.fc(x)
