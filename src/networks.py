"""
Actor and Critic Neural Networks for PPO
==========================================
Both networks use a shared MLP backbone with separate heads.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class MLP(nn.Module):
    """Shared multi-layer perceptron backbone."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Actor(nn.Module):
    """
    Gaussian policy network.
    Outputs mean and log-std of the action distribution.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128,
                 log_std_init: float = -0.5):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden)
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std   = nn.Parameter(
            torch.ones(act_dim) * log_std_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_head(self.backbone(obs))

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.forward(obs)
        std  = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)


class Critic(nn.Module):
    """State-value network V(s)."""

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            MLP(obs_dim, hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
