"""
Rollout Buffer for PPO
=======================
Stores (obs, action, log_prob, reward, done, value) tuples
collected during environment interaction.
"""

from __future__ import annotations
from typing import List
import numpy as np


class RolloutBuffer:
    """Fixed-capacity circular rollout buffer."""

    def __init__(self):
        self.obs:       List[np.ndarray] = []
        self.actions:   List[np.ndarray] = []
        self.log_probs: List[float]      = []
        self.rewards:   List[float]      = []
        self.dones:     List[bool]       = []
        self.values:    List[float]      = []

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.rewards)
