"""
PPO Agent for Active Flow Control
===================================
Proximal Policy Optimization (PPO) with an Actor-Critic architecture.
Designed for the continuous-action flat plate flow control environment.

References
----------
Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.networks import Actor, Critic
from src.buffer   import RolloutBuffer


class PPOAgent:
    """
    PPO Agent with Gaussian policy for continuous action spaces.

    Parameters
    ----------
    obs_dim    : int   — observation space dimension
    act_dim    : int   — action space dimension
    lr_actor   : float — actor learning rate
    lr_critic  : float — critic learning rate
    gamma      : float — discount factor
    gae_lambda : float — GAE lambda for advantage estimation
    clip_eps   : float — PPO clipping epsilon
    n_epochs   : int   — number of PPO update epochs per rollout
    batch_size : int   — mini-batch size for updates
    ent_coef   : float — entropy bonus coefficient
    vf_coef    : float — value function loss coefficient
    max_grad_norm : float — gradient clipping norm
    device     : str   — 'cuda', 'mps', or 'cpu'
    """

    def __init__(
        self,
        obs_dim:       int,
        act_dim:       int,
        lr_actor:      float = 3e-4,
        lr_critic:     float = 1e-3,
        gamma:         float = 0.99,
        gae_lambda:    float = 0.95,
        clip_eps:      float = 0.2,
        n_epochs:      int   = 10,
        batch_size:    int   = 64,
        ent_coef:      float = 0.01,
        vf_coef:       float = 0.5,
        max_grad_norm: float = 0.5,
        device:        str   = "",
    ):
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_eps      = clip_eps
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.ent_coef      = ent_coef
        self.vf_coef       = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device        = self._get_device(device)

        self.actor  = Actor(obs_dim, act_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        self.actor_optim  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = RolloutBuffer()

    @staticmethod
    def _get_device(r: str) -> torch.device:
        if r: return torch.device(r)
        if torch.cuda.is_available():    return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        """
        Sample action from the policy.

        Returns
        -------
        action : np.ndarray (act_dim,)
        log_prob : float
        value : float
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        dist  = self.actor.get_distribution(obs_t)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value    = self.critic(obs_t).squeeze()
        return (action.cpu().numpy().flatten(),
                log_prob.item(),
                value.item())

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        return self.critic(obs_t).item()

    # ------------------------------------------------------------------
    # GAE advantage computation
    # ------------------------------------------------------------------

    def compute_gae(self, last_value: float = 0.0):
        """Compute Generalised Advantage Estimation and returns."""
        rewards   = np.array(self.buffer.rewards,   dtype=np.float32)
        values    = np.array(self.buffer.values,     dtype=np.float32)
        dones     = np.array(self.buffer.dones,      dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = last_value if t == len(rewards) - 1 \
                       else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) \
                    - values[t]
            gae   = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, last_value: float = 0.0):
        """Run PPO update over all stored transitions."""
        advantages, returns = self.compute_gae(last_value)

        obs_arr      = np.array(self.buffer.obs,       dtype=np.float32)
        actions_arr  = np.array(self.buffer.actions,   dtype=np.float32)
        log_probs_arr = np.array(self.buffer.log_probs, dtype=np.float32)

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t      = torch.FloatTensor(obs_arr).to(self.device)
        actions_t  = torch.FloatTensor(actions_arr).to(self.device)
        old_lp_t   = torch.FloatTensor(log_probs_arr).to(self.device)
        adv_t      = torch.FloatTensor(advantages).to(self.device)
        ret_t      = torch.FloatTensor(returns).to(self.device)

        n = len(obs_arr)
        total_loss_log = {"actor": 0.0, "critic": 0.0, "entropy": 0.0}

        for _ in range(self.n_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = idx[start: start + self.batch_size]
                dist  = self.actor.get_distribution(obs_t[batch])
                new_lp = dist.log_prob(actions_t[batch]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                values  = self.critic(obs_t[batch]).squeeze()

                ratio = torch.exp(new_lp - old_lp_t[batch])
                surr1 = ratio * adv_t[batch]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * adv_t[batch]
                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, ret_t[batch])
                loss = (actor_loss
                        + self.vf_coef * critic_loss
                        - self.ent_coef * entropy)

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),
                                         self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(),
                                         self.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

                total_loss_log["actor"]   += actor_loss.item()
                total_loss_log["critic"]  += critic_loss.item()
                total_loss_log["entropy"] += entropy.item()

        self.buffer.clear()
        return total_loss_log

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        print(f"Loaded checkpoint from {path}")
