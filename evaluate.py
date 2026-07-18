"""
Evaluate Trained PPO Policy on Flow Control Task
==================================================
Runs the trained policy deterministically and reports:
  - Mean/std episode reward
  - Mean Cf reduction vs. baseline
  - Separation point shift

Usage
-----
python evaluate.py \
  --checkpoint ./checkpoints/best_policy.pth \
  --data_dir ./data/cfd \
  --episodes 20 \
  --output_dir ./results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.environment import FlatPlateFlowEnv
from src.agent       import PPOAgent
from src.utils       import plot_velocity_field, compute_cf


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PPO flow control")
    p.add_argument("--checkpoint",     type=str, required=True)
    p.add_argument("--data_dir",       type=str, default="./data/cfd")
    p.add_argument("--episodes",       type=int, default=20)
    p.add_argument("--max_steps",      type=int, default=200)
    p.add_argument("--n_stations",     type=int, default=5)
    p.add_argument("--profile_points", type=int, default=20)
    p.add_argument("--output_dir",     type=str, default="./results")
    p.add_argument("--device",         type=str, default="")
    return p.parse_args()


@torch.no_grad()
def run_episode(agent, env, deterministic: bool = True):
    obs, _ = env.reset()
    total_reward = 0.0
    cf_reductions = []
    actuator_positions = []

    while True:
        action, _, _ = agent.select_action(obs)
        if deterministic:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            action = agent.actor(obs_t).cpu().numpy().flatten()

        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        cf_reductions.append(info.get("drag_reduction", 0.0))
        actuator_positions.append(info.get("actuator_pos", 0.0))
        if done:
            break

    return total_reward, np.mean(cf_reductions), actuator_positions


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = FlatPlateFlowEnv(
        data_dir=args.data_dir,
        n_stations=args.n_stations,
        profile_points=args.profile_points,
        max_steps=args.max_steps,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, device=args.device)
    agent.load(args.checkpoint)

    rewards, cf_reds = [], []
    print(f"Running {args.episodes} evaluation episodes...")

    for ep in range(args.episodes):
        r, cf, acts = run_episode(agent, env)
        rewards.append(r)
        cf_reds.append(cf)
        print(f"  Ep {ep+1:>3d}  Reward={r:.4f}  ΔCf={cf:.5f}  "
              f"ActMean={np.mean(np.abs(acts)):.3f}")

    summary = {
        "n_episodes"      : args.episodes,
        "reward_mean"     : float(np.mean(rewards)),
        "reward_std"      : float(np.std(rewards)),
        "cf_reduction_mean": float(np.mean(cf_reds)),
        "cf_reduction_std" : float(np.std(cf_reds)),
    }
    print(f"\n--- Summary ---")
    for k, v in summary.items():
        print(f"  {k:<26}: {v:.4f}" if isinstance(v, float) else
              f"  {k:<26}: {v}")

    with open(str(out_dir / "eval_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
