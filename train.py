"""
PPO Training Script for Flat Plate Flow Control
================================================
Trains a PPO agent to suppress flow separation by modulating
a linear actuator based on PIV/CFD-derived friction coefficient data.

Usage
-----
python train.py --data_dir ./data/cfd --output_dir ./checkpoints --episodes 2000
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.environment import FlatPlateFlowEnv
from src.agent       import PPOAgent


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO for flow control")
    p.add_argument("--data_dir",       type=str, default="./data/cfd")
    p.add_argument("--output_dir",     type=str, default="./checkpoints")
    p.add_argument("--episodes",       type=int, default=2000)
    p.add_argument("--max_steps",      type=int, default=200)
    p.add_argument("--rollout_steps",  type=int, default=400,
                   help="Collect this many steps before each PPO update")
    p.add_argument("--n_stations",     type=int, default=5)
    p.add_argument("--profile_points", type=int, default=20)
    p.add_argument("--lr_actor",       type=float, default=3e-4)
    p.add_argument("--lr_critic",      type=float, default=1e-3)
    p.add_argument("--gamma",          type=float, default=0.99)
    p.add_argument("--gae_lambda",     type=float, default=0.95)
    p.add_argument("--clip_eps",       type=float, default=0.2)
    p.add_argument("--n_epochs",       type=int,   default=10)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--ent_coef",       type=float, default=0.01)
    p.add_argument("--actuator_penalty", type=float, default=0.01)
    p.add_argument("--device",         type=str,   default="")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--log_every",      type=int,   default=50)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Environment ----
    env = FlatPlateFlowEnv(
        data_dir=args.data_dir,
        n_stations=args.n_stations,
        profile_points=args.profile_points,
        max_steps=args.max_steps,
        actuator_penalty=args.actuator_penalty,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"Obs dim: {obs_dim}  Act dim: {act_dim}")

    # ---- Agent ----
    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        device=args.device,
    )

    episode_rewards = []
    cf_reductions   = []
    best_reward     = -np.inf
    global_step     = 0

    for ep in range(1, args.episodes + 1):
        obs, _       = env.reset()
        ep_reward    = 0.0
        ep_cf_red    = 0.0
        step_in_ep   = 0

        while step_in_ep < args.max_steps:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, _, info = env.step(action)

            agent.buffer.add(obs, action, log_prob, reward, done, value)
            obs = next_obs
            ep_reward  += reward
            ep_cf_red  += info.get("drag_reduction", 0.0)
            step_in_ep += 1
            global_step += 1

            # PPO update every rollout_steps
            if global_step % args.rollout_steps == 0:
                last_val = agent.get_value(obs) if not done else 0.0
                agent.update(last_val)

            if done:
                break

        episode_rewards.append(ep_reward)
        cf_reductions.append(ep_cf_red / max(step_in_ep, 1))

        # Save best
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(str(out_dir / "best_policy.pth"))

        if ep % args.log_every == 0:
            mean_r  = np.mean(episode_rewards[-args.log_every:])
            mean_cf = np.mean(cf_reductions[-args.log_every:])
            print(f"Ep {ep:>5d}/{args.episodes}  "
                  f"MeanReward={mean_r:.4f}  "
                  f"MeanΔCf={mean_cf:.5f}  "
                  f"BestReward={best_reward:.4f}")

    # Save final checkpoint and history
    agent.save(str(out_dir / "final_policy.pth"))
    history = {"episode_rewards": episode_rewards,
               "cf_reductions":   cf_reductions}
    with open(str(out_dir / "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best reward: {best_reward:.4f}")
    print(f"Checkpoints and history saved to {out_dir}")


if __name__ == "__main__":
    main()
