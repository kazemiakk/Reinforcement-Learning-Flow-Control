"""
Visualise Training Results and Flow Fields
==========================================
Usage
-----
python visualize.py --log_dir ./checkpoints --output_dir ./results
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.utils import plot_training_curves, plot_velocity_field, compute_cf


def parse_args():
    p = argparse.ArgumentParser(description="Visualise RL training results")
    p.add_argument("--log_dir",    type=str, default="./checkpoints")
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--data_dir",   type=str, default="./data/cfd",
                   help="CFD data dir to visualise a sample velocity field")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training curves ----
    hist_file = Path(args.log_dir) / "training_history.json"
    if hist_file.exists():
        with open(str(hist_file)) as f:
            hist = json.load(f)
        plot_training_curves(
            rewards=hist["episode_rewards"],
            cf_reductions=hist["cf_reductions"],
            save_path=str(out_dir / "training_curves.pdf"),
        )
        print(f"Saved training curves to {out_dir / 'training_curves.pdf'}")
    else:
        print(f"No training_history.json found in {args.log_dir}")

    # ---- Sample velocity field ----
    data_path = Path(args.data_dir)
    npy_files = sorted(data_path.glob("*.npy"))
    if npy_files:
        field = np.load(str(npy_files[0]))
        if field.ndim == 3 and field.shape[-1] >= 2:
            u = field[..., 0]
            v = field[..., 1]
            cf = compute_cf(u)
            plot_velocity_field(
                u, v, cf=cf,
                title="Sample Flow Field (Baseline)",
                save_path=str(out_dir / "sample_velocity_field.pdf"),
            )
            print(f"Saved velocity field to {out_dir / 'sample_velocity_field.pdf'}")
    else:
        print(f"No NPY flow fields found in {args.data_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
