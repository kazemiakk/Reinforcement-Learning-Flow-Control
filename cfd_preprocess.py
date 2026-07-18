"""
Preprocess CFD/PIV Data for the RL Environment
===============================================
Converts ANSYS Fluent CSV exports or PIV data into
NPY files of shape (Nx, Ny, C) where C = [u, v, p].

Usage
-----
# From ANSYS Fluent CSV export:
python cfd_preprocess.py \
  --input_dir /path/to/fluent/csv \
  --output_dir ./data/cfd \
  --fmt fluent_csv

# From PIV data:
python cfd_preprocess.py \
  --input_dir /path/to/piv \
  --output_dir ./data/piv \
  --fmt piv_csv
"""

import argparse
import glob
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess CFD/PIV data")
    p.add_argument("--input_dir",  type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./data/cfd")
    p.add_argument("--fmt",        type=str, default="fluent_csv",
                   choices=["fluent_csv", "piv_csv", "npy"])
    p.add_argument("--Nx",         type=int, default=60,
                   help="Target grid size in x (streamwise)")
    p.add_argument("--Ny",         type=int, default=30,
                   help="Target grid size in y (wall-normal)")
    return p.parse_args()


def load_fluent_csv(path: str, Nx: int, Ny: int) -> np.ndarray:
    """
    Load an ANSYS Fluent data export (CSV with x, y, u, v, p columns).
    Returns (Nx, Ny, 3) array [u, v, p].
    """
    import pandas as pd
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower()

    # Try common Fluent column names
    u_col = next((c for c in df.columns if 'velocity-x' in c or c == 'u'), None)
    v_col = next((c for c in df.columns if 'velocity-y' in c or c == 'v'), None)
    p_col = next((c for c in df.columns if 'pressure' in c or c == 'p'), None)

    u = df[u_col].values if u_col else np.zeros(len(df))
    v = df[v_col].values if v_col else np.zeros(len(df))
    p = df[p_col].values if p_col else np.zeros(len(df))

    # Reshape to grid (assume row-major ordering)
    n = len(u)
    Nx_r = min(Nx, int(np.sqrt(n * Nx / Ny)))
    Ny_r = min(Ny, n // Nx_r)
    u = u[:Nx_r * Ny_r].reshape(Nx_r, Ny_r)
    v = v[:Nx_r * Ny_r].reshape(Nx_r, Ny_r)
    p = p[:Nx_r * Ny_r].reshape(Nx_r, Ny_r)

    # Resize to target
    from scipy.ndimage import zoom
    def _resize(arr): return zoom(arr, (Nx / Nx_r, Ny / Ny_r))
    return np.stack([_resize(u), _resize(v), _resize(p)],
                    axis=-1).astype(np.float32)


def load_piv_csv(path: str, Nx: int, Ny: int) -> np.ndarray:
    """
    Load PIV data CSV (x, y, u, v columns).
    Returns (Nx, Ny, 3) array [u, v, zeros_for_p].
    """
    import pandas as pd
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip().str.lower()
    u = df.get("u", df.iloc[:, 2]).values.astype(np.float32)
    v = df.get("v", df.iloc[:, 3]).values.astype(np.float32)
    n = len(u)
    Nx_r = int(np.sqrt(n * Nx / Ny))
    Ny_r = n // Nx_r
    u = u[:Nx_r * Ny_r].reshape(Nx_r, Ny_r)
    v = v[:Nx_r * Ny_r].reshape(Nx_r, Ny_r)
    p = np.zeros_like(u)
    from scipy.ndimage import zoom
    def _r(a): return zoom(a, (Nx / Nx_r, Ny / Ny_r))
    return np.stack([_r(u), _r(v), _r(p)], axis=-1).astype(np.float32)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(Path(args.input_dir) / "*.csv")))
    if not files:
        files = sorted(glob.glob(str(Path(args.input_dir) / "*.npy")))
        args.fmt = "npy"

    print(f"Found {len(files)} files in {args.input_dir}")

    for i, f in enumerate(files):
        name = Path(f).stem
        if args.fmt == "fluent_csv":
            field = load_fluent_csv(f, args.Nx, args.Ny)
        elif args.fmt == "piv_csv":
            field = load_piv_csv(f, args.Nx, args.Ny)
        else:
            field = np.load(f).astype(np.float32)

        out = str(out_dir / f"velocity_field_{i:03d}.npy")
        np.save(out, field)
        if i % 50 == 0:
            print(f"  [{i+1}/{len(files)}] {name} → {field.shape}")

    print(f"\nDone. Saved {len(files)} NPY files to {out_dir}")


if __name__ == "__main__":
    main()
