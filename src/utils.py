"""
Utility Functions for Flow Control RL
=======================================
- compute_cf         : Friction coefficient from velocity field
- load_piv_data      : Load PIV measurement data
- preprocess_velocity: Normalise velocity fields
- plot_training_curves : Reward and Cf reduction over training
- plot_velocity_field  : Visualise 2D velocity field + separation region
"""

from __future__ import annotations
from typing import Optional, List, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def compute_cf(u: np.ndarray, dy: float = 1.0,
               rho: float = 1.225, U_ref: float = 1.0) -> np.ndarray:
    """
    Compute friction coefficient Cf = 2*τ_w / (ρ*U_ref²)
    where τ_w = μ*(∂u/∂y)|_{y=0} ≈ μ*(u[:,1] - u[:,0]) / dy.

    Parameters
    ----------
    u     : (Nx, Ny) streamwise velocity array
    dy    : wall-normal grid spacing (m)
    rho   : fluid density (kg/m³)
    U_ref : reference velocity (m/s)

    Returns
    -------
    cf : (Nx,) friction coefficient along x
    """
    du_dy_wall = (u[:, 1] - u[:, 0]) / dy        # ∂u/∂y at wall
    tau_w = du_dy_wall                            # assuming μ=1 (normalised)
    cf = 2.0 * tau_w / (rho * U_ref ** 2)
    return cf.astype(np.float32)


def separation_point(cf: np.ndarray) -> Optional[int]:
    """
    Find the streamwise index where Cf first becomes negative
    (i.e., where flow reversal / separation begins).
    """
    neg = np.where(cf < 0)[0]
    return int(neg[0]) if len(neg) > 0 else None


def preprocess_velocity(field: np.ndarray) -> np.ndarray:
    """
    Normalise a velocity field array to zero mean and unit variance.

    Parameters
    ----------
    field : (Nx, Ny, C) array

    Returns
    -------
    normalised : (Nx, Ny, C) float32
    """
    mean = field.mean(axis=(0, 1), keepdims=True)
    std  = field.std(axis=(0, 1), keepdims=True) + 1e-8
    return ((field - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(
    rewards:       List[float],
    cf_reductions: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot episode reward and mean Cf reduction over training.
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    episodes = np.arange(1, len(rewards) + 1)

    axes[0].plot(episodes, rewards, alpha=0.4, color="#2196F3", linewidth=0.8)
    # Rolling average
    w = min(20, len(rewards) // 5 + 1)
    if len(rewards) >= w:
        smoothed = np.convolve(rewards, np.ones(w)/w, mode='valid')
        axes[0].plot(np.arange(w, len(rewards) + 1), smoothed,
                     color="#0D47A1", linewidth=1.8, label=f"MA({w})")
    axes[0].set_ylabel("Episode Reward")
    axes[0].set_title("RL Training: Flow Separation Suppression")
    axes[0].legend(fontsize=9)

    axes[1].plot(episodes, cf_reductions, alpha=0.4,
                 color="#4CAF50", linewidth=0.8)
    if len(cf_reductions) >= w:
        smoothed_cf = np.convolve(cf_reductions, np.ones(w)/w, mode='valid')
        axes[1].plot(np.arange(w, len(cf_reductions) + 1), smoothed_cf,
                     color="#1B5E20", linewidth=1.8, label=f"MA({w})")
    axes[1].axhline(0, color="red", linewidth=1, linestyle="--",
                    label="Baseline (Cf reduction = 0)")
    axes[1].set_ylabel("Drag Reduction ΔCf")
    axes[1].set_xlabel("Episode")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_velocity_field(
    u: np.ndarray,
    v: np.ndarray,
    cf: Optional[np.ndarray] = None,
    title: str = "Velocity Field",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualise 2D velocity magnitude + streamlines and Cf profile.
    """
    speed = np.sqrt(u ** 2 + v ** 2)
    Nx, Ny = u.shape
    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    ncols = 2 if cf is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    im = axes[0].contourf(X, Y, speed, levels=30, cmap="RdYlBu_r")
    axes[0].streamplot(x, y, u.T, v.T, color="white", linewidth=0.5,
                       density=1.0)
    plt.colorbar(im, ax=axes[0], label="Speed (m/s)")
    axes[0].set_xlabel("x (streamwise)")
    axes[0].set_ylabel("y (wall-normal)")
    axes[0].set_title(title)

    if cf is not None:
        sep = separation_point(cf)
        axes[1].plot(x, cf, "b-", linewidth=1.8)
        axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
        if sep is not None:
            axes[1].axvline(sep, color="orange", linestyle=":",
                            linewidth=1.5, label=f"Separation @ x={sep}")
        axes[1].set_xlabel("x (streamwise)")
        axes[1].set_ylabel("Cf")
        axes[1].set_title("Friction Coefficient Cf")
        axes[1].legend(fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
