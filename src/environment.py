"""
Flat Plate Flow Separation Environment
=======================================
Custom OpenAI Gym-compatible environment for active flow control
over a flat plate with cylindrical roughness elements.

Physics
-------
- Cylindrical roughness elements induce boundary layer separation.
- A linear actuator modulates near-wall flow structure.
- The friction coefficient Cf is the primary observable and reward signal.

State
-----
  s = [Cf_station_1, ..., Cf_station_N, u_profile, v_profile, actuator_pos]

Action
------
  a ∈ [-1, +1] continuous: normalized actuator displacement.

Reward
------
  r = Cf_ref - Cf_controlled   (positive = drag reduction)
  + small penalty for large actuator displacement (energy cost)

Episode terminates when max_steps is reached.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlatPlateFlowEnv(gym.Env):
    """
    Flat Plate Active Flow Control Environment.

    Parameters
    ----------
    data_dir : str
        Directory containing pre-processed velocity field NPY files.
    n_stations : int
        Number of streamwise measurement stations (default 5).
    profile_points : int
        Number of wall-normal points per velocity profile (default 20).
    max_steps : int
        Maximum steps per episode.
    actuator_penalty : float
        Coefficient penalising large actuator displacements (energy cost).
    noise_std : float
        Gaussian observation noise (simulates sensor noise).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        data_dir: str = "./data/cfd",
        n_stations: int = 5,
        profile_points: int = 20,
        max_steps: int = 200,
        actuator_penalty: float = 0.01,
        noise_std: float = 0.001,
    ):
        super().__init__()
        self.data_dir        = Path(data_dir)
        self.n_stations      = n_stations
        self.profile_points  = profile_points
        self.max_steps       = max_steps
        self.actuator_penalty = actuator_penalty
        self.noise_std       = noise_std

        # Load flow snapshots
        self._load_data()

        # Observation: [Cf × N, u_profile × P, v_profile × P, actuator_pos]
        obs_dim = n_stations + 2 * profile_points + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action: continuous actuator displacement ∈ [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self._step_count    = 0
        self._actuator_pos  = 0.0
        self._snapshot_idx  = 0

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        """Load all velocity field snapshots from data_dir."""
        files = sorted(self.data_dir.glob("*.npy"))
        if not files:
            # Generate synthetic placeholder data for testing
            print("[ENV] No data files found — using synthetic Poiseuille flow.")
            self._snapshots = self._synthetic_snapshots()
        else:
            self._snapshots = [np.load(str(f)).astype(np.float32)
                               for f in files]
        print(f"[ENV] Loaded {len(self._snapshots)} flow snapshots.")

    @staticmethod
    def _synthetic_snapshots(n: int = 50, Nx: int = 60, Ny: int = 30):
        """Generate simple synthetic flow fields for testing."""
        snaps = []
        for _ in range(n):
            y = np.linspace(0, 1, Ny)
            u = np.outer(np.ones(Nx), 6 * y * (1 - y)) + \
                0.02 * np.random.randn(Nx, Ny)   # Poiseuille + noise
            v = 0.01 * np.random.randn(Nx, Ny)
            p = np.zeros((Nx, Ny))
            snaps.append(np.stack([u, v, p], axis=-1).astype(np.float32))
        return snaps

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------

    def _get_snapshot(self) -> np.ndarray:
        return self._snapshots[self._snapshot_idx % len(self._snapshots)]

    def _compute_cf(self, field: np.ndarray, actuator_pos: float) -> np.ndarray:
        """
        Compute friction coefficient Cf at each measurement station.

        Cf ≈ 2 * (∂u/∂y)|_{y=0} / U_ref²
        Modulated by actuator displacement (linear perturbation model).
        """
        u = field[..., 0]   # (Nx, Ny) streamwise velocity
        Nx, Ny = u.shape
        du_dy = (u[:, 1] - u[:, 0])          # finite diff at wall

        # Station indices evenly distributed in x
        station_idx = np.linspace(0, Nx - 1, self.n_stations, dtype=int)
        cf_stations = du_dy[station_idx]

        # Actuator effect: linear perturbation (positive → reduces separation)
        cf_stations += actuator_pos * 0.05 * np.abs(cf_stations).mean()

        return cf_stations.astype(np.float32)

    def _get_profiles(self, field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract wall-normal u and v profiles at the mid-station."""
        Nx = field.shape[0]
        mid = Nx // 2
        u_profile = field[mid, :self.profile_points, 0]
        v_profile = field[mid, :self.profile_points, 1]
        # Pad or trim to profile_points
        def _fix(arr):
            if len(arr) < self.profile_points:
                return np.pad(arr, (0, self.profile_points - len(arr)))
            return arr[:self.profile_points]
        return _fix(u_profile), _fix(v_profile)

    def _build_obs(self, field: np.ndarray) -> np.ndarray:
        cf = self._compute_cf(field, self._actuator_pos)
        u_p, v_p = self._get_profiles(field)
        obs = np.concatenate([cf, u_p, v_p, [self._actuator_pos]],
                             dtype=np.float32)
        obs += np.random.normal(0, self.noise_std, obs.shape).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count   = 0
        self._actuator_pos = 0.0
        self._snapshot_idx = np.random.randint(0, len(self._snapshots))
        field = self._get_snapshot()
        self._cf_ref = float(np.mean(np.abs(
            self._compute_cf(field, 0.0))))
        obs = self._build_obs(field)
        return obs, {}

    def step(self, action: np.ndarray):
        action = float(np.clip(action[0], -1.0, 1.0))
        self._actuator_pos = action
        self._step_count  += 1
        self._snapshot_idx += 1

        field = self._get_snapshot()
        cf_controlled = float(np.mean(np.abs(
            self._compute_cf(field, action))))

        # Reward: drag reduction minus energy penalty
        drag_reduction = self._cf_ref - cf_controlled
        energy_penalty = self.actuator_penalty * abs(action)
        reward = float(drag_reduction - energy_penalty)

        obs  = self._build_obs(field)
        done = self._step_count >= self.max_steps
        info = {
            "Cf_ref": self._cf_ref,
            "Cf_controlled": cf_controlled,
            "drag_reduction": drag_reduction,
            "actuator_pos": action,
        }
        return obs, reward, done, False, info

    def render(self):
        pass   # visualization handled externally


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = FlatPlateFlowEnv(data_dir="./data/cfd", max_steps=10)
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")
    for _ in range(5):
        a = env.action_space.sample()
        obs, r, done, _, info = env.step(a)
        print(f"  action={a[0]:.3f}  reward={r:.5f}  Cf={info['Cf_controlled']:.5f}")
    print("Environment test passed.")
