# Suppressing Flow Separation Over a Flat Plate Using Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-APS%20DFD%202019-blue?style=flat)](https://ui.adsabs.harvard.edu/abs/2019APS..DFDQ33006K/abstract)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-EE4C2C?logo=pytorch)](https://pytorch.org)

> **Official implementation** of the reinforcement learning framework for active drag reduction by suppressing flow separation over a flat plate, using a linear actuator modulated by a PPO (Proximal Policy Optimization) agent trained on PIV-derived friction coefficient measurements.

---

## 📄 Abstract

Suppressing flow separation in wall-bounded flows is essential for **drag reduction** — decreasing fuel consumption, pollutant emissions, and noise in aerial and marine vehicles. Multiple cylindrical roughness elements were placed on a flat plate to induce flow separation. The boundary layer development was examined using **Particle Image Velocimetry (PIV)**, a soap-film setup, and numerical simulations (CFD). A **linear actuator** modulates the near-wall structure and measures the resulting change in the **friction coefficient (Cf)**. This data is incorporated with a **Reinforcement Learning (RL)** algorithm to reduce drag by delaying flow separation.

---

## 🧠 Method Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 Flat Plate Flow Environment                  │
│                                                              │
│  Roughness elements → Boundary layer separation             │
│  PIV / CFD velocity field  →  Friction coefficient (Cf)     │
│  Linear actuator  →  Near-wall flow modulation              │
└──────────────────┬────────────────────────────┬─────────────┘
                   │  State: Cf, u, v profile    │  Action: actuator position
                   ▼                             │
          ┌────────────────┐                     │
          │   PPO Agent    │─────────────────────┘
          │  (Actor-Critic)│
          └────────┬───────┘
                   │  Reward: ΔCf reduction (drag ↓)
                   ▼
           Training Loop (episodes → convergence)
```

**State space:** Friction coefficient $C_f$, streamwise ($u$) and wall-normal ($v$) velocity profiles extracted from PIV / CFD at measurement stations.

**Action space:** Continuous actuator displacement $[-1, +1]$ (normalized).

**Reward:** Reduction in mean friction coefficient $\Delta C_f = C_{f,\text{ref}} - C_{f,\text{controlled}}$

---

## 📁 Repository Structure

```
Reinforcement-Learning-Flow-Control/
├── src/
│   ├── environment.py   # Custom OpenAI Gym environment for flat-plate flow
│   ├── agent.py         # PPO agent (Actor-Critic networks)
│   ├── networks.py      # Actor and Critic neural networks
│   ├── buffer.py        # Rollout buffer for PPO
│   └── utils.py         # Cf computation, PIV preprocessing, plotting
├── train.py             # Main RL training script
├── evaluate.py          # Evaluate trained policy + drag reduction metrics
├── visualize.py         # Plot training curves, velocity fields, Cf maps
├── cfd_preprocess.py    # Preprocess CFD/PIV data into environment-ready format
├── configs/
│   └── config.yaml      # All hyperparameters
├── data/
│   └── README.md        # Data format description
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/kazemiakk/Reinforcement-Learning-Flow-Control.git
cd Reinforcement-Learning-Flow-Control

python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

---

## 📂 Data Format

The environment reads pre-processed flow field data in **NPY or CSV** format:

```
data/
├── cfd/
│   ├── velocity_field_000.npy    ← (Nx, Ny, 3): [u, v, p] at each grid point
│   └── ...
└── piv/
    ├── piv_frame_000.npy         ← (Nx, Ny, 2): [u, v] measured by PIV
    └── ...
```

See [`data/README.md`](data/README.md) for details on how to export CFD data from ANSYS Fluent.

---

## 🚀 Usage

### Training

```bash
python train.py \
  --data_dir  ./data/cfd \
  --output_dir ./checkpoints \
  --episodes  2000 \
  --max_steps 200
```

### Evaluation

```bash
python evaluate.py \
  --checkpoint ./checkpoints/best_policy.pth \
  --data_dir   ./data/cfd \
  --output_dir ./results
```

### Visualisation

```bash
python visualize.py \
  --log_dir    ./checkpoints \
  --output_dir ./results
```

### Preprocess CFD Data

```bash
python cfd_preprocess.py \
  --input_dir  /path/to/ansys/export \
  --output_dir ./data/cfd
```

---

## 📊 Key Results

| Metric | Baseline (No Control) | RL-Controlled |
|--------|----------------------|---------------|
| Mean Cf | Reference | Reduced ↓ |
| Separation onset | Early | Delayed |
| Actuator energy | — | Minimal |

---

## 📚 Citation

```bibtex
@inproceedings{kazemi2019suppressing,
  title     = {Suppressing flow separation over a flat plate using machine learning},
  author    = {Kazemi, Amirkhosro and Rousseau, Paul and Gomez, Daniel and
               Sureshkumar Nair, Aishwarya and Castillo, Luciano and
               Verma, Siddhartha and Curet, Oscar M.},
  booktitle = {APS Division of Fluid Dynamics Meeting Abstracts},
  volume    = {2019},
  pages     = {Q33.006},
  year      = {2019},
  month     = {November}
}
```

---

## 🏛️ Acknowledgements

This work was conducted at **Florida Atlantic University** (FAU) and **Texas Tech University** in collaboration with researchers in experimental fluid mechanics and active flow control.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
