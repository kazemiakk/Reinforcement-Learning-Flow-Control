# Suppressing Flow Separation Using Machine Learning

[![Paper](https://img.shields.io/badge/Paper-APS%20DFD-blue)](https://ui.adsabs.harvard.edu/abs/2019APS..DFDQ33006K/abstract)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code and models for the paper: **"Suppressing flow separation over a flat plate using machine learning"** (APS Division of Fluid Dynamics, 2019).

## 📌 Overview
Flow separation is a major source of efficiency loss in aerodynamics. In this project, we employ Machine Learning to actively control and suppress flow separation over a flat plate. 

Specifically, this repository implements machine learning algorithms to intelligently optimize active flow control parameters, achieving significant drag reduction and separation suppression with minimal energy expenditure.

### Baseline Flow vs. ML-Controlled Flow
*(Insert a side-by-side GIF or image here comparing the uncontrolled flow with separation vs. the controlled flow)*
`![Flow Comparison](results/flow_comparison.gif)`

## ⚙️ Installation

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/kazemiakk/ML-Flow-Control.git
cd ML-Flow-Control
pip install -r requirements.txt
```

## 🚀 Quickstart

To evaluate the pre-trained model on the sample dataset:
```bash
python src/evaluate.py --weights results/pretrained_model.pt --data data/sample_flow.vtp
```

To train the model from scratch:
```bash
python src/train.py --config config.yaml
```

## 📚 Citation

If you find this code useful in your research, please consider citing our work:

```bibtex
@inproceedings{kazemi2019suppressing,
  title={Suppressing flow separation over a flat plate using machine learning},
  author={Kazemi, Amirkhosro and Rousseau, P and Gomez, D and Sureshkumar Nair, A and Castillo, L and Verma, S and Curet, OM},
  booktitle={APS Division of Fluid Dynamics Meeting Abstracts},
  year={2019}
}
```
