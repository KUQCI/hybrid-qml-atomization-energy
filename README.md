# Hybrid QML — Atomization Energy Prediction

> QCI R&D | Quantum Machine Learning

A hybrid quantum-classical neural network that predicts molecular atomization energies on the QM7 dataset. The model encodes molecular features (Coulomb matrix eigenvalues) into a variational quantum circuit (PennyLane), whose expectation value outputs feed into a classical output layer trained end-to-end via PyTorch.

**Target:** < 1.0 kcal/mol MAE — chemical accuracy.

## Structure

```
.
├── data/               # Raw and preprocessed QM7 splits
├── models/             # Quantum, classical, and hybrid model definitions
├── notebooks/          # Experiments, results, and final demo notebook
├── results/            # Experiment logs (CSV), checkpoints, plots
└── scripts/            # Training, evaluation, and baseline scripts
```

## Setup

```bash
pip install -r requirements.txt
```

## Stack

| Component | Library |
|-----------|---------|
| Dataset | DeepChem (QM7, Coulomb matrix featurizer) |
| Quantum layer | PennyLane (`qml.QNode` + `qml.qnn.TorchLayer`) |
| Classical layers | PyTorch (`nn.Module`) |
| Baselines | scikit-learn (Kernel Ridge Regression) |
| Experiment tracking | CSV logger (model, MAE, hyperparams) |

## Models

- **Hybrid QML** — classical encoder → variational quantum circuit → linear output
- **Classical ablation** — identical architecture, quantum layer replaced with linear layer
- **KRR baseline** — Laplacian kernel ridge regression on eigenvalue features

## Status

Active development — Phase 1 (data + environment setup)
