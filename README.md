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

Use Python 3.12 (DeepChem does not support 3.13 yet).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify environment (must PASS before Phase 2)
python scripts/smoke_test.py

# Download QM7 and create the seeded 70/15/15 splits
python data/load_qm7.py
```

macOS note: if `load_qm7.py` fails with `SSL: CERTIFICATE_VERIFY_FAILED`
(python.org framework builds don't use system certificates), run it as:

```bash
SSL_CERT_FILE="$(python -m certifi)" python data/load_qm7.py
```

The loader drops ~650 of QM7's 7,165 molecules that fail Coulomb matrix
featurization, leaving 6,515. Labels are normalized to train-set statistics;
`data/splits/y_transformer.pkl` maps predictions back to kcal/mol.

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

Active development — Phase 1 complete (environment verified, QM7 splits generated); starting Phase 2 (baseline + hybrid model training)
