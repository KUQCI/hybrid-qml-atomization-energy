# Hybrid QML — Atomization Energy Prediction

> QCI R&D | Quantum Machine Learning

A hybrid quantum-classical neural network that predicts molecular atomization energies on the QM7 dataset. The model encodes molecular features (Coulomb matrix eigenvalues) into a variational quantum circuit (PennyLane), whose expectation value outputs feed into a classical output layer trained end-to-end via PyTorch.

## Targets & baselines

Accuracy standards for this project, in order of ambition:

| Bar | MAE (kcal/mol) | Meaning |
|-----|----------------|---------|
| Chemical accuracy | < 1.0 | The chemistry gold standard. **Not reachable with eigenvalue features** — published QM7 results below ~3 all use richer representations (full sorted/randomized Coulomb matrices) and large classical nets. Kept as aspirational context. |
| Eigenvalue-feature literature | ~10 | KRR on Coulomb-matrix *eigenvalues* (Rupp et al. 2012). The honest reference for any model trained on our feature set. |
| **Our KRR baseline** | **10.76 (test)** | Laplacian-kernel KRR on our exact features and splits (`scripts/baseline_krr.py`). Reproduces the literature value → pipeline verified. **This is the bar the hybrid model is judged against.** |
| Classical ablation | TBD | Same hybrid architecture with the quantum layer swapped for a linear layer. The hybrid must beat this for the quantum layer to be claimed useful. |

Caveats for any literature comparison: we train on 6,515 molecules (not
7,165 — see Setup) with a 70/15/15 random split, and our features are the
23-dim eigenvalue spectrum, which discards spatial information by design.

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

## Training & experiments

```bash
# Build eigenvalue features (once, after load_qm7.py)
python data/features.py

# Train the hybrid model (all runs append to results/experiments.csv)
python scripts/train.py --model-name hybrid_4q_d2 --epochs 200

# Classical ablation with identical settings
python scripts/train.py --model-name ablation_4q --classical --epochs 200

# KRR baseline
python scripts/baseline_krr.py
```

`--eval-test` evaluates the best checkpoint on the held-out test set —
use it once per final model, not during tuning.

## Status

Active development — Phases 1–2 complete (environment verified, splits + features
generated, hybrid model trains end-to-end). Phase 3 in progress: hyperparameter
sweep over qubits/depth/lr. First full-data hybrid run: 21.7 kcal/mol val MAE
vs the 10.76 KRR bar.
