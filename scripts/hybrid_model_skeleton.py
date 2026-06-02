"""
Minimal hybrid model skeleton for QM7-style eigenvalue features.

Run from the repo root:
    python scripts/hybrid_model_skeleton.py
"""
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.quantum_layer import build_quantum_layer


FEATURE_DIM = 23
N_QUBITS = 4
DEPTH = 2
BATCH_SIZE = 8


def build_model() -> nn.Sequential:
    quantum_layer = build_quantum_layer(n_qubits=N_QUBITS, depth=DEPTH)
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, N_QUBITS), # input
        
        # hidden
        quantum_layer,

        nn.Linear(N_QUBITS, 1), # output
    )


def main() -> None:
    torch.manual_seed(42)

    model = build_model()
    x = torch.randn(BATCH_SIZE, FEATURE_DIM)
    y = torch.randn(BATCH_SIZE, 1)

    predictions = model(x)
    loss = nn.functional.mse_loss(predictions, y)
    loss.backward()

    has_gradients = any(
        parameter.grad is not None for parameter in model.parameters()
    )

    print(f"Prediction shape: {tuple(predictions.shape)}")
    print(f"Backward gradients present: {has_gradients}")


if __name__ == "__main__":
    main()
