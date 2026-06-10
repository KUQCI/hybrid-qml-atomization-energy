"""
Minimal hybrid model demo for QM7-style eigenvalue features.

The model itself lives in models/hybrid_model.py; this script just runs a
forward/backward pass on random data to show the pipeline is differentiable.

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

from models.hybrid_model import FEATURE_DIM, build_model

N_QUBITS = 4
DEPTH = 2
BATCH_SIZE = 8


def main() -> None:
    torch.manual_seed(42)

    model = build_model(n_qubits=N_QUBITS, depth=DEPTH)
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
