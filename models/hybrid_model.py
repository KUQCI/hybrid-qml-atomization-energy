"""
Hybrid quantum-classical regression model for QM7 eigenvalue features.

Architecture: Linear(23 -> n_qubits) -> AngleRescale -> quantum layer -> Linear(n_qubits -> 1)
"""
import torch
from torch import nn

from models.quantum_layer import build_quantum_layer

FEATURE_DIM = 23


class AngleRescale(nn.Module):
    """Squash encoder outputs into [-pi, pi] so AngleEmbedding receives
    bounded rotation angles regardless of encoder weight magnitudes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pi * torch.tanh(x)


def build_model(n_qubits: int = 4, depth: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, n_qubits),
        AngleRescale(),
        build_quantum_layer(n_qubits=n_qubits, depth=depth),
        nn.Linear(n_qubits, 1),
    )


def build_classical_ablation(n_qubits: int = 4, depth: int = 2) -> nn.Sequential:
    """Identical architecture with the quantum layer swapped for a plain
    linear layer of the same input/output width. The scientific control:
    if the hybrid model doesn't beat this, the quantum layer isn't helping.
    `depth` is accepted (and ignored) so both builders share a config."""
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, n_qubits),
        AngleRescale(),
        nn.Linear(n_qubits, n_qubits),
        nn.Linear(n_qubits, 1),
    )
