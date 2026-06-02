import unittest

import torch
from torch import nn

from models.quantum_layer import build_quantum_layer
from scripts.hybrid_model_skeleton import (
    BATCH_SIZE,
    FEATURE_DIM,
    N_QUBITS,
    build_model,
)


class QuantumLayerTests(unittest.TestCase):
    def test_forward_shape_matches_qubits(self):
        layer = build_quantum_layer(n_qubits=3, depth=1)
        x = torch.randn(5, 3)

        out = layer(x)

        self.assertEqual(tuple(out.shape), (5, 3))

    def test_quantum_layer_backward_sets_gradients(self):
        layer = build_quantum_layer(n_qubits=N_QUBITS, depth=1)
        x = torch.randn(2, N_QUBITS)

        loss = layer(x).sum()
        loss.backward()

        self.assertTrue(all(p.grad is not None for p in layer.parameters()))


class HybridModelSkeletonTests(unittest.TestCase):
    def test_build_model_returns_expected_sequential_model(self):
        model = build_model()

        self.assertIsInstance(model, nn.Sequential)
        self.assertEqual(len(model), 3)
        self.assertIsInstance(model[0], nn.Linear)
        self.assertIsInstance(model[2], nn.Linear)
        self.assertEqual(model[0].in_features, FEATURE_DIM)
        self.assertEqual(model[0].out_features, N_QUBITS)
        self.assertEqual(model[2].in_features, N_QUBITS)
        self.assertEqual(model[2].out_features, 1)

    def test_hybrid_model_forward_and_backward(self):
        torch.manual_seed(42)
        model = build_model()
        x = torch.randn(BATCH_SIZE, FEATURE_DIM)
        y = torch.randn(BATCH_SIZE, 1)

        predictions = model(x)
        loss = nn.functional.mse_loss(predictions, y)
        loss.backward()

        self.assertEqual(tuple(predictions.shape), (BATCH_SIZE, 1))
        self.assertTrue(any(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
