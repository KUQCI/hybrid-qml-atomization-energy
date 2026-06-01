"""
Smoke test: confirms PennyLane + PyTorch are installed correctly and
that gradients flow end-to-end through a TorchLayer before Phase 2.
Run once after install: python scripts/smoke_test.py
"""
import torch
import pennylane as qml

N_QUBITS = 4


def build_circuit():
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    return circuit


def main():
    circuit = build_circuit()

    weight_shapes = {"weights": (2, N_QUBITS)}
    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    x = torch.randn(3, N_QUBITS, requires_grad=False)
    out = qlayer(x)
    loss = out.sum()
    loss.backward()

    all_grads_ok = all(
        p.grad is not None for p in qlayer.parameters()
    )

    if all_grads_ok:
        print(f"PASS: gradients flow through TorchLayer  |  output shape: {out.shape}")
    else:
        print("FAIL: some parameters have None gradients — fix before Phase 2")


if __name__ == "__main__":
    main()
