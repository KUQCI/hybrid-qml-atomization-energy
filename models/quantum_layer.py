from torch import nn
import pennylane as qml


def build_quantum_layer(
    n_qubits: int = 4,
    depth: int = 2,
    device_name: str = "default.qubit",
) -> nn.Module:
    """Create a PennyLane quantum layer compatible with torch.nn.Sequential."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")
    if depth < 1:
        raise ValueError("depth must be at least 1")

    wires = range(n_qubits)
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=wires)
        qml.BasicEntanglerLayers(weights, wires=wires)
        return [qml.expval(qml.PauliZ(i)) for i in wires]

    weight_shapes = {"weights": (depth, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)
