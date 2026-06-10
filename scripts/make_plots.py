"""
Produce the Phase 5 visualizations as PNGs in results/:

  1. mae_comparison.png  — test MAE bar chart, all models vs chemical accuracy
  2. pred_vs_actual.png  — predicted vs true energies for the best hybrid
  3. loss_curve.png      — training curve of the best hybrid run
  4. circuit_diagram.png — the variational circuit, drawn with qml.draw_mpl

Run from repo root after training: python scripts/make_plots.py
"""
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.features import load_features
from models.hybrid_model import build_model

RESULTS = ROOT / "results"
BEST = {"n_qubits": 6, "depth": 3, "run_name": "hybrid_6q_d3_e400"}
CHEMICAL_ACCURACY = 1.0


def latest_test_mae(model_name: str) -> float:
    """Most recent non-nan test MAE logged for a model in experiments.csv."""
    value = None
    with open(RESULTS / "experiments.csv") as f:
        for row in csv.DictReader(f):
            if row["model_name"] == model_name and row["test_mae"] not in ("", "nan"):
                value = float(row["test_mae"])
    if value is None:
        raise ValueError(f"no test MAE logged for {model_name}")
    return value


def best_checkpoint() -> Path:
    matches = sorted(RESULTS.glob(f"checkpoints/{BEST['run_name']}_valmae*.pt"))
    if not matches:
        raise FileNotFoundError(f"no checkpoint for {BEST['run_name']}")
    return matches[0]


def plot_mae_comparison() -> None:
    models = [
        ("KRR\n(Laplacian)", latest_test_mae("krr_laplacian"), "#4878cf"),
        ("Hybrid quantum\n(6q, depth 3)", latest_test_mae("hybrid_6q_d3_e400_TEST"), "#9467bd"),
        ("Classical\nablation", latest_test_mae("ablation_6q_d3"), "#7f7f7f"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([m[0] for m in models], [m[1] for m in models], color=[m[2] for m in models])
    ax.bar_label(bars, fmt="%.2f")
    ax.axhline(CHEMICAL_ACCURACY, color="crimson", linestyle="--", linewidth=1.5)
    ax.text(2.4, CHEMICAL_ACCURACY + 0.3, "chemical accuracy (1.0)", color="crimson", ha="right", fontsize=9)
    ax.set_ylabel("Test MAE (kcal/mol)")
    ax.set_title("QM7 atomization energy — test MAE by model (eigenvalue features)")
    fig.tight_layout()
    fig.savefig(RESULTS / "mae_comparison.png", dpi=200)
    print("saved mae_comparison.png")


def plot_pred_vs_actual() -> None:
    (_, _, _, _, X_test, y_test), meta = load_features()
    model = build_model(n_qubits=BEST["n_qubits"], depth=BEST["depth"])
    model.load_state_dict(torch.load(best_checkpoint(), weights_only=True))
    model.eval()
    with torch.no_grad():
        pred = model(X_test).numpy().ravel()
    to_kcal = lambda a: a * meta["y_std"] + meta["y_mean"]
    y_true, y_pred = to_kcal(y_test.numpy().ravel()), to_kcal(pred)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.4, color="#9467bd", edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("True atomization energy (kcal/mol)")
    ax.set_ylabel("Predicted (kcal/mol)")
    mae = np.abs(y_true - y_pred).mean()
    ax.set_title(f"Best hybrid model — test set (MAE {mae:.2f} kcal/mol)")
    fig.tight_layout()
    fig.savefig(RESULTS / "pred_vs_actual.png", dpi=200)
    print("saved pred_vs_actual.png")


def plot_loss_curve() -> None:
    hist = np.genfromtxt(RESULTS / "history" / f"{BEST['run_name']}.csv", delimiter=",", names=True)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(hist["epoch"], hist["train_mse"], color="#4878cf", label="train MSE (normalized)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train MSE (normalized)", color="#4878cf")
    ax2 = ax1.twinx()
    ax2.plot(hist["epoch"], hist["val_mae_kcal"], color="#d65f5f", label="val MAE (kcal/mol)")
    ax2.set_ylabel("Val MAE (kcal/mol)", color="#d65f5f")
    ax1.set_title(f"Training curve — {BEST['run_name']}")
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_curve.png", dpi=200)
    print("saved loss_curve.png")


def plot_circuit() -> None:
    import pennylane as qml

    n, depth = BEST["n_qubits"], BEST["depth"]
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n))
        qml.BasicEntanglerLayers(weights, wires=range(n))
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    fig, _ = qml.draw_mpl(circuit)(np.zeros(n), np.zeros((depth, n)))
    fig.suptitle(f"Variational circuit: AngleEmbedding + {depth}x BasicEntanglerLayers, {n} qubits")
    fig.savefig(RESULTS / "circuit_diagram.png", dpi=200, bbox_inches="tight")
    print("saved circuit_diagram.png")


if __name__ == "__main__":
    plot_mae_comparison()
    plot_pred_vs_actual()
    plot_loss_curve()
    plot_circuit()
