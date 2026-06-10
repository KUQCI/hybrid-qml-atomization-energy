"""
Training loop for the hybrid (or classical-ablation) QM7 model.

Examples (from repo root, inside the venv):
    python scripts/train.py --model-name sanity --n-train 200 --epochs 20
    python scripts/train.py --model-name hybrid_4q_d2 --n-train 1000 --epochs 100
    python scripts/train.py --model-name ablation --classical --n-train 1000 --epochs 100

Every run appends a row to results/experiments.csv via ExperimentLogger.
Checkpoints land in results/checkpoints/ whenever val MAE improves; only
the best 3 checkpoints (across all runs) are kept.
"""
import argparse
import math
import re
import time
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.features import load_features
from models.hybrid_model import build_classical_ablation, build_model
from utils.logger import ExperimentLogger

CHECKPOINT_DIR = ROOT / "results" / "checkpoints"
KEEP_TOP_K = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", required=True, help="run label for logs/checkpoints")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--n-train", type=int, default=0, help="train subsample size (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--classical", action="store_true", help="train the classical ablation instead")
    p.add_argument("--eval-test", action="store_true", help="evaluate best checkpoint on the test set (do once, at phase end)")
    p.add_argument("--notes", default="")
    return p.parse_args()


def mae_kcal(model: nn.Module, X: torch.Tensor, y: torch.Tensor, y_std: float, batch_size: int = 256) -> float:
    """MAE in kcal/mol. Labels are train-normalized, so MAE_norm * y_std."""
    model.eval()
    abs_err_sum = 0.0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            pred = model(X[i : i + batch_size])
            abs_err_sum += (pred - y[i : i + batch_size]).abs().sum().item()
    model.train()
    return abs_err_sum / len(X) * y_std


def prune_checkpoints() -> None:
    """Keep only the KEEP_TOP_K checkpoints with the lowest val MAE
    (parsed from filenames like name_valmae12.3456.pt)."""
    pattern = re.compile(r"_valmae([\d.]+)\.pt$")
    ranked = sorted(
        (f for f in CHECKPOINT_DIR.glob("*.pt") if pattern.search(f.name)),
        key=lambda f: float(pattern.search(f.name).group(1).rstrip(".")),
    )
    for stale in ranked[KEEP_TOP_K:]:
        stale.unlink()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    (X_train, y_train, X_val, y_val, X_test, y_test), meta = load_features()
    y_std = meta["y_std"]

    if args.n_train:
        idx = torch.randperm(len(X_train))[: args.n_train]
        X_train, y_train = X_train[idx], y_train[idx]

    builder = build_classical_ablation if args.classical else build_model
    model = builder(n_qubits=args.n_qubits, depth=args.depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    kind = "classical-ablation" if args.classical else "hybrid"
    print(f"{args.model_name}: {kind}, {len(X_train)} train samples, "
          f"{args.n_qubits} qubits, depth {args.depth}, lr {args.lr}, {args.epochs} epochs")

    best_val_mae = math.inf
    best_path = None
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        for i in range(0, len(X_train), args.batch_size):
            batch = perm[i : i + args.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        epoch_loss /= len(X_train)

        val_mae = mae_kcal(model, X_val, y_val, y_std)
        marker = ""
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            if best_path is not None and best_path.exists():
                best_path.unlink()
            best_path = CHECKPOINT_DIR / f"{args.model_name}_valmae{val_mae:.4f}.pt"
            torch.save(model.state_dict(), best_path)
            marker = "  *best*"
        print(f"epoch {epoch:3d} | train MSE {epoch_loss:.5f} | val MAE {val_mae:8.3f} kcal/mol{marker}", flush=True)

    elapsed = time.time() - start
    prune_checkpoints()

    test_mae = float("nan")
    if args.eval_test and best_path is not None and best_path.exists():
        model.load_state_dict(torch.load(best_path, weights_only=True))
        test_mae = mae_kcal(model, X_test, y_test, y_std)
        print(f"OFFICIAL test MAE: {test_mae:.3f} kcal/mol")

    ExperimentLogger().log(
        config={
            "model_name": args.model_name,
            "n_qubits": args.n_qubits,
            "depth": args.depth,
            "lr": args.lr,
            "epochs": args.epochs,
            "notes": f"{kind}, n_train={len(X_train)}, {elapsed:.0f}s. {args.notes}".strip(),
        },
        val_mae=best_val_mae,
        test_mae=test_mae,
    )
    print(f"done in {elapsed:.0f}s | best val MAE {best_val_mae:.3f} kcal/mol | logged to results/experiments.csv")


if __name__ == "__main__":
    main()
