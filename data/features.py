"""
Coulomb matrix -> eigenvalue features for QM7.

Converts the pickled DeepChem splits (23x23 Coulomb matrices) into the
23-dim sorted-eigenvalue vectors the hybrid model consumes, standardized
with train-set statistics only. Results are cached in data/splits/features.npz
so training and baselines never touch DeepChem again.

Run once after data/load_qm7.py:
    python data/features.py
"""
import os
import pickle

import numpy as np

SPLITS_DIR = os.path.join(os.path.dirname(__file__), "splits")
FEATURES_PATH = os.path.join(SPLITS_DIR, "features.npz")
EPS = 1e-8


def coulomb_eigenvalues(X: np.ndarray) -> np.ndarray:
    """(N, 23, 23) symmetric Coulomb matrices -> (N, 23) eigenvalues,
    sorted by descending absolute value (largest spectral content first)."""
    eigvals = np.linalg.eigvalsh(X)
    order = np.argsort(-np.abs(eigvals), axis=1)
    return np.take_along_axis(eigvals, order, axis=1)


def build_features() -> None:
    """Compute eigenvalue features for all splits and cache them as npz."""
    splits = {}
    for name in ("train", "val", "test"):
        with open(os.path.join(SPLITS_DIR, f"{name}.pkl"), "rb") as f:
            ds = pickle.load(f)
        splits[name] = (coulomb_eigenvalues(np.asarray(ds.X)), np.asarray(ds.y, dtype=np.float64))

    with open(os.path.join(SPLITS_DIR, "y_transformer.pkl"), "rb") as f:
        transformer = pickle.load(f)
    y_mean = float(np.asarray(transformer.y_means).ravel()[0])
    y_std = float(np.asarray(transformer.y_stds).ravel()[0])

    # Standardize features on train statistics only (same no-leakage rule
    # as the labels). Zero-variance dims (padding eigenvalues) divide by 1.
    x_mean = splits["train"][0].mean(axis=0)
    x_std = splits["train"][0].std(axis=0)
    x_std = np.where(x_std < EPS, 1.0, x_std)

    arrays = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    for name, (X, y) in splits.items():
        arrays[f"X_{name}"] = (X - x_mean) / x_std
        arrays[f"y_{name}"] = y.reshape(-1, 1)

    np.savez(FEATURES_PATH, **arrays)
    for name in ("train", "val", "test"):
        print(f"{name}: X {arrays[f'X_{name}'].shape}  y {arrays[f'y_{name}'].shape}")
    print(f"y_std = {y_std:.2f} kcal/mol (multiply normalized MAE by this)")
    print(f"Saved features → {FEATURES_PATH}")


def load_features():
    """Return standardized features and normalized labels as float32 torch
    tensors, plus a meta dict with y_mean/y_std for kcal/mol conversion.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test), meta
    """
    import torch

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found — run `python data/features.py` first"
        )
    data = np.load(FEATURES_PATH)
    tensors = tuple(
        torch.tensor(data[key], dtype=torch.float32)
        for key in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")
    )
    meta = {"y_mean": float(data["y_mean"]), "y_std": float(data["y_std"])}
    return tensors, meta


if __name__ == "__main__":
    build_features()
