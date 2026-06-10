"""
Kernel Ridge Regression baseline on the same eigenvalue features and splits
as the hybrid model.

Literature reference: with *eigenvalue* features, QM7 KRR lands around
~10 kcal/mol (Rupp et al. 2012). The often-quoted ~3 kcal/mol figures use
richer representations (full sorted/randomized Coulomb matrices), so ~10
is the honest bar for any model trained on the eigenvalue spectrum.
Verified here: standardized 10.76, raw 10.83 kcal/mol test MAE.

Grid-searches alpha/gamma on the validation split, refits the winner, and
logs the official test MAE to results/experiments.csv.

Run from repo root: python scripts/baseline_krr.py
"""
from pathlib import Path
import sys

import numpy as np
from sklearn.kernel_ridge import KernelRidge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.features import FEATURES_PATH
from utils.logger import ExperimentLogger

ALPHAS = np.logspace(-11, -1, 6)
GAMMAS = np.logspace(-5, 0, 6)


def main() -> None:
    data = np.load(FEATURES_PATH)
    X_train, y_train = data["X_train"], data["y_train"].ravel()
    X_val, y_val = data["X_val"], data["y_val"].ravel()
    X_test, y_test = data["X_test"], data["y_test"].ravel()
    y_std = float(data["y_std"])

    best = {"val_mae": np.inf}
    for alpha in ALPHAS:
        for gamma in GAMMAS:
            model = KernelRidge(kernel="laplacian", alpha=alpha, gamma=gamma)
            model.fit(X_train, y_train)
            val_mae = np.abs(model.predict(X_val) - y_val).mean() * y_std
            if val_mae < best["val_mae"]:
                best = {"val_mae": val_mae, "alpha": alpha, "gamma": gamma, "model": model}
            print(f"alpha={alpha:.0e} gamma={gamma:.0e} -> val MAE {val_mae:8.3f} kcal/mol", flush=True)

    test_mae = np.abs(best["model"].predict(X_test) - y_test).mean() * y_std
    print(f"\nBest: alpha={best['alpha']:.0e} gamma={best['gamma']:.0e}")
    print(f"val MAE {best['val_mae']:.3f} | OFFICIAL test MAE {test_mae:.3f} kcal/mol "
          f"(literature ~10 for eigenvalue features)")

    ExperimentLogger().log(
        config={
            "model_name": "krr_laplacian",
            "notes": f"KRR baseline, alpha={best['alpha']:.0e}, gamma={best['gamma']:.0e}, "
                     f"eigenvalue features, 4560 train",
        },
        val_mae=best["val_mae"],
        test_mae=test_mae,
    )


if __name__ == "__main__":
    main()
