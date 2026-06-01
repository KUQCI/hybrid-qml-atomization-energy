import os
import pickle
import deepchem as dc
import numpy as np

SAVE_DIR = os.path.join(os.path.dirname(__file__), "splits")
SEED = 42


def load_and_split():
    tasks, datasets, transformers = dc.molnet.load_qm7(
        featurizer="CoulombMatrix",
        split="random",
        move_mean=True,
    )

    train, val, test = datasets

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"Feature shape: {train.X.shape}")

    labels = np.concatenate([train.y, val.y, test.y])
    print(f"Label stats — min: {labels.min():.2f}  max: {labels.max():.2f}  mean: {labels.mean():.2f}  (kcal/mol)")

    os.makedirs(SAVE_DIR, exist_ok=True)
    for name, ds in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(SAVE_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(ds, f)
        print(f"Saved {name} split → {path}")

    return train, val, test, transformers


if __name__ == "__main__":
    load_and_split()
