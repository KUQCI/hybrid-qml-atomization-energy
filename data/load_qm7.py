import os
import pickle
import deepchem as dc
import numpy as np

SAVE_DIR = os.path.join(os.path.dirname(__file__), "splits")
SEED = 42


def load_and_split():
    # Load the full QM7 dataset unsplit (splitter=None) so we control the
    # split ourselves — molnet's internal splitter is unseeded, which would
    # give every team member a different partition.
    tasks, datasets, _ = dc.molnet.load_qm7(
        featurizer=dc.feat.CoulombMatrix(max_atoms=23),
        splitter=None,
        transformers=[],
    )
    dataset = datasets[0]

    print(f"Total molecules: {len(dataset)}")
    print(f"Feature shape: {dataset.X.shape}")
    print(
        f"Label stats — min: {dataset.y.min():.2f}  max: {dataset.y.max():.2f}  "
        f"mean: {dataset.y.mean():.2f}  (kcal/mol)"
    )

    splitter = dc.splits.RandomSplitter()
    train, val, test = splitter.train_valid_test_split(
        dataset,
        frac_train=0.70,
        frac_valid=0.15,
        frac_test=0.15,
        seed=SEED,
    )
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Normalize labels using train statistics only, to avoid leakage.
    # The transformer is saved so predictions can be mapped back to kcal/mol.
    transformer = dc.trans.NormalizationTransformer(transform_y=True, dataset=train)
    train = transformer.transform(train)
    val = transformer.transform(val)
    test = transformer.transform(test)

    os.makedirs(SAVE_DIR, exist_ok=True)
    for name, obj in [
        ("train", train),
        ("val", val),
        ("test", test),
        ("y_transformer", transformer),
    ]:
        path = os.path.join(SAVE_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"Saved {name} → {path}")

    return train, val, test, transformer


if __name__ == "__main__":
    load_and_split()
