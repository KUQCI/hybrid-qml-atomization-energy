import csv
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LOG_FILE = os.path.join(RESULTS_DIR, "experiments.csv")

COLUMNS = [
    "timestamp", "model_name", "val_mae", "test_mae",
    "n_qubits", "depth", "lr", "epochs", "notes"
]


class ExperimentLogger:
    def __init__(self, log_file: str = LOG_FILE):
        self.log_file = os.path.abspath(log_file)
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

    def log(self, config: dict, val_mae: float, test_mae: float):
        row = {col: "" for col in COLUMNS}
        row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row["val_mae"] = round(val_mae, 4)
        row["test_mae"] = round(test_mae, 4)
        for key in ("model_name", "n_qubits", "depth", "lr", "epochs", "notes"):
            if key in config:
                row[key] = config[key]
        with open(self.log_file, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)
        print(f"Logged: {row['model_name']}  val_mae={val_mae:.4f}  test_mae={test_mae:.4f}")


if __name__ == "__main__":
    logger = ExperimentLogger()
    logger.log(
        config={"model_name": "test_run", "n_qubits": 4, "depth": 1, "lr": 1e-3, "epochs": 10, "notes": "smoke test"},
        val_mae=99.0,
        test_mae=99.0,
    )
    print(f"experiments.csv created at: {logger.log_file}")
