import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import load_data, build_design_matrix, align_columns

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed.csv"
RESULTS_PATH = BASE_DIR / "results.tsv"
PRIMARY_TARGET = "xg"
# RESULTS_PATH = Path("results.tsv")


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def evaluate(model, X, y):
    preds = model.predict(X)
    residual = y - preds

    rmse = float(np.sqrt(np.mean(residual ** 2)))

    denom = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1.0 - np.sum(residual ** 2) / denom) if denom > 0 else 0.0

    return rmse, r2


def log_result(commit, rmse, r2, status, description):
    row = f"{commit}\t{rmse:.6f}\t{r2:.6f}\t{status}\t{description}\n"

    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text("commit\trmse\tr2\tstatus\tdescription\n")

    with open(RESULTS_PATH, "a") as f:
        f.write(row)

def get_best_rmse():
    if not RESULTS_PATH.exists():
        return None

    df = pd.read_csv(RESULTS_PATH, sep="\t")
    if df.empty:
        return None

    return df["rmse"].min()


def main():
    args = sys.argv[1:]
    status = "keep"
    description_parts = []

    for a in args:
        if a == "--baseline":
            status = "baseline"
        elif a == "--discard":
            status = "discard"
        else:
            description_parts.append(a)

    description = " ".join(description_parts) if description_parts else "experiment"

    # 1️⃣ Load data
    data_path = DATA_PATH  # <-- adjust if needed
    frame = load_data(data_path)

    train = frame["split"] == "train"
    val = frame["split"] == "validation"

    # Build features (baseline: include travel)
    X_train = build_design_matrix(frame.loc[train], include_travel=True)
    X_val = build_design_matrix(frame.loc[val], include_travel=True)

    X_train, [X_val] = align_columns(X_train, [X_val])

    y_train = frame.loc[train, PRIMARY_TARGET].to_numpy()
    y_val = frame.loc[val, PRIMARY_TARGET].to_numpy()

    print(f"Data: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_train.shape[1]} features")

    # 2️⃣ Build model (agent edits this)
    from model import build_model
    model = build_model()
    print(f"Model: {model}")

    # 3️⃣ Train
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f}s")

    # 4️⃣ Evaluate
    val_rmse, val_r2 = evaluate(model, X_val, y_val)

    print(f"val_rmse: {val_rmse:.6f}")
    print(f"val_r2:   {val_r2:.6f}")

    # 5️⃣ Log
    best_rmse = get_best_rmse()

    if "--baseline" in args:
        status = "baseline"
    elif "--discard" in args:
        status = "discard"
    else:
        if best_rmse is None:
            status = "baseline"
        elif val_rmse < best_rmse:
            status = "keep"
        else:
            status = "discard"

    # Log
    commit = get_git_hash()
    log_result(commit, val_rmse, val_r2, status, description)
    print(f"Best previous RMSE: {best_rmse}")
    print(f"Result logged to results.tsv (status={status})")


if __name__ == "__main__":
    main()