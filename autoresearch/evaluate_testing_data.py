from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_PATH = BASE_DIR / "data" / "processed.csv"
MANIFEST_PATH = ROOT_DIR / "data" / "splits" / "split_manifest.csv"
TSV_PATH = BASE_DIR / "testing_data_final_results.tsv"
MD_PATH = BASE_DIR / "testing_data_final_results.md"
PRIMARY_TARGET = "xg"
MODEL_DESCRIPTION = (
    "batch5 tw 78 leaf 17 mf 075 depth 15 n 400 alpha 0.1 seed 42 inter none"
)


def evaluate(model, X, y):
    preds = model.predict(X)
    residual = y - preds
    rmse = float(np.sqrt(np.mean(residual**2)))
    denom = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1.0 - np.sum(residual**2) / denom) if denom > 0 else 0.0
    return rmse, r2


def to_markdown_table(results: pd.DataFrame) -> str:
    headers = list(results.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in results.iterrows():
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    sys.path.insert(0, str(BASE_DIR))
    from model import build_model
    from prepare import align_columns, build_design_matrix, load_data

    frame = load_data(DATA_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest["date"] = pd.to_datetime(manifest["date"], format="%Y-%m-%d")

    test_row_ids = set(manifest.loc[manifest["split"] == "test", "row_id"])
    after_cutoff = manifest["date"] > pd.Timestamp("2024-05-19")
    if not manifest.loc[manifest["row_id"].isin(test_row_ids), "season"].eq(2025).all():
        raise ValueError("Testing split contains non-2025 rows.")
    if not after_cutoff.loc[manifest["row_id"].isin(test_row_ids)].all():
        raise ValueError("Testing split contains rows on or before 2024-05-19.")

    train = frame["split"] == "train"
    test = frame["row_id"].isin(test_row_ids)

    X_train = build_design_matrix(frame.loc[train], include_travel=True)
    X_test = build_design_matrix(frame.loc[test], include_travel=True)
    X_train, [X_test] = align_columns(X_train, [X_test])

    y_train = frame.loc[train, PRIMARY_TARGET].to_numpy()
    y_test = frame.loc[test, PRIMARY_TARGET].to_numpy()

    rows = []
    for run in range(1, 6):
        sys.argv = ["evaluate_testing_data.py", *MODEL_DESCRIPTION.split()]
        model = build_model()
        started = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - started
        rmse, r2 = evaluate(model, X_test, y_test)
        rows.append(
            {
                "run": run,
                "model_description": MODEL_DESCRIPTION,
                "train_rows": int(train.sum()),
                "test_rows": int(test.sum()),
                "test_start_date": manifest.loc[
                    manifest["row_id"].isin(test_row_ids), "date"
                ].min().strftime("%Y-%m-%d"),
                "test_end_date": manifest.loc[
                    manifest["row_id"].isin(test_row_ids), "date"
                ].max().strftime("%Y-%m-%d"),
                "rmse": rmse,
                "r2": r2,
                "train_time_seconds": train_time,
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(TSV_PATH, sep="\t", index=False, float_format="%.6f")

    markdown = [
        "# Testing Data Final Results",
        "",
        f"Locked model: `{MODEL_DESCRIPTION}`",
        "",
        "Testing set: 760 rows, season 2025, dated 2024-08-16 through 2025-05-25.",
        "",
        to_markdown_table(results),
        "",
    ]
    MD_PATH.write_text("\n".join(markdown), encoding="utf-8")
    print(results.to_string(index=False))
    print(f"Saved {TSV_PATH}")
    print(f"Saved {MD_PATH}")


if __name__ == "__main__":
    main()
