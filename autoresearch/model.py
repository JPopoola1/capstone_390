from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


SPLIT_MAP = {
    2021: "train",
    2022: "train",
    2023: "train",
    2024: "validation",
    2025: "test",
}

PRIMARY_TARGET = "xg"
SECONDARY_TARGET = "xga"
PRIMARY_METRIC = "rmse"
BOOTSTRAP_SEED = 7
PERMUTATION_DRAWS = 5000
RUNTIME_BUDGET_SECONDS = 60.0


@dataclass
class LinearModel:
    feature_names: List[str]
    coefficients: np.ndarray

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame[self.feature_names].to_numpy(dtype=float)
        return matrix @ self.coefficients

    def coefficient_map(self) -> Dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(self.feature_names, self.coefficients, strict=True)
        }


def load_data(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d")
    frame["season"] = frame["season"].astype(int)
    frame["xg"] = frame["xg"].astype(float)
    frame["xga"] = frame["xga"].astype(float)
    frame["away_travel_miles"] = frame["away_travel_miles"].astype(float)
    frame["is_home"] = (frame["venue"] == "Home").astype(int)
    frame["team_travel_miles"] = np.where(
        frame["venue"] == "Away", frame["away_travel_miles"], 0.0
    )
    frame["travel_100_miles"] = frame["team_travel_miles"] / 100.0
    frame["split"] = frame["season"].map(SPLIT_MAP)

    if frame["split"].isna().any():
        unknown = sorted(frame.loc[frame["split"].isna(), "season"].unique())
        raise ValueError(f"Unmapped seasons in split definition: {unknown}")

    frame = frame.sort_values(["date", "team", "opponent", "venue"]).reset_index(drop=True)
    frame["row_id"] = np.arange(len(frame), dtype=int)
    frame["fixture_id"] = (
        frame["date"].dt.strftime("%Y-%m-%d")
        + "__"
        + frame["home_team"]
        + "__vs__"
        + frame["away_team"]
    )
    return frame


def save_split_manifest(frame: pd.DataFrame, output_path: Path) -> None:
    manifest = frame[
        [
            "row_id",
            "fixture_id",
            "date",
            "season",
            "team",
            "opponent",
            "venue",
            "split",
        ]
    ].copy()
    manifest.to_csv(output_path, index=False)


def build_design_matrix(frame: pd.DataFrame, include_travel: bool) -> pd.DataFrame:
    numeric_columns = ["is_home"]
    if include_travel:
        numeric_columns.append("travel_100_miles")

    numeric = frame[numeric_columns].astype(float)
    team_dummies = pd.get_dummies(frame["team"], prefix="team", dtype=float)
    opponent_dummies = pd.get_dummies(frame["opponent"], prefix="opp", dtype=float)
    season_dummies = pd.get_dummies(frame["season"].astype(str), prefix="season", dtype=float)

    design = pd.concat([numeric, team_dummies, opponent_dummies, season_dummies], axis=1)
    design.insert(0, "intercept", 1.0)
    return design


def align_columns(
    train_frame: pd.DataFrame, other_frames: Iterable[pd.DataFrame]
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    train_columns = list(train_frame.columns)
    aligned = [other.reindex(columns=train_columns, fill_value=0.0) for other in other_frames]
    return train_frame[train_columns], aligned


def fit_linear_regression(features: pd.DataFrame, target: pd.Series) -> LinearModel:
    x = features.to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    return LinearModel(feature_names=list(features.columns), coefficients=coefficients)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    residual = y_true - y_pred
    mse = float(np.mean(np.square(residual)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    denom = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = float(1.0 - (np.sum(np.square(residual)) / denom)) if denom > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "mse": mse, "r2": r2}


def permutation_test_improvement(
    loss_baseline: np.ndarray, loss_travel: np.ndarray, seed: int, draws: int
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    diff = loss_baseline - loss_travel
    observed = float(np.mean(diff))

    null_samples = np.empty(draws, dtype=float)
    bootstrap_means = np.empty(draws, dtype=float)
    for idx in range(draws):
        signs = rng.choice(np.array([-1.0, 1.0]), size=diff.shape[0])
        null_samples[idx] = float(np.mean(diff * signs))
        sample_idx = rng.integers(0, diff.shape[0], size=diff.shape[0])
        bootstrap_means[idx] = float(np.mean(diff[sample_idx]))

    exceedances = int(np.sum(np.abs(null_samples) >= abs(observed)))
    two_sided_p = float((exceedances + 1) / (draws + 1))
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    direction = "improves" if observed > 0 else "worsens" if observed < 0 else "no_change"
    return {
        "mean_loss_improvement": observed,
        "direction": direction,
        "ci_95_low": float(lower),
        "ci_95_high": float(upper),
        "p_value_two_sided": two_sided_p,
        "permutation_draws": int(draws),
    }


def average_away_distance(frame: pd.DataFrame) -> float:
    away_only = frame.loc[frame["venue"] == "Away", "team_travel_miles"]
    return float(away_only.mean())


def build_summary(
    model: LinearModel, target_name: str, evaluation_frame: pd.DataFrame
) -> Dict[str, float]:
    coefficients = model.coefficient_map()
    summary = {
        "target": target_name,
        "home_effect_xg_units": coefficients.get("is_home", 0.0),
    }
    if "travel_100_miles" in coefficients:
        summary["travel_effect_per_100_miles_xg_units"] = coefficients["travel_100_miles"]
        summary["average_away_distance_miles"] = average_away_distance(evaluation_frame)
        summary["travel_effect_at_average_away_trip_xg_units"] = (
            coefficients["travel_100_miles"] * summary["average_away_distance_miles"] / 100.0
        )
    return summary


def run_experiment(data_path: Path, output_dir: Path) -> Dict[str, object]:
    start = time.perf_counter()
    frame = load_data(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path = output_dir.parent / "data" / "splits" / "split_manifest.csv"
    save_split_manifest(frame, split_path)

    train_rows = frame["split"] == "train"
    validation_rows = frame["split"] == "validation"

    results: Dict[str, object] = {
        "data_path": str(data_path),
        "split_counts": frame["split"].value_counts().sort_index().to_dict(),
        "primary_target": PRIMARY_TARGET,
        "primary_metric": PRIMARY_METRIC,
        "runtime_budget_seconds": RUNTIME_BUDGET_SECONDS,
        "models": {},
        "targets": {},
        "test_set_policy": (
            "Season 2025 is locked as the final test set and is not used for fitting "
            "or model selection in this baseline run."
        ),
    }

    train_base = build_design_matrix(frame.loc[train_rows], include_travel=False)
    validation_base = build_design_matrix(frame.loc[validation_rows], include_travel=False)
    train_base, [validation_base] = align_columns(train_base, [validation_base])

    train_travel = build_design_matrix(frame.loc[train_rows], include_travel=True)
    validation_travel = build_design_matrix(frame.loc[validation_rows], include_travel=True)
    train_travel, [validation_travel] = align_columns(train_travel, [validation_travel])

    for target_name in [PRIMARY_TARGET, SECONDARY_TARGET]:
        y_train = frame.loc[train_rows, target_name]
        y_validation = frame.loc[validation_rows, target_name].to_numpy(dtype=float)

        baseline_model = fit_linear_regression(train_base, y_train)
        travel_model = fit_linear_regression(train_travel, y_train)

        baseline_pred = baseline_model.predict(validation_base)
        travel_pred = travel_model.predict(validation_travel)

        baseline_metrics = regression_metrics(y_validation, baseline_pred)
        travel_metrics = regression_metrics(y_validation, travel_pred)

        significance = permutation_test_improvement(
            np.square(y_validation - baseline_pred),
            np.square(y_validation - travel_pred),
            seed=BOOTSTRAP_SEED,
            draws=PERMUTATION_DRAWS,
        )

        results["targets"][target_name] = {
            "baseline_home_away": {
                "validation_metrics": baseline_metrics,
                "coefficient_summary": build_summary(
                    baseline_model, target_name, frame.loc[validation_rows]
                ),
            },
            "home_away_plus_travel": {
                "validation_metrics": travel_metrics,
                "coefficient_summary": build_summary(
                    travel_model, target_name, frame.loc[validation_rows]
                ),
            },
            "travel_vs_baseline_significance": significance,
        }

    runtime_seconds = float(time.perf_counter() - start)
    results["runtime_seconds"] = runtime_seconds
    results["within_runtime_budget"] = runtime_seconds <= RUNTIME_BUDGET_SECONDS

    return results


def write_results(results: Dict[str, object], output_dir: Path) -> None:
    metrics_path = output_dir / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    budget_path = output_dir / "runtime_budget.json"
    budget_payload = {
        "runtime_seconds": results["runtime_seconds"],
        "runtime_budget_seconds": results["runtime_budget_seconds"],
        "within_runtime_budget": results["within_runtime_budget"],
    }
    budget_path.write_text(json.dumps(budget_payload, indent=2), encoding="utf-8")


def format_experiment_log(results: Dict[str, object]) -> str:
    xg = results["targets"]["xg"]
    base_rmse = xg["baseline_home_away"]["validation_metrics"]["rmse"]
    travel_rmse = xg["home_away_plus_travel"]["validation_metrics"]["rmse"]
    p_value = xg["travel_vs_baseline_significance"]["p_value_two_sided"]
    direction = xg["travel_vs_baseline_significance"]["direction"]
    home_effect = xg["home_away_plus_travel"]["coefficient_summary"]["home_effect_xg_units"]
    travel_effect = xg["home_away_plus_travel"]["coefficient_summary"][
        "travel_effect_per_100_miles_xg_units"
    ]
    return "\n".join(
        [
            "# Experiment Log",
            "",
            "## 2026-04-22 - Baseline v1",
            "",
            "- Objective: predict team xG and xGA without using the locked 2025 test set.",
            "- Split: train=2021-2023, validation=2024, test=2025 locked.",
            f"- Primary metric: validation RMSE on xG.",
            f"- Baseline RMSE (home/away only plus team/opponent/season fixed effects): {base_rmse:.4f}.",
            f"- Travel RMSE (baseline + away travel distance): {travel_rmse:.4f}.",
            f"- Travel feature result on xG: {direction}.",
            f"- Paired permutation p-value for the xG loss difference: {p_value:.4f}.",
            f"- Estimated home effect on xG: {home_effect:.4f}.",
            f"- Estimated travel effect on xG per 100 away miles: {travel_effect:.4f}.",
            f"- Runtime: {results['runtime_seconds']:.2f} seconds.",
        ]
    )
