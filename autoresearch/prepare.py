"""
This file defines data preparation utilities.
It does NOT run experiments or produce outputs when executed directly.
"""


from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


SPLIT_MAP = {
    2021: "train",
    2022: "train",
    2023: "train",
    2024: "validation",
    2025: "test",
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
        raise ValueError(f"Unmapped seasons: {unknown}")

    frame = frame.sort_values(
        ["date", "team", "opponent", "venue"]
    ).reset_index(drop=True)

    frame["row_id"] = np.arange(len(frame), dtype=int)

    frame["fixture_id"] = (
        frame["date"].dt.strftime("%Y-%m-%d")
        + "__"
        + frame["home_team"]
        + "__vs__"
        + frame["away_team"]
    )

    return frame


def build_design_matrix(frame: pd.DataFrame, include_travel: bool) -> pd.DataFrame:
    numeric_columns = ["is_home"]
    if include_travel:
        numeric_columns.append("travel_100_miles")

    numeric = frame[numeric_columns].astype(float)

    team_dummies = pd.get_dummies(frame["team"], prefix="team", dtype=float)
    opponent_dummies = pd.get_dummies(frame["opponent"], prefix="opp", dtype=float)
    season_dummies = pd.get_dummies(frame["season"].astype(str), prefix="season", dtype=float)

    design = pd.concat(
        [numeric, team_dummies, opponent_dummies, season_dummies], axis=1
    )

    design.insert(0, "intercept", 1.0)

    return design


def align_columns(
    train_frame: pd.DataFrame, other_frames: Iterable[pd.DataFrame]
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    train_columns = list(train_frame.columns)
    aligned = [
        other.reindex(columns=train_columns, fill_value=0.0)
        for other in other_frames
    ]
    return train_frame[train_columns], aligned

if __name__ == "__main__":
    input_path = Path("data/raw/final_matches_distance.csv")
    output_path = Path("data/processed.csv")

    df = load_data(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved processed data to {output_path}")