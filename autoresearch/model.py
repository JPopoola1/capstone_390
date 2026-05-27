"""
EDITABLE -- The agent modifies this file.

Define the model pipeline for xG / xGA regression.
The function build_model() must return an sklearn-compatible estimator.
"""

import sys

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler


class InteractionFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, mode="none"):
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.mode == "none":
            return X

        frame = X.copy() if hasattr(X, "copy") else pd.DataFrame(X)
        columns = list(frame.columns)
        team_columns = [column for column in columns if column.startswith("team_")]
        opp_columns = [column for column in columns if column.startswith("opp_")]
        season_columns = [column for column in columns if column.startswith("season_")]
        modes = set(self.mode.split("+"))
        additions = []

        def add_scaled(prefix, base_column, selected_columns):
            if base_column not in frame:
                return
            block = frame[selected_columns].multiply(frame[base_column], axis=0)
            block.columns = [f"{prefix}_{column}" for column in selected_columns]
            additions.append(block)

        def add_cross(prefix, left_columns, right_columns):
            block = {}
            for left in left_columns:
                left_values = frame[left]
                for right in right_columns:
                    block[f"{prefix}_{left}__{right}"] = left_values * frame[right]
            additions.append(pd.DataFrame(block, index=frame.index))

        if "home_team" in modes:
            add_scaled("home", "is_home", team_columns)
        if "home_opp" in modes:
            add_scaled("home", "is_home", opp_columns)
        if "travel_team" in modes:
            add_scaled("travel", "travel_100_miles", team_columns)
        if "travel_opp" in modes:
            add_scaled("travel", "travel_100_miles", opp_columns)
        if "team_season" in modes:
            add_cross("team_season", team_columns, season_columns)
        if "opp_season" in modes:
            add_cross("opp_season", opp_columns, season_columns)
        if "team_opp" in modes:
            add_cross("team_opp", team_columns, opp_columns)

        if not additions:
            return frame

        return pd.concat([frame, *additions], axis=1)


def _read_experiment_config(defaults):
    """Allow run descriptions like 'tw 067 leaf 18 mf 075 depth 12' to tune safely."""
    config = defaults.copy()
    tokens = " ".join(sys.argv[1:]).split()
    pairs = {
        "tw": ("tree_weight", lambda value: float(value) / 100.0),
        "leaf": ("min_samples_leaf", int),
        "mf": ("max_features", lambda value: float(value) / 100.0),
        "depth": ("max_depth", lambda value: None if value == "none" else int(value)),
        "n": ("n_estimators", int),
        "alpha": ("ridge_alpha", float),
        "seed": ("random_state", int),
        "inter": ("interactions", str),
    }

    for index, token in enumerate(tokens[:-1]):
        if token in pairs:
            key, parser = pairs[token]
            config[key] = parser(tokens[index + 1])

    return config


def build_model():
    """
    Return an sklearn Pipeline.
    The agent is expected to modify:
    - model choice
    - hyperparameters
    - preprocessing steps
    """

    config = _read_experiment_config({
        "ridge_alpha": 0.1,
        "tree_weight": 0.78,
        "n_estimators": 400,
        "max_depth": 15,
        "min_samples_leaf": 17,
        "max_features": 0.75,
        "random_state": 42,
        "interactions": "none",
    })

    ridge_alpha = config["ridge_alpha"]
    tree_weight = config["tree_weight"]
    ridge_weight = 1.0 - tree_weight
    n_estimators = config["n_estimators"]
    max_depth = config["max_depth"]
    min_samples_leaf = config["min_samples_leaf"]
    max_features = config["max_features"]
    random_state = config["random_state"]
    interactions = config["interactions"]

    return Pipeline([
        ("interactions", InteractionFeatures(mode=interactions)),
        ("scaler", MaxAbsScaler()),
        ("model", VotingRegressor([
            ("ridge", Ridge(alpha=ridge_alpha)),
            ("extra_trees", ExtraTreesRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state,
                n_jobs=1,
            )),
        ], weights=[ridge_weight, tree_weight])),
    ])
