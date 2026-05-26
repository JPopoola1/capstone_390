"""
EDITABLE -- The agent modifies this file.

Define the model pipeline for xG / xGA regression.
The function build_model() must return an sklearn-compatible estimator.
"""

import sys

from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
        "tree_weight": 0.66,
        "n_estimators": 400,
        "max_depth": 15,
        "min_samples_leaf": 18,
        "max_features": 0.75,
        "random_state": 42,
    })

    ridge_alpha = config["ridge_alpha"]
    tree_weight = config["tree_weight"]
    ridge_weight = 1.0 - tree_weight
    n_estimators = config["n_estimators"]
    max_depth = config["max_depth"]
    min_samples_leaf = config["min_samples_leaf"]
    max_features = config["max_features"]
    random_state = config["random_state"]

    return Pipeline([
        ("scaler", StandardScaler()),
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
