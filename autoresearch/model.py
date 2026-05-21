"""
EDITABLE -- The agent modifies this file.

Define the model pipeline for xG / xGA regression.
The function build_model() must return an sklearn-compatible estimator.
"""

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.pipeline import Pipeline


def build_model():
    """
    Return an sklearn Pipeline.
    The agent is expected to modify:
    - model choice
    - hyperparameters
    - preprocessing steps
    """

    return Pipeline([
        ("model", VotingRegressor([
            ("extra_trees", ExtraTreesRegressor(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=15,
                max_features=0.7,
                random_state=42,
                n_jobs=1,
            )),
            ("hist_gb", HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.035,
                max_iter=220,
                max_leaf_nodes=8,
                min_samples_leaf=35,
                l2_regularization=0.35,
                random_state=42,
            )),
            ("random_forest", RandomForestRegressor(
                n_estimators=450,
                max_depth=9,
                min_samples_leaf=18,
                min_samples_split=45,
                max_features=0.5,
                random_state=42,
                n_jobs=1,
            )),
        ], weights=[0.55, 0.30, 0.15])),
    ])
