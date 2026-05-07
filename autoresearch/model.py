"""
EDITABLE -- The agent modifies this file.

Define the model pipeline for xG / xGA regression.
The function build_model() must return an sklearn-compatible estimator.
"""

from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model():
    """
    Return an sklearn Pipeline.
    The agent is expected to modify:
    - model choice
    - hyperparameters
    - preprocessing steps
    """

    return Pipeline([
        # Optional scaling (useful for linear models)
        ("scaler", StandardScaler()),
        ("model", VotingRegressor([
            ("ridge", Ridge(alpha=0.1)),
            ("extra_trees", ExtraTreesRegressor(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=15,
                max_features=0.7,
                random_state=42,
                n_jobs=1,
            )),
        ], weights=[0.5, 0.5])),
    ])
