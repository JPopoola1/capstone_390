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

    ridge_alpha = 0.1
    tree_weight = 0.52
    ridge_weight = 1.0 - tree_weight
    n_estimators = 400
    max_depth = 12
    min_samples_leaf = 15
    max_features = 0.7
    random_state = 42

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
