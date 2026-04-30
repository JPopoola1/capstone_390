"""
EDITABLE -- The agent modifies this file.

Define the model pipeline for xG / xGA regression.
The function build_model() must return an sklearn-compatible estimator.
"""

from sklearn.ensemble import HistGradientBoostingRegressor
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

        # Default model (agent can swap this)
        ("model", HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=30,
            l2_regularization=0.1,
            random_state=42,
        )),
    ])