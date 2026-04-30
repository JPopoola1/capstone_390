"""
EDITABLE -- The agent modifies this file.

Define the model pipeline for xG / xGA regression.
The function build_model() must return an sklearn-compatible estimator.
"""

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
        ("model", Ridge(alpha=350.0)),
    ])
