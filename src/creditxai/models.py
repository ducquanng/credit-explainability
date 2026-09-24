"""The black box being explained, and a transparent model to compare it against."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED


def black_box():
    """Multi-layer perceptron: one hidden layer of 50 units, as in the original study."""
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(50,), activation="relu", solver="adam",
                      max_iter=800, random_state=SEED),
    )


def glass_box():
    """Logistic regression: its coefficients are the explanation, no surrogate needed."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
