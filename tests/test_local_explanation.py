import numpy as np
import pandas as pd
import pytest

from creditxai.explain.local import explain_instance
from creditxai.explain.quality import jaccard, stability_jaccard, top_features


class OneFeatureModel:
    """Predicted risk depends on `duration` alone, so the explanation has a known answer."""

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(X["duration"].to_numpy() - 20) / 5))
        return np.column_stack([1 - p, p])


class BinaryFeatureModel:
    """Risk depends only on a binary attribute, which the on/off representation captures exactly."""

    def predict_proba(self, X):
        p = np.where(X["status_a"].to_numpy() == 1, 0.9, 0.1)
        return np.column_stack([1 - p, p])


class NoiseModel:
    def predict_proba(self, X):
        p = np.full(len(X), 0.5)
        return np.column_stack([1 - p, p])


@pytest.fixture
def background():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "duration": rng.integers(4, 48, 300).astype(float),
            "amount": rng.integers(500, 9000, 300).astype(float),
            "status_a": rng.integers(0, 2, 300).astype(float),
        }
    ).assign(status_b=lambda d: 1 - d["status_a"])


GROUPS = {"duration": ["duration"], "amount": ["amount"], "status": ["status_a", "status_b"]}


def test_explanation_finds_the_feature_the_model_actually_uses(background):
    instance = pd.Series({"duration": 44.0, "amount": 5000.0, "status_a": 1.0, "status_b": 0.0})
    ex = explain_instance(OneFeatureModel(), background, instance, groups=GROUPS, n_samples=1500, seed=3)
    assert top_features(ex.weights, 1) == ["duration"]
    assert ex.weights["duration"] > 0  # long duration raises risk, and the sign says so


def test_fidelity_is_high_when_the_representation_can_express_the_model(background):
    """Binary driver: 'was this attribute kept' says everything, so the surrogate fits."""
    instance = pd.Series({"duration": 30.0, "amount": 5000.0, "status_a": 1.0, "status_b": 0.0})
    ex = explain_instance(BinaryFeatureModel(), background, instance, groups=GROUPS, n_samples=1500, seed=3)
    assert ex.local_r2 > 0.9
    assert top_features(ex.weights, 1) == ["status"]


def test_fidelity_is_capped_when_the_swapped_value_matters(background):
    """Continuous driver: the indicator cannot say which duration replaced it, so R² drops.

    This is the ceiling the report on the real data runs into, reproduced in miniature.
    """
    instance = pd.Series({"duration": 44.0, "amount": 5000.0, "status_a": 1.0, "status_b": 0.0})
    ex = explain_instance(OneFeatureModel(), background, instance, groups=GROUPS, n_samples=1500, seed=3)
    assert ex.local_r2 < 0.9


def test_constant_model_yields_no_meaningful_weights(background):
    instance = background.iloc[0]
    ex = explain_instance(NoiseModel(), background, instance, groups=GROUPS, n_samples=800, seed=3)
    assert float(ex.weights.abs().max()) < 1e-6


def test_perturbations_keep_one_hot_groups_consistent(background, monkeypatch):
    """A sampled neighbour must never hold two mutually exclusive statuses at once."""
    seen = []

    class Recorder(OneFeatureModel):
        def predict_proba(self, X):
            seen.append(X.copy())
            return super().predict_proba(X)

    instance = pd.Series({"duration": 30.0, "amount": 3000.0, "status_a": 1.0, "status_b": 0.0})
    explain_instance(Recorder(), background, instance, groups=GROUPS, n_samples=500, seed=5)
    sampled = seen[0]
    assert ((sampled["status_a"] + sampled["status_b"]) == 1).all()


def test_stability_is_perfect_for_a_deterministic_single_feature_model(background):
    instance = pd.Series({"duration": 44.0, "amount": 5000.0, "status_a": 1.0, "status_b": 0.0})
    result = stability_jaccard(OneFeatureModel(), background, instance, groups=GROUPS, k=1, n_samples=800)
    assert result["mean_jaccard"] == 1.0


def test_jaccard():
    assert jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert jaccard(["a"], ["b"]) == 0.0
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
