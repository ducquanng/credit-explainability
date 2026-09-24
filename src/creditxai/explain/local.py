"""A LIME-style local surrogate, written out rather than imported.

The idea: to explain one prediction, sample points around it, ask the black box what
it predicts for each, and fit a weighted linear model on that neighbourhood. The
surrogate's coefficients are the explanation, and its weighted R-squared says how
much the explanation can be trusted.

Implemented directly because the fidelity and stability checks in `quality.py` need
access to the neighbourhood, the weights and the surrogate itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


@dataclass
class LocalExplanation:
    """Feature contributions for one prediction, with the surrogate's own fidelity."""

    instance: pd.Series
    prediction: float
    weights: pd.Series        # signed contribution per interpretable feature
    local_r2: float           # weighted R^2 of the surrogate in this neighbourhood
    n_samples: int

    def top(self, k: int = 10) -> pd.Series:
        return self.weights.reindex(self.weights.abs().sort_values(ascending=False).index).head(k)


def explain_instance(
    model,
    X_background: pd.DataFrame,
    instance: pd.Series,
    groups: dict[str, list[str]] | None = None,
    n_samples: int = 5000,
    keep_probability: float = 0.5,
    kernel_width: float = 0.5,
    seed: int = 42,
    alpha: float = 1.0,
) -> LocalExplanation:
    """Explain one prediction with a weighted ridge fitted on sampled neighbours.

    Each interpretable feature is an original attribute (`groups` maps it to its
    encoded columns). A neighbour is built by keeping some attributes at the
    instance's values and taking the rest wholesale from a real background
    applicant, so every sampled point is a combination that actually occurs.
    """
    rng = np.random.default_rng(seed)
    columns = list(X_background.columns)
    groups = groups or {c: [c] for c in columns}
    names = list(groups)
    positions = {c: j for j, c in enumerate(columns)}
    background = X_background.to_numpy()
    x = instance[columns].to_numpy(dtype=float)

    kept = rng.random((n_samples, len(names))) < keep_probability
    kept[0, :] = True  # the instance itself anchors the neighbourhood
    donors = rng.integers(0, background.shape[0], size=n_samples)

    samples = np.repeat(x[None, :], n_samples, axis=0)
    for gi, name in enumerate(names):
        cols = [positions[c] for c in groups[name]]
        replace = ~kept[:, gi]
        if replace.any():
            samples[np.ix_(replace, cols)] = background[np.ix_(donors[replace], cols)]

    # The interpretable feature is "does this attribute still hold the applicant's
    # value", not "was it resampled". A donor can supply the same value, and counting
    # that as a change would put pure noise into the surrogate's target.
    binary = np.column_stack(
        [
            (samples[:, [positions[c] for c in groups[name]]]
             == x[[positions[c] for c in groups[name]]]).all(axis=1).astype(float)
            for name in names
        ]
    )
    probabilities = model.predict_proba(pd.DataFrame(samples, columns=columns))[:, 1]

    # Distance in the interpretable space: share of attributes taken from the donor.
    distance = 1.0 - binary.mean(axis=1)
    weights = np.exp(-(distance**2) / (kernel_width**2))

    surrogate = Ridge(alpha=alpha)
    surrogate.fit(binary, probabilities, sample_weight=weights)
    fitted = surrogate.predict(binary)

    return LocalExplanation(
        instance=instance[columns],
        prediction=float(model.predict_proba(pd.DataFrame([x], columns=columns))[0, 1]),
        # Sign convention: positive means "holding this attribute at the applicant's
        # value pushes predicted risk up, relative to swapping it for someone else's".
        weights=pd.Series(surrogate.coef_, index=names),
        local_r2=float(r2_score(probabilities, fitted, sample_weight=weights)),
        n_samples=n_samples,
    )
