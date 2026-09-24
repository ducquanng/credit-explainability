"""Checks on the explanations themselves.

An explanation is a model of a model. It can be wrong in three ways worth testing:
it can fail to track the black box near the instance (low fidelity), it can change
when nothing changed but the random seed (low stability), and it can be indifferent
to what the black box actually learned (fails the model randomisation check).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .local import explain_instance


def top_features(weights: pd.Series, k: int = 5) -> list[str]:
    return list(weights.abs().sort_values(ascending=False).head(k).index)


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def stability_jaccard(model, X_background, instance, groups=None, seeds=(1, 2, 3, 4, 5), k: int = 5,
                      n_samples: int = 2000) -> dict[str, float]:
    """Re-explain the same prediction under different seeds and compare the top-k sets.

    Reported as the mean pairwise Jaccard overlap. An explanation that reshuffles its
    own top five between runs is not something to show a credit committee.
    """
    tops = [
        top_features(
            explain_instance(model, X_background, instance, groups=groups,
                             n_samples=n_samples, seed=s).weights,
            k,
        )
        for s in seeds
    ]
    pairs = [jaccard(tops[i], tops[j]) for i in range(len(tops)) for j in range(i + 1, len(tops))]
    return {"mean_jaccard": float(np.mean(pairs)), "min_jaccard": float(np.min(pairs)), "k": k}


def randomisation_check(fitted_model, random_model, X_background, instance, groups=None, k: int = 5,
                        n_samples: int = 2000) -> dict[str, float]:
    """Compare explanations of the real model and of one trained on shuffled labels.

    If both highlight the same features, the explanation is describing the data or the
    sampling scheme rather than the model, and it fails the check.
    """
    real = explain_instance(fitted_model, X_background, instance, groups=groups, n_samples=n_samples, seed=7)
    fake = explain_instance(random_model, X_background, instance, groups=groups, n_samples=n_samples, seed=7)
    overlap = jaccard(top_features(real.weights, k), top_features(fake.weights, k))
    correlation = float(np.corrcoef(real.weights.to_numpy(), fake.weights.to_numpy())[0, 1])
    return {"top_k_overlap": overlap, "weight_correlation": correlation, "k": k}


def fidelity_ceiling(model, X_background, instance, groups=None, n_samples: int = 4000,
                     keep_probability: float = 0.5, kernel_width: float = 0.5,
                     seed: int = 11) -> dict[str, float]:
    """Separate 'the surrogate is too simple' from 'the representation is too coarse'.

    Two linear fits on the same neighbourhood:

    - the LIME representation, one on/off indicator per attribute ("was this
      attribute kept at the instance's value?")
    - the same neighbourhood regressed on the actual encoded feature values

    A large gap means the explanation's ceiling is set by the representation, not
    by linearity: an on/off indicator cannot say *which* value an attribute was
    swapped to, and on categorical credit data that is most of the variance.
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(seed)
    columns = list(X_background.columns)
    groups = groups or {c: [c] for c in columns}
    names = list(groups)
    positions = {c: j for j, c in enumerate(columns)}
    background = X_background.to_numpy()
    x = instance[columns].to_numpy(dtype=float)

    kept = rng.random((n_samples, len(names))) < keep_probability
    donors = rng.integers(0, background.shape[0], size=n_samples)
    samples = np.repeat(x[None, :], n_samples, axis=0)
    for gi, name in enumerate(names):
        cols = [positions[c] for c in groups[name]]
        replace = ~kept[:, gi]
        samples[np.ix_(replace, cols)] = background[np.ix_(donors[replace], cols)]

    unchanged = np.column_stack(
        [
            (samples[:, [positions[c] for c in groups[name]]]
             == x[[positions[c] for c in groups[name]]]).all(axis=1).astype(float)
            for name in names
        ]
    )
    probabilities = model.predict_proba(pd.DataFrame(samples, columns=columns))[:, 1]
    weights = np.exp(-((1.0 - unchanged.mean(axis=1)) ** 2) / (kernel_width**2))

    def weighted_r2(design: np.ndarray) -> float:
        fitted = Ridge(alpha=1.0).fit(design, probabilities, sample_weight=weights).predict(design)
        return float(r2_score(probabilities, fitted, sample_weight=weights))

    binary_r2 = weighted_r2(unchanged)
    values_r2 = weighted_r2(samples)
    return {
        "binary_representation_r2": binary_r2,
        "actual_values_r2": values_r2,
        "representation_gap": values_r2 - binary_r2,
        "neighbourhood_sd": float(probabilities.std()),
    }
