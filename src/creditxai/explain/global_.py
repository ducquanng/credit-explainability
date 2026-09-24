"""Global attribution: what the model relies on across the whole test set."""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance

from ..config import SEED


def permutation_importance_table(model, X, y, n_repeats: int = 10, scoring: str = "roc_auc") -> pd.DataFrame:
    """Drop in test score when each column is shuffled, with its standard deviation.

    Permutation importance on correlated one-hot columns understates any single
    column, because a shuffled column can be reconstructed from its siblings. Read
    it as a floor on importance, not a ranking to defend to three decimal places.
    """
    result = permutation_importance(
        model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=SEED, n_jobs=1
    )
    return (
        pd.DataFrame(
            {"feature": X.columns, "importance": result.importances_mean, "std": result.importances_std}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
