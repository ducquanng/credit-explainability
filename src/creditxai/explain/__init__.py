"""Global and local explanations, plus the checks that say whether to trust them."""

from .global_ import permutation_importance_table
from .local import LocalExplanation, explain_instance
from .quality import stability_jaccard, top_features

__all__ = [
    "permutation_importance_table",
    "explain_instance",
    "LocalExplanation",
    "stability_jaccard",
    "top_features",
]
