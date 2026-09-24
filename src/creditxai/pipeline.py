"""Train the black box, explain it, check the explanations, write the report.

Usage: python -m creditxai.pipeline
"""

from __future__ import annotations

import json
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from .config import FIG_DIR, MIN_LOCAL_R2, MIN_TOP_K_JACCARD, REPORT_DIR, SEED  # noqa: E402
from .data import load, split  # noqa: E402
from .explain.global_ import permutation_importance_table  # noqa: E402
from .explain.local import explain_instance  # noqa: E402
from .explain.quality import (  # noqa: E402
    fidelity_ceiling,
    randomisation_check,
    stability_jaccard,
    top_features,
)
from .models import black_box, glass_box  # noqa: E402

N_EXPLAINED = 25


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def run() -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    X_tr, X_te, y_tr, y_te, prot_tr, prot_te, groups = split(df)

    mlp = black_box().fit(X_tr, y_tr)
    logit = glass_box().fit(X_tr, y_tr)
    auc_mlp = roc_auc_score(y_te, mlp.predict_proba(X_te)[:, 1])
    auc_logit = roc_auc_score(y_te, logit.predict_proba(X_te)[:, 1])

    # A model fitted on shuffled labels, for the randomisation check.
    rng = np.random.default_rng(SEED)
    y_shuffled = pd.Series(rng.permutation(y_tr.to_numpy()), index=y_tr.index)
    mlp_random = black_box().fit(X_tr, y_shuffled)

    global_imp = permutation_importance_table(mlp, X_te, y_te)

    # Explain the highest-risk applicants: those are the decisions someone will contest.
    probs = mlp.predict_proba(X_te)[:, 1]
    order = np.argsort(-probs)[:N_EXPLAINED]
    explanations = [explain_instance(mlp, X_tr, X_te.iloc[i], groups=groups, seed=SEED) for i in order]
    fidelities = [e.local_r2 for e in explanations]

    focus_idx = int(order[0])
    focus = explanations[0]
    stability = stability_jaccard(mlp, X_tr, X_te.iloc[focus_idx], groups=groups)
    randomisation = randomisation_check(mlp, mlp_random, X_tr, X_te.iloc[focus_idx], groups=groups)
    ceiling = fidelity_ceiling(mlp, X_tr, X_te.iloc[focus_idx], groups=groups)

    # ------------------------------------------------------------------- figures
    top_global = global_imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(top_global["feature"], top_global["importance"], xerr=top_global["std"], color="#4C72B0")
    ax.set(xlabel="Drop in test AUC when shuffled", title="Global: what the MLP relies on")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "global_importance.png", dpi=150)
    plt.close(fig)

    local_top = focus.top(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(local_top.index, local_top.to_numpy(),
            color=["#C44E52" if v > 0 else "#55A868" for v in local_top.to_numpy()])
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel="Contribution to predicted risk (red = raises risk)",
           title=f"Local: applicant {focus_idx}, predicted PD {focus.prediction:.2f}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "local_explanation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(fidelities, bins=12, color="#4C72B0", edgecolor="white")
    ax.axvline(MIN_LOCAL_R2, color="#C44E52", ls="--", label=f"threshold {MIN_LOCAL_R2}")
    ax.set(xlabel="Weighted local $R^2$ of the surrogate", ylabel="Applicants",
           title=f"Explanation fidelity across {N_EXPLAINED} explained decisions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fidelity.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------- report
    passed = {
        "fidelity": float(np.mean(fidelities)) >= MIN_LOCAL_R2,
        "stability": stability["mean_jaccard"] >= MIN_TOP_K_JACCARD,
        "randomisation": randomisation["top_k_overlap"] < 0.5,
    }
    verdict = {k: ("PASS" if v else "FAIL") for k, v in passed.items()}

    lines = [
        "# Explaining a black-box credit model",
        "",
        f"Generated {date.today().isoformat()} by `python -m creditxai.pipeline`. "
        f"Data: Statlog (German Credit), {len(df):,} applicants, {df['bad_risk'].mean():.1%} bad risk. "
        "Sex/marital status and foreign-worker status are excluded from the features.",
        "",
        "## 1. Models",
        "",
        md_table(
            ["Model", "Test AUC", "Role"],
            [
                ["MLP (50 hidden units)", f"{auc_mlp:.3f}", "the black box being explained"],
                ["Logistic regression", f"{auc_logit:.3f}", "transparent comparison"],
            ],
        ),
        "",
        f"The gap in AUC is {auc_mlp - auc_logit:+.3f}. That difference is what the black box "
        "buys, and it is the number to weigh against having to explain it at all.",
        "",
        "## 2. Are the explanations trustworthy?",
        "",
        md_table(
            ["Check", "What it asks", "Result", "Verdict"],
            [
                ["Fidelity", "Does the surrogate track the model near the instance?",
                 f"mean local R² {np.mean(fidelities):.2f} (min {np.min(fidelities):.2f})",
                 verdict["fidelity"]],
                ["Stability", "Same top-5 features when only the seed changes?",
                 f"mean Jaccard {stability['mean_jaccard']:.2f} (worst pair {stability['min_jaccard']:.2f})",
                 verdict["stability"]],
                ["Randomisation", "Does the explanation change when the model is trained on shuffled labels?",
                 f"top-5 overlap {randomisation['top_k_overlap']:.2f}, "
                 f"weight correlation {randomisation['weight_correlation']:+.2f}", verdict["randomisation"]],
            ],
        ),
        "",
        "The fidelity check fails, and the next table says why it is the explanation "
        "method rather than the model that is at fault.",
        "",
        md_table(
            ["Linear fit on the same neighbourhood", "Weighted R²"],
            [
                ["LIME representation: one on/off indicator per attribute",
                 f"{ceiling['binary_representation_r2']:.3f}"],
                ["The actual encoded feature values", f"{ceiling['actual_values_r2']:.3f}"],
            ],
        ),
        "",
        f"Two ceilings stack up here. The representation costs "
        f"{ceiling['representation_gap']:.2f} of R²: an indicator records *whether* an "
        "attribute changed, never *what it changed to*. Even given the actual values, a "
        f"linear fit reaches only {ceiling['actual_values_r2']:.2f}, so the rest is the "
        "model's own non-linearity across a neighbourhood where predicted PD spans almost "
        f"the whole interval (SD {ceiling['neighbourhood_sd']:.2f}). Presenting the top-5 "
        f"features from a surrogate with an R² of {np.mean(fidelities):.2f} as *the reason* "
        "for a decline would overstate what was measured. As a ranking of what to examine "
        "first, it is still useful.",
        "",
        "![Fidelity](figures/fidelity.png)",
        "",
        "## 3. Global attribution",
        "",
        "![Global importance](figures/global_importance.png)",
        "",
        md_table(["Feature", "AUC drop when shuffled", "SD"],
                 [[r.feature, f"{r.importance:.4f}", f"{r.std:.4f}"]
                  for r in global_imp.head(8).itertuples()]),
        "",
        "## 4. One decision, explained",
        "",
        f"Applicant {focus_idx}, predicted probability of default {focus.prediction:.2f}, "
        f"surrogate local R² {focus.local_r2:.2f}.",
        "",
        "![Local explanation](figures/local_explanation.png)",
        "",
        md_table(["Feature", "Contribution to risk"],
                 [[f, f"{v:+.4f}"] for f, v in focus.top(8).items()]),
        "",
        "## 5. Limitations",
        "",
        "- A local surrogate describes a neighbourhood, not a rule. It does not say what would "
        "have made this application succeed; that is a counterfactual question and a different method.",
        "- Permutation importance on one-hot columns understates individual levels, because a "
        "shuffled column is partly recoverable from its siblings.",
        "- Excluded protected attributes can still act through proxies. Checking that means testing "
        "outcomes by group, not reading an explanation.",
        "- 1,000 rows from 1994 German banking. The method transfers; the coefficients do not.",
    ]
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "explainability_report.md").write_text("\n".join(lines))

    summary = {
        "auc_mlp": auc_mlp, "auc_logit": auc_logit,
        "mean_local_r2": float(np.mean(fidelities)), "min_local_r2": float(np.min(fidelities)),
        "stability": stability, "randomisation": randomisation, "fidelity_ceiling": ceiling,
        "checks": verdict,
        "focus_applicant": focus_idx, "focus_top_features": top_features(focus.weights, 5),
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
