"""Statlog (German Credit) data: 1,000 applicants, 20 attributes, good/bad risk.

Column names and code meanings come from the dataset's own documentation
(`german.doc`). Codes are decoded to readable labels so an explanation says
"checking account < 0 DM" rather than "A11".
"""

from __future__ import annotations

import io
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, DATA_URL, RAW_FILE, SEED

COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment_since", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age_years", "other_installment_plans",
    "housing", "existing_credits", "job", "dependents", "telephone", "foreign_worker",
]

CODES = {
    "checking_status": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "none"},
    "credit_history": {
        "A30": "no credit taken", "A31": "all paid back", "A32": "paid back to date",
        "A33": "past delay", "A34": "critical / other credits",
    },
    "savings_status": {
        "A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM",
        "A65": "unknown / none",
    },
    "employment_since": {
        "A71": "unemployed", "A72": "< 1 year", "A73": "1-4 years", "A74": "4-7 years",
        "A75": ">= 7 years",
    },
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {
        "A121": "real estate", "A122": "building society savings", "A123": "car or other",
        "A124": "unknown / none",
    },
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
}

# Attributes excluded from the model: the dataset encodes sex and marital status in one
# column, and foreign worker status is a protected characteristic. They are kept aside
# so the explanations can be checked for proxy effects instead of being trained on them.
PROTECTED = ["personal_status_sex", "foreign_worker"]

NUMERIC = [
    "duration_months", "credit_amount", "installment_rate", "residence_since",
    "age_years", "existing_credits", "dependents",
]


def download() -> None:
    if RAW_FILE.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATA_URL) as resp:  # noqa: S310 (fixed, trusted URL)
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(DATA_DIR)


def load() -> pd.DataFrame:
    """Return the decoded dataframe with a binary `bad_risk` target."""
    download()
    df = pd.read_csv(RAW_FILE, sep=r"\s+", header=None, names=[*COLUMNS, "target"])
    for column, mapping in CODES.items():
        df[column] = df[column].map(mapping).fillna(df[column])
    # 1 = good, 2 = bad in the source file; model the bad-risk event.
    df["bad_risk"] = (df.pop("target") == 2).astype(int)
    return df


def feature_groups(raw: pd.DataFrame, encoded: pd.DataFrame) -> dict[str, list[str]]:
    """Map each original attribute to the encoded columns it produced.

    Explanations are built per attribute, not per dummy column: perturbing
    `checking_status_none` on its own would produce an applicant with two checking
    account statuses at once, which the model has never seen and no one can read.
    """
    groups: dict[str, list[str]] = {}
    for attribute in raw.columns:
        if attribute in ("bad_risk", *PROTECTED):
            continue
        if attribute in encoded.columns:
            groups[attribute] = [attribute]
        else:
            groups[attribute] = [c for c in encoded.columns if c.startswith(f"{attribute}_")]
    return {k: v for k, v in groups.items() if v}


def split(df: pd.DataFrame):
    """One-hot encode, drop protected attributes, and split 70/30 stratified."""
    y = df["bad_risk"]
    protected = df[PROTECTED].copy()
    X = pd.get_dummies(df.drop(columns=["bad_risk", *PROTECTED]), drop_first=False).astype(float)
    groups = feature_groups(df, X)
    X_tr, X_te, y_tr, y_te, p_tr, p_te = train_test_split(
        X, y, protected, test_size=0.3, stratify=y, random_state=SEED
    )
    return X_tr, X_te, y_tr, y_te, p_tr, p_te, groups
