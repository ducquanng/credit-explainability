import pandas as pd

from creditxai.data import NUMERIC, PROTECTED, feature_groups


def test_feature_groups_map_attributes_to_their_encoded_columns():
    raw = pd.DataFrame({"age_years": [30], "housing": ["own"], "bad_risk": [0],
                        "personal_status_sex": ["x"], "foreign_worker": ["yes"]})
    encoded = pd.DataFrame({"age_years": [30.0], "housing_own": [1.0], "housing_rent": [0.0]})
    groups = feature_groups(raw, encoded)
    assert groups["age_years"] == ["age_years"]
    assert sorted(groups["housing"]) == ["housing_own", "housing_rent"]
    assert not set(PROTECTED) & set(groups)


def test_numeric_columns_are_declared_consistently():
    assert "credit_amount" in NUMERIC and "age_years" in NUMERIC
