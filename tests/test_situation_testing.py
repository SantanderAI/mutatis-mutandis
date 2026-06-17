# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from situation_testing import SituationTesting


@pytest.fixture
def toy_df():
    # 4 protected ("F") + 4 non-protected ("M") individuals with a continuous
    # score and a binary outcome Y (1 = positive, 0 = negative).
    return pd.DataFrame(
        {
            "Gender": ["F", "F", "F", "F", "M", "M", "M", "M"],
            "Score": [10.0, 11.0, 12.0, 13.0, 10.5, 11.5, 12.5, 13.5],
            "Y": [0, 0, 1, 1, 1, 1, 1, 1],
        }
    )


def test_setup_baseline_normalizes_continuous(toy_df):
    st = SituationTesting()
    st.setup_baseline(
        toy_df.copy(),
        nominal_atts=["Gender"],
        continuous_atts=["Score"],
        normalize=True,
    )
    # StandardScaler -> zero mean on the continuous column.
    assert st.df["Score"].mean() == pytest.approx(0.0, abs=1e-9)


def test_run_returns_diff_series_for_protected_group(toy_df):
    st = SituationTesting()
    st.setup_baseline(
        toy_df.copy(),
        nominal_atts=["Gender"],
        continuous_atts=["Score"],
        normalize=True,
    )

    res = st.run(
        target_att="Y",
        target_val={"negative": 0},
        sensitive_att="Gender",
        sensitive_val={"protected": "F"},
        k=2,
    )

    # One entry per individual; values are finite differences in [-1, 1].
    assert isinstance(res, pd.Series)
    assert len(res) == len(toy_df)
    assert np.isfinite(res.to_numpy()).all()
    assert res.between(-1.0, 1.0).all()


def test_get_test_discrimination_reports_protected_individuals(toy_df):
    st = SituationTesting()
    st.setup_baseline(
        toy_df.copy(),
        nominal_atts=["Gender"],
        continuous_atts=["Score"],
        normalize=True,
    )
    st.run(
        target_att="Y",
        target_val={"negative": 0},
        sensitive_att="Gender",
        sensitive_val={"protected": "F"},
        k=2,
    )

    report = st.get_test_discrimination()

    # One row per protected ("F") individual, with the expected columns.
    assert len(report) == 4
    assert {"individual", "p1", "p2", "org_diff", "cfST"}.issubset(report.columns)
    assert set(report["cfST"]).issubset({"Yes", "No"})


def test_counterfactual_run_reports_unfairness(toy_df):
    # Counterfactual twin dataset: same features, but flipped outcomes for some
    # protected individuals to exercise the counterfactual-unfairness branches.
    cf_df = toy_df.copy()
    cf_df["Y"] = [1, 1, 0, 1, 1, 1, 1, 1]

    st = SituationTesting()
    st.setup_baseline(
        toy_df.copy(),
        cf_df=cf_df,
        nominal_atts=["Gender"],
        continuous_atts=["Score"],
        normalize=True,
    )

    res = st.run(
        target_att="Y",
        target_val={"negative": 0},
        sensitive_att="Gender",
        sensitive_val={"protected": "F"},
        k=2,
        return_counterfactual_fairness=True,
    )

    cf_unfairness = st.res_counterfactual_unfairness

    assert isinstance(res, pd.Series)
    # 0 = no change, 1 = neg->pos discrimination, 2 = pos->neg discrimination.
    assert set(cf_unfairness.unique()).issubset({0, 1, 2})
    # Individual 0 flips 0 -> 1 (negative outcome under factual) => coded as 1.
    assert cf_unfairness.loc[0] == 1
    # Individual 2 flips 1 -> 0 (positive outcome under factual) => coded as 2.
    assert cf_unfairness.loc[2] == 2
