# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from situation_testing._distance_functions import kdd2011dist, manhattan


def _atts(normalize=True):
    return {
        "continuous_atts": ["X"],
        "ordinal_atts": [],
        "nominal_atts": ["C"],
        "normalize": normalize,
    }


def test_kdd2011dist_mixed_continuous_and_nominal():
    tset = pd.DataFrame({"X": [0.0, 1.0, 2.0], "C": ["a", "b", "a"]})
    t = pd.Series({"X": 0.0, "C": "a"})

    dist = kdd2011dist(t, tset, ["X", "C"], _atts(normalize=True))

    # continuous contribution: |0 - X| = [0, 1, 2]
    # nominal contribution:    (a != C) = [0, 1, 0]
    # divided by n_atts = 2  -> [0, 1, 1]
    assert dist.tolist() == [0.0, 1.0, 1.0]


def test_kdd2011dist_preserves_index():
    tset = pd.DataFrame({"X": [0.0, 5.0], "C": ["a", "a"]}, index=[10, 20])
    t = pd.Series({"X": 0.0, "C": "a"})

    dist = kdd2011dist(t, tset, ["X", "C"], _atts(normalize=True))

    assert list(dist.index) == [10, 20]


def test_kdd2011dist_zero_distance_to_identical_row():
    tset = pd.DataFrame({"X": [3.0], "C": ["z"]})
    t = pd.Series({"X": 3.0, "C": "z"})

    dist = kdd2011dist(t, tset, ["X", "C"], _atts(normalize=True))

    assert dist.iloc[0] == 0.0


def test_kdd2011dist_ordinal_attribute():
    atts = {
        "continuous_atts": [],
        "ordinal_atts": ["O"],
        "nominal_atts": [],
        "normalize": True,
    }
    tset = pd.DataFrame({"O": [0.0, 1.0, 2.0]})
    t = pd.Series({"O": 0.0})

    dist = kdd2011dist(t, tset, ["O"], atts)

    # n_vals = nunique - 1 = 2; |0/2 - [0, 1, 2]/2| = [0, 0.5, 1.0]
    assert dist.tolist() == [0.0, 0.5, 1.0]


def test_manhattan_distance_normalized():
    atts = {
        "continuous_atts": ["X"],
        "ordinal_atts": [],
        "nominal_atts": [],
        "normalize": True,
    }
    tset = pd.DataFrame({"X": [0.0, 1.0, 2.0]})
    t = pd.Series({"X": 0.0})

    dist = manhattan(t, tset, ["X"], atts)

    assert dist.tolist() == [0.0, 1.0, 2.0]
