# Copyright (c) 2026 José M. Álvarez
# SPDX-License-Identifier: Apache-2.0

import pytest

from situation_testing._utils import get_neg_value, get_pro_value


def test_get_pro_value_returns_protected():
    assert get_pro_value({"protected": "F"}) == "F"


def test_get_pro_value_missing_key_raises():
    with pytest.raises(ValueError, match="protected"):
        get_pro_value({"unprotected": "M"})


def test_get_neg_value_returns_negative():
    assert get_neg_value({"negative": 0}) == 0


def test_get_neg_value_missing_key_raises():
    with pytest.raises(ValueError, match="negative"):
        get_neg_value({"positive": 1})
