"""
Pure-Python tests for mia.py's percentile-rank math and small-forget-set caveat --
no torch/model needed.
"""
import pytest

from verification import config as v_config
from verification.mia import percentile_rank


def test_percentile_rank_below_all_null_values_is_zero():
    null_sorted = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile_rank(0.5, null_sorted) == 0.0


def test_percentile_rank_above_all_null_values_is_one():
    null_sorted = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile_rank(5.5, null_sorted) == 1.0


def test_percentile_rank_matches_position_for_an_exact_value():
    null_sorted = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile_rank(3.0, null_sorted) == 2 / 5


def test_percentile_rank_rejects_empty_null():
    with pytest.raises(ValueError):
        percentile_rank(1.0, [])


def test_small_forget_set_threshold_is_configured_and_sane():
    assert v_config.MIA_MIN_FORGET_SET_FOR_CONFIDENCE >= 1
