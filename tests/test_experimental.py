"""Tests for utils.experimental (pure-Python functions only)."""

import numpy as np
import pytest

from utils.experimental import bootstrap_ci, outlier_indices, remove_outliers


# ---------------------------------------------------------------------------
# outlier_indices
# ---------------------------------------------------------------------------

def test_outlier_indices_detects_upper_outlier():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    idxs = outlier_indices(data)
    assert 5 in idxs  # index of 100.0


def test_outlier_indices_detects_lower_outlier():
    data = np.array([-100.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    idxs = outlier_indices(data)
    assert 0 in idxs  # index of -100.0


def test_outlier_indices_no_outliers():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    idxs = outlier_indices(data)
    assert idxs == []


def test_outlier_indices_returns_list():
    data = np.array([1.0, 2.0, 3.0])
    assert isinstance(outlier_indices(data), list)


# ---------------------------------------------------------------------------
# remove_outliers
# ---------------------------------------------------------------------------

def test_remove_outliers_removes_extreme():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
    result = remove_outliers(data)
    assert 100.0 not in result


def test_remove_outliers_preserves_inliers():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
    result = remove_outliers(data)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        assert v in result


def test_remove_outliers_no_outliers_unchanged():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = remove_outliers(data)
    assert result == data


def test_remove_outliers_returns_list():
    data = [1.0, 2.0, 3.0]
    assert isinstance(remove_outliers(data), list)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

def test_bootstrap_ci_returns_tuple():
    np.random.seed(0)
    result = bootstrap_ci([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_bootstrap_ci_lower_bound_less_than_upper():
    np.random.seed(0)
    lo, hi = bootstrap_ci([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert lo < hi


def test_bootstrap_ci_identical_groups_ci_contains_zero():
    np.random.seed(42)
    group = [1.0, 2.0, 3.0, 4.0, 5.0]
    lo, hi = bootstrap_ci(group, group)
    assert lo < 0 < hi or lo == 0 or hi == 0  # CI should straddle 0


def test_bootstrap_ci_separated_groups_positive_diff():
    np.random.seed(1)
    lo, hi = bootstrap_ci([1.0, 1.1, 0.9], [10.0, 10.1, 9.9])
    # group2 mean >> group1 mean, so diff (group2 - group1) should be positive
    assert lo > 0


def test_bootstrap_ci_default_n_iter():
    """CI bounds should be reproducible with fixed seed."""
    np.random.seed(99)
    result1 = bootstrap_ci([1.0, 2.0], [3.0, 4.0])
    np.random.seed(99)
    result2 = bootstrap_ci([1.0, 2.0], [3.0, 4.0], n_iter=1000)
    assert result1 == result2
