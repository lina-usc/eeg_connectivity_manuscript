"""Tests for utils.simulation."""

import numpy as np
import pytest

from utils.constants import confounder_list
from utils.simulation import get_ground_truth_dict, normalize, simulate_confounder


def test_get_ground_truth_dict_keys():
    result = get_ground_truth_dict()
    assert set(result.keys()) == {'common_input', 'indirect_connections', 'volume_conduction'}


def test_get_ground_truth_dict_shape():
    result = get_ground_truth_dict()
    for key, mat in result.items():
        assert mat.shape == (3, 3), f"{key} matrix has wrong shape"


def test_get_ground_truth_dict_value_range():
    np.random.seed(42)
    result = get_ground_truth_dict()
    for key, mat in result.items():
        nonzero = mat[mat != 0]
        assert np.all(nonzero >= 0.2), f"{key} has value below 0.2"
        assert np.all(nonzero <= 1.0), f"{key} has value above 1.0"


def test_get_ground_truth_dict_common_input_structure():
    np.random.seed(0)
    result = get_ground_truth_dict()
    mat = result['common_input']
    # Only row 2 (y2) drives others; rows 0 and 1 should be zero
    assert np.all(mat[:2, :] == 0)
    # Diagonal always 0
    assert mat[2, 2] == 0


def test_get_ground_truth_dict_volume_conduction_structure():
    np.random.seed(0)
    result = get_ground_truth_dict()
    mat = result['volume_conduction']
    # Only (2,0) should be nonzero
    assert mat[2, 0] != 0
    assert mat[2, 1] == 0


@pytest.mark.parametrize("confounder", confounder_list)
def test_simulate_confounder_static_keys(confounder):
    result = simulate_confounder(confounder, dynamic=False)
    assert 'signals' in result
    assert 'f0' in result and 'f1' in result and 'f2' in result


@pytest.mark.parametrize("confounder", confounder_list)
def test_simulate_confounder_static_signal_shape(confounder):
    result = simulate_confounder(confounder, dynamic=False)
    signals = result['signals']
    assert len(signals) == 3
    # 100 s at 250 Hz
    assert len(signals[0]) == 25000


@pytest.mark.parametrize("confounder", confounder_list)
def test_simulate_confounder_static_frequencies_in_range(confounder):
    np.random.seed(7)
    result = simulate_confounder(confounder, dynamic=False)
    assert 1 <= result['f0'] <= 40
    assert 1 <= result['f1'] <= 40
    assert 1 <= result['f2'] <= 40


@pytest.mark.parametrize("confounder", confounder_list)
def test_simulate_confounder_dynamic_keys(confounder):
    result = simulate_confounder(confounder, dynamic=True)
    assert 'signals' in result
    assert len(result['signals']) == 3
    assert len(result['signals'][0]) == 25000


def test_normalize_range():
    mat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = normalize(mat)
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)


def test_normalize_uniform_array():
    mat = np.array([3.0, 3.0, 3.0])
    result = normalize(mat)
    # Division by zero produces NaN
    assert np.all(np.isnan(result))


def test_normalize_2d():
    mat = np.array([[0.0, 1.0], [2.0, 3.0]])
    result = normalize(mat)
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)
