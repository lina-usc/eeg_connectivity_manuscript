"""Tests for utils.statistics."""

import numpy as np
import pandas as pd
import pytest

from utils.constants import confounder_list
from utils.statistics import build_ci_dataframe, compute_bootstrap_mse_corr, compute_ci_dict


def _make_dummy_dicts(confounder_list, methods, connections, n_samples=100):
    """Build minimal estimate/ground_truth dicts for testing."""
    estimate_dict = {}
    ground_truth_dict = {}
    for conf in confounder_list:
        estimate_dict[conf] = {}
        ground_truth_dict[conf] = {}
        for method in methods:
            estimate_dict[conf][method] = {}
            ground_truth_dict[conf][method] = {}
            for conn in connections:
                estimate_dict[conf][method][conn] = list(np.random.rand(n_samples))
                ground_truth_dict[conf][method][conn] = list(np.random.rand(n_samples))
    return estimate_dict, ground_truth_dict


@pytest.fixture(scope="module")
def bootstrap_results():
    np.random.seed(0)
    methods = ['coh', 'ciplv']
    connections = ['(0, 1)', '(1, 2)']
    pairs = [('coh', 'ciplv')]
    est, gt = _make_dummy_dicts(confounder_list, methods, connections)
    mse_dict, corr_dict = compute_bootstrap_mse_corr(est, gt, confounder_list, methods)
    return mse_dict, corr_dict, est, pairs, connections


# ---------------------------------------------------------------------------
# compute_bootstrap_mse_corr
# ---------------------------------------------------------------------------

def test_compute_bootstrap_mse_corr_top_keys(bootstrap_results):
    mse_dict, corr_dict, _, _, _ = bootstrap_results
    assert set(mse_dict.keys()) == set(confounder_list)
    assert set(corr_dict.keys()) == set(confounder_list)


def test_compute_bootstrap_mse_corr_mse_nonnegative(bootstrap_results):
    mse_dict, _, _, _, connections = bootstrap_results
    for conf in confounder_list:
        for method in mse_dict[conf]:
            for conn in connections:
                vals = mse_dict[conf][method][conn]
                assert all(v >= 0 for v in vals), "MSE should be non-negative"


def test_compute_bootstrap_mse_corr_corr_in_range(bootstrap_results):
    _, corr_dict, _, _, connections = bootstrap_results
    for conf in confounder_list:
        for method in corr_dict[conf]:
            for conn in connections:
                vals = corr_dict[conf][method][conn]
                assert all(-1 <= v <= 1 for v in vals), "Correlation must be in [-1, 1]"


def test_compute_bootstrap_mse_corr_sample_count(bootstrap_results):
    mse_dict, _, _, _, connections = bootstrap_results
    # 1000 bootstrap iterations
    for conf in confounder_list:
        for method in mse_dict[conf]:
            for conn in connections:
                assert len(mse_dict[conf][method][conn]) == 1000


# ---------------------------------------------------------------------------
# compute_ci_dict
# ---------------------------------------------------------------------------

def test_compute_ci_dict_structure(bootstrap_results):
    mse_dict, corr_dict, est, pairs, connections = bootstrap_results
    ci_mse, ci_corr = compute_ci_dict(mse_dict, corr_dict, est, confounder_list, pairs)
    assert set(ci_mse.keys()) == set(confounder_list)
    assert set(ci_corr.keys()) == set(confounder_list)


def test_compute_ci_dict_tuples_length_2(bootstrap_results):
    mse_dict, corr_dict, est, pairs, connections = bootstrap_results
    ci_mse, ci_corr = compute_ci_dict(mse_dict, corr_dict, est, confounder_list, pairs)
    for conf in confounder_list:
        for pair_key, conn_dict in ci_mse[conf].items():
            for conn, ci in conn_dict.items():
                assert len(ci) == 2, "CI should be a 2-tuple"


def test_compute_ci_dict_lower_le_upper(bootstrap_results):
    mse_dict, corr_dict, est, pairs, connections = bootstrap_results
    ci_mse, ci_corr = compute_ci_dict(mse_dict, corr_dict, est, confounder_list, pairs)
    for conf in confounder_list:
        for pair_key, conn_dict in ci_mse[conf].items():
            for conn, (lo, hi) in conn_dict.items():
                assert lo <= hi, "Lower CI bound must be <= upper"


def test_compute_ci_dict_abbreviates_directed_pairs():
    """Directed pair names should be abbreviated to short codes."""
    np.random.seed(1)
    methods = [
        'direct_directed_transfer_function',
        'generalized_partial_directed_coherence',
        'pairwise_spectral_granger_prediction',
    ]
    connections = ['(0, 1)']
    pairs = [
        ('direct_directed_transfer_function', 'generalized_partial_directed_coherence'),
        ('direct_directed_transfer_function', 'pairwise_spectral_granger_prediction'),
        ('generalized_partial_directed_coherence', 'pairwise_spectral_granger_prediction'),
    ]
    est, gt = _make_dummy_dicts(confounder_list, methods, connections)
    mse_dict, corr_dict = compute_bootstrap_mse_corr(est, gt, confounder_list, methods)
    ci_mse, _ = compute_ci_dict(mse_dict, corr_dict, est, confounder_list, pairs)
    # Abbreviated keys should appear
    all_pairs_flat = [str(k) for conf in confounder_list for k in ci_mse[conf].keys()]
    assert any('dDTF' in p for p in all_pairs_flat)
    assert any('gPDC' in p for p in all_pairs_flat)
    assert any('pSGP' in p for p in all_pairs_flat)


# ---------------------------------------------------------------------------
# build_ci_dataframe
# ---------------------------------------------------------------------------

def test_build_ci_dataframe_returns_list(bootstrap_results):
    mse_dict, corr_dict, est, pairs, connections = bootstrap_results
    ci_mse, _ = compute_ci_dict(mse_dict, corr_dict, est, confounder_list, pairs)
    result = build_ci_dataframe(ci_mse, confounder_list, slice(None))
    assert isinstance(result, list)


def test_build_ci_dataframe_length_matches_confounders(bootstrap_results):
    mse_dict, corr_dict, est, pairs, connections = bootstrap_results
    ci_mse, _ = compute_ci_dict(mse_dict, corr_dict, est, confounder_list, pairs)
    result = build_ci_dataframe(ci_mse, confounder_list, slice(None))
    assert len(result) == len(confounder_list)


def test_build_ci_dataframe_each_element_is_dataframe(bootstrap_results):
    mse_dict, corr_dict, est, pairs, connections = bootstrap_results
    ci_mse, _ = compute_ci_dict(mse_dict, corr_dict, est, confounder_list, pairs)
    result = build_ci_dataframe(ci_mse, confounder_list, slice(None))
    for df in result:
        assert isinstance(df, pd.DataFrame)


def test_build_ci_dataframe_has_confounder_column(bootstrap_results):
    mse_dict, corr_dict, est, pairs, connections = bootstrap_results
    ci_mse, _ = compute_ci_dict(mse_dict, corr_dict, est, confounder_list, pairs)
    result = build_ci_dataframe(ci_mse, confounder_list, slice(None))
    for df in result:
        assert 'Confounder' in df.columns


def test_build_ci_dataframe_has_pairs_column(bootstrap_results):
    mse_dict, corr_dict, est, pairs, connections = bootstrap_results
    ci_mse, _ = compute_ci_dict(mse_dict, corr_dict, est, confounder_list, pairs)
    result = build_ci_dataframe(ci_mse, confounder_list, slice(None))
    for df in result:
        assert 'Pairs' in df.columns
