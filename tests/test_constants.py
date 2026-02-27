"""Tests for utils.constants."""

from utils.constants import (
    all_methods,
    comparison_pairs,
    confounder_list,
    directed_methods,
    undirected_methods,
)


def test_confounder_list_contents():
    assert confounder_list == [
        'common_input',
        'indirect_connections',
        'volume_conduction',
    ]


def test_undirected_methods_contents():
    assert set(undirected_methods) == {'coh', 'ciplv', 'wpli2_debiased', 'imaginary_coherence'}


def test_directed_methods_contents():
    assert set(directed_methods) == {
        'generalized_partial_directed_coherence',
        'direct_directed_transfer_function',
        'pairwise_spectral_granger_prediction',
    }


def test_all_methods_is_union():
    assert set(all_methods) == set(undirected_methods) | set(directed_methods)
    assert len(all_methods) == len(undirected_methods) + len(directed_methods)


def test_comparison_pairs_are_tuples():
    assert all(isinstance(p, tuple) and len(p) == 2 for p in comparison_pairs)


def test_comparison_pairs_reference_known_methods():
    for m1, m2 in comparison_pairs:
        assert m1 in all_methods, f"{m1} not in all_methods"
        assert m2 in all_methods, f"{m2} not in all_methods"


def test_comparison_pairs_no_self_pairs():
    for m1, m2 in comparison_pairs:
        assert m1 != m2
