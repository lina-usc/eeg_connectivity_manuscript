"""Tests for utils.plotting (layout helpers that don't require a display)."""

import networkx as nx

from utils.plotting import setup_graph_layout


def test_setup_graph_layout_return_length():
    result = setup_graph_layout()
    assert len(result) == 5


def test_setup_graph_layout_graph_type():
    G, pos, edge_list_1, edge_list_2, arc_rad = setup_graph_layout()
    assert isinstance(G, nx.DiGraph)


def test_setup_graph_layout_node_count():
    G, pos, edge_list_1, edge_list_2, arc_rad = setup_graph_layout()
    assert len(G.nodes) == 3


def test_setup_graph_layout_default_node_names():
    G, pos, edge_list_1, edge_list_2, arc_rad = setup_graph_layout()
    assert set(G.nodes) == {'y0', 'y1', 'y2'}


def test_setup_graph_layout_custom_node_names():
    G, pos, edge_list_1, edge_list_2, arc_rad = setup_graph_layout(('Y0', 'Y1', 'Y2'))
    assert set(G.nodes) == {'Y0', 'Y1', 'Y2'}


def test_setup_graph_layout_edge_lists_length():
    G, pos, edge_list_1, edge_list_2, arc_rad = setup_graph_layout()
    assert len(edge_list_1) == 3
    assert len(edge_list_2) == 3


def test_setup_graph_layout_pos_keys_match_nodes():
    G, pos, edge_list_1, edge_list_2, arc_rad = setup_graph_layout()
    assert set(pos.keys()) == set(G.nodes)


def test_setup_graph_layout_arc_rad_type():
    G, pos, edge_list_1, edge_list_2, arc_rad = setup_graph_layout()
    assert isinstance(arc_rad, float)


def test_setup_graph_layout_edge_list_1_edges_have_weight():
    G, pos, edge_list_1, edge_list_2, arc_rad = setup_graph_layout()
    for src, dst, attrs in edge_list_1:
        assert 'w' in attrs


def test_setup_graph_layout_edge_list_2_edges_have_weight():
    G, pos, edge_list_1, edge_list_2, arc_rad = setup_graph_layout()
    for src, dst, attrs in edge_list_2:
        assert 'w' in attrs
