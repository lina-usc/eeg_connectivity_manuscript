#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from utils.constants import (
    all_methods, confounder_list, directed_methods, undirected_methods,
)
from utils.simulation import estimate_connectivity, get_ground_truth_dict


# ### COMPILING EDGE LISTS

edge_lists_undirected = {}
for confounder in confounder_list:
    edge_lists = {}
    for method in undirected_methods:
        edges_list = []
        for i in range(100):
            con_mat_normalized = estimate_connectivity(method, confounder)
            edges = [
                round(con_mat_normalized[1][0], 2),
                round(con_mat_normalized[2][1], 2),
                round(con_mat_normalized[2][0], 2),
            ]
            edges_list.append(edges)
        mean_edges = np.array(edges_list)
        edge_lists[method] = np.round(mean_edges.mean(axis=0), 2)
    edge_lists_undirected[confounder] = edge_lists

print(edge_lists_undirected)

edge_list_1 = [('Y0', 'Y1', {'w': 'A1'}), ('Y1', 'Y2', {'w': 'B1'}), ('Y0', 'Y2', {'w': 'C1'})]
edge_list_2 = [('Y1', 'Y0', {'w': 'A2'}), ('Y2', 'Y1', {'w': 'B2'}), ('Y2', 'Y0', {'w': 'C2'})]


def _draw_topology_panel(axes, confounder, alpha_gt_1, alpha_gt_2, gt_edge_labels,
                          edge_lists, methods, node_names):
    G = nx.DiGraph()
    G.add_edges_from(edge_list_1)
    pos = nx.shell_layout(G, rotate=np.pi / 2)
    arc_rad = 0.05

    nx.draw_networkx_edges(G, pos, ax=axes[0], edgelist=edge_list_1,
                           connectionstyle=f'arc3, rad = {arc_rad}',
                           alpha=alpha_gt_1, arrowsize=25)
    nx.draw_networkx_edges(G, pos, ax=axes[0], edgelist=edge_list_2,
                           connectionstyle=f'arc3, rad = {arc_rad}',
                           alpha=alpha_gt_2, arrowsize=25)
    nx.draw_networkx_edge_labels(G, pos, ax=axes[0], edge_labels=gt_edge_labels,
                                 font_color='red')
    nx.draw_networkx_nodes(G, pos, ax=axes[0], node_color='orange', alpha=1.0)
    nx.draw_networkx_labels(G, pos, ax=axes[0])

    undirected_edge_list = [('Y0', 'Y1', {'w': 'A1'}),
                            ('Y1', 'Y2', {'w': 'B1'}),
                            ('Y0', 'Y2', {'w': 'C1'})]
    for ax_idx, method in zip(range(1, len(methods) + 1), methods):
        G2 = nx.DiGraph()
        G2.add_edges_from(undirected_edge_list)
        pos2 = nx.shell_layout(G2, rotate=np.pi / 2, scale=0.1)

        alpha = edge_lists[confounder][method]
        nx.draw_networkx_edges(G2, pos2, ax=axes[ax_idx],
                               edgelist=undirected_edge_list, alpha=alpha,
                               arrowstyle='<->')
        nx.draw_networkx_edge_labels(G2, pos2, ax=axes[ax_idx],
                                     edge_labels={
                                         ('Y0', 'Y1'): alpha[0],
                                         ('Y1', 'Y2'): alpha[1],
                                         ('Y0', 'Y2'): alpha[2],
                                     }, font_color='red')
        nx.draw_networkx_nodes(G2, pos2, ax=axes[ax_idx],
                               node_color='grey', alpha=1.0)
        nx.draw_networkx_labels(G2, pos2, ax=axes[ax_idx])

    labels = ['Ground truth'] + list(node_names)
    for ax, label in zip(axes, labels):
        ax.set_ylabel(label, rotation='horizontal', labelpad=50)


# ## Common input

fig, axes = plt.subplots(5, 1, figsize=(3, 8))
_draw_topology_panel(
    axes, 'common_input',
    alpha_gt_1=[0, 0, 0], alpha_gt_2=[0, 1, 1],
    gt_edge_labels={('Y1', 'Y0'): 0, ('Y2', 'Y1'): 1, ('Y2', 'Y0'): 1},
    edge_lists=edge_lists_undirected, methods=undirected_methods,
    node_names=['coh', 'ciPLV', 'imcoh', 'dwPLI'],
)
fig.suptitle('                   Non-dynamic system')
fig.tight_layout()
plt.savefig('topo_func_comm_non_dyn.png', dpi=300)


# ## Indirect connections

fig, axes = plt.subplots(5, 1, figsize=(3, 8))
_draw_topology_panel(
    axes, 'indirect_connections',
    alpha_gt_1=[0, 0, 0], alpha_gt_2=[1, 1, 0],
    gt_edge_labels={('Y1', 'Y0'): 1, ('Y2', 'Y1'): 1, ('Y2', 'Y0'): 0},
    edge_lists=edge_lists_undirected, methods=undirected_methods,
    node_names=['coh', 'ciPLV', 'imcoh', 'dwPLI'],
)
fig.suptitle('                   Non-dynamic system')
fig.tight_layout()
plt.savefig('topo_func_ind_non_dyn.png', dpi=300)


# ## Volume conduction

fig, axes = plt.subplots(5, 1, figsize=(3, 8))

G_vc = nx.Graph()
vc_edge_list = [('Y0', 'Y1', {'w': 'A1'}), ('Y1', 'Y2', {'w': 'B1'}), ('Y0', 'Y2', {'w': 'C1'})]
G_vc.add_edges_from(vc_edge_list)
pos_vc = nx.shell_layout(G_vc, rotate=np.pi / 2, scale=0.1)
nx.draw_networkx_edges(G_vc, pos_vc, ax=axes[0], edgelist=vc_edge_list, alpha=[0, 0, 0])
nx.draw_networkx_edge_labels(G_vc, pos_vc, ax=axes[0],
                              edge_labels={('Y1', 'Y0'): 0, ('Y2', 'Y1'): 0, ('Y2', 'Y0'): 1},
                              font_color='red')
nx.draw_networkx_nodes(G_vc, pos_vc, ax=axes[0], node_color='orange', alpha=1.0)
nx.draw_networkx_labels(G_vc, pos_vc, ax=axes[0])

for ax_idx, method in zip(range(1, 5), undirected_methods):
    G2 = nx.DiGraph()
    G2.add_edges_from(vc_edge_list)
    pos2 = nx.shell_layout(G2, rotate=np.pi / 2, scale=0.1)

    alpha = edge_lists_undirected['volume_conduction'][method]
    nx.draw_networkx_edges(G2, pos2, ax=axes[ax_idx], edgelist=vc_edge_list,
                           alpha=alpha, arrowstyle='<->')
    nx.draw_networkx_edge_labels(G2, pos2, ax=axes[ax_idx],
                                 edge_labels={
                                     ('Y0', 'Y1'): alpha[0],
                                     ('Y1', 'Y2'): alpha[1],
                                     ('Y0', 'Y2'): alpha[2],
                                 }, font_color='red')
    nx.draw_networkx_nodes(G2, pos2, ax=axes[ax_idx], node_color='grey', alpha=1.0)
    nx.draw_networkx_labels(G2, pos2, ax=axes[ax_idx])

axes[0].set_title('Non-dynamic system')
for ax, label in zip(axes, ['Ground truth', 'coh', 'ciPLV', 'imcoh', 'dwPLI']):
    ax.set_ylabel(label, rotation='horizontal', labelpad=50)

fig.tight_layout()
plt.savefig('topo_func_vol_non_dyn.png', dpi=300)
