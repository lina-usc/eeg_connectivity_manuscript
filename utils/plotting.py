import networkx as nx
import numpy as np
import pandas as pd


def plot_bar_from_dict(confounder, axes_cell, values_dict, node_key, methods_list, xticklabels, title):
    df_list = []
    yerr_list = []

    for method in methods_list:
        df = pd.DataFrame(values_dict[confounder][method][node_key])
        df = df.rename(columns={0: method})
        df = df.sort_values(by=method).reset_index(drop=True)

        err = df[method][974] - df[method][24]
        yerr_list.append(err)
        df_list.append(df)

    plot_df = pd.concat(df_list, axis=1)
    means = [plot_df[method].mean() for method in methods_list]
    axes_cell.bar(range(len(methods_list)), means, yerr=yerr_list)
    axes_cell.set_title(title, fontweight="bold", fontsize=11)
    axes_cell.set_xticks(range(len(methods_list)))
    axes_cell.set_xticklabels(xticklabels, fontsize=10)


def setup_graph_layout(node_names=('y0', 'y1', 'y2')):
    n0, n1, n2 = node_names
    edge_list_1 = [(n0, n1, {'w': 'A1'}), (n1, n2, {'w': 'B1'}), (n0, n2, {'w': 'C1'})]
    edge_list_2 = [(n1, n0, {'w': 'A2'}), (n2, n1, {'w': 'B2'}), (n2, n0, {'w': 'C2'})]

    G = nx.DiGraph()
    G.add_edges_from(edge_list_1)
    pos = nx.shell_layout(G, rotate=np.pi / 2)
    arc_rad = 0.05

    return G, pos, edge_list_1, edge_list_2, arc_rad


def draw_confounder_network(ax, G, pos, edge_list_1, edge_list_2, arc_rad,
                            alpha_1, alpha_2, edge_labels, title):
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edge_list_1,
                           connectionstyle=f'arc3, rad = {arc_rad}', alpha=alpha_1, arrowsize=25)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edge_list_2,
                           connectionstyle=f'arc3, rad = {arc_rad}', alpha=alpha_2, arrowsize=25)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_color='red')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='orange', alpha=1.0)
    latex_labels = {n: f'$y_{n[-1]}$' for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, ax=ax, labels=latex_labels)
    ax.set_title(title, fontsize=12, fontweight="bold")
