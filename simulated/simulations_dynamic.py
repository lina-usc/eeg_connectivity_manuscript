#!/usr/bin/env python
# coding: utf-8

import pathlib
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np

from utils.constants import (
    all_methods, comparison_pairs, confounder_list,
    directed_methods, undirected_methods,
)
from utils.plotting import (
    draw_confounder_network, plot_bar_from_dict, setup_graph_layout,
)
from utils.simulation import estimate_connectivity
from utils.statistics import (
    build_ci_dataframe, compute_bootstrap_mse_corr, compute_ci_dict,
)

ROOT = pathlib.Path(__file__).parent.parent
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
CACHE_FILE = ROOT / "cache" / "simulations_dynamic.pkl"
CACHE_FILE.parent.mkdir(exist_ok=True)


# ### COMPILING ESTIMATE AND GROUND-TRUTH DICTIONARIES

np.random.seed(42)

if CACHE_FILE.exists():
    print("Loading cached simulation results...")
    with open(CACHE_FILE, "rb") as f:
        overall_estimate_dict, overall_ground_truth_dict, mse_dict, corr_coef_dict = pickle.load(f)
else:
    start_time = time.time()

    overall_estimate_dict = {}
    overall_ground_truth_dict = {}

    for confounder in confounder_list:

        confounder_estimate_dict = {}
        confounder_ground_truth_dict = {}

        for method in all_methods:

            estimate_dict = {}
            ground_truth_dict = {}

            if method in undirected_methods:
                matrix_indices = list(zip([1, 2, 2], [0, 0, 1]))
            else:
                matrix_indices = list(zip([1, 2, 2, 0, 0, 1], [0, 0, 1, 1, 2, 2]))

            for i, j in matrix_indices:

                estimate_list = []
                ground_truth_list = []

                for r in range(100):
                    estimated_matrix, ground_truth_matrix = estimate_connectivity(method, confounder, dynamic=True)
                    estimate_list.append(estimated_matrix[i][j])
                    ground_truth_list.append(ground_truth_matrix[i][j])

                label = ("zero" if ground_truth_matrix[i][j] == 0 else "non-zero") + f"_{i}_{j}"
                estimate_dict[label] = estimate_list
                ground_truth_dict[label] = ground_truth_list

            confounder_estimate_dict[method] = estimate_dict
            confounder_ground_truth_dict[method] = ground_truth_dict

        overall_estimate_dict[confounder] = confounder_estimate_dict
        overall_ground_truth_dict[confounder] = confounder_ground_truth_dict

    print(time.time() - start_time)

    # ## CALCULATING MSE AND CORR_COEF

    mse_dict, corr_coef_dict = compute_bootstrap_mse_corr(
        overall_estimate_dict, overall_ground_truth_dict, confounder_list, all_methods
    )

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(
            (overall_estimate_dict, overall_ground_truth_dict, mse_dict, corr_coef_dict), f
        )


# ## GRAPHS - MSE (FUNCTIONAL)

fig, axes = plt.subplots(4, 3, figsize=(10, 10))

G, pos, edge_list_1, edge_list_2, arc_rad = setup_graph_layout()
draw_confounder_network(axes[0][0], G, pos, edge_list_1, edge_list_2, arc_rad,
                        [0, 0, 0], [0, 1, 1],
                        {('y1', 'y0'): 0, ('y2', 'y1'): 1, ('y2', 'y0'): 1},
                        "Common input")
draw_confounder_network(axes[0][1], G, pos, edge_list_1, edge_list_2, arc_rad,
                        [0, 0, 0], [1, 1, 0],
                        {('y1', 'y0'): 1, ('y2', 'y1'): 1, ('y2', 'y0'): 0},
                        "Indirect connections")
draw_confounder_network(axes[0][2], G, pos, edge_list_1, edge_list_2, arc_rad,
                        [0, 0, 0], [0, 0, 1],
                        {('y1', 'y0'): 0, ('y2', 'y1'): 0, ('y2', 'y0'): 1},
                        "Volume conduction")

plot_bar_from_dict("common_input", axes[1][0], mse_dict, "zero_1_0",
                   undirected_methods, ["Coh", "ciPLV", "imCoh", "dwPLI"], "(zero)")
plot_bar_from_dict("common_input", axes[2][0], mse_dict, "non-zero_2_0",
                   undirected_methods, ["Coh", "ciPLV", "imCoh", "dwPLI"], "(non-zero)")
plot_bar_from_dict("common_input", axes[3][0], mse_dict, "non-zero_2_1",
                   undirected_methods, ["Coh", "ciPLV", "imCoh", "dwPLI"], "(non-zero)")

plot_bar_from_dict("indirect_connections", axes[1][1], mse_dict, "non-zero_1_0",
                   undirected_methods, ["Coh", "ciPLV", "imCoh", "dwPLI"],
                   "Node y1 ↔ Node y0 \n (non-zero)")
plot_bar_from_dict("indirect_connections", axes[2][1], mse_dict, "zero_2_0",
                   undirected_methods, ["Coh", "ciPLV", "imCoh", "dwPLI"],
                   "Node y2 ↔ Node y0 \n (zero)")
plot_bar_from_dict("indirect_connections", axes[3][1], mse_dict, "non-zero_2_1",
                   undirected_methods, ["Coh", "ciPLV", "imCoh", "dwPLI"],
                   "Node y2 ↔ Node y1 \n (non-zero)")

plot_bar_from_dict("volume_conduction", axes[1][2], mse_dict, "zero_1_0",
                   undirected_methods, ["Coh", "ciPLV", "imCoh", "dwPLI"], "(zero)")
plot_bar_from_dict("volume_conduction", axes[2][2], mse_dict, "non-zero_2_0",
                   undirected_methods, ["Coh", "ciPLV", "imCoh", "dwPLI"], "(non-zero)")
plot_bar_from_dict("volume_conduction", axes[3][2], mse_dict, "zero_2_1",
                   undirected_methods, ["Coh", "ciPLV", "imCoh", "dwPLI"], "(zero)")

fig.supylabel("Mean-squared error (MSE)", fontsize=12, fontweight="bold")
fig.suptitle("          Dynamic system", fontweight="bold", fontsize=14)
fig.tight_layout()
plt.savefig(FIGURES_DIR / "mse_func_dyn.png", dpi=300)


# ## GRAPHS - MSE (EFFECTIVE)

fig, axes = plt.subplots(7, 3, figsize=(10, 15))

G, pos, edge_list_1, edge_list_2, arc_rad = setup_graph_layout()
draw_confounder_network(axes[0][0], G, pos, edge_list_1, edge_list_2, arc_rad,
                        [0, 0, 0], [0, 1, 1],
                        {('y1', 'y0'): 0, ('y2', 'y1'): 1, ('y2', 'y0'): 1},
                        "Common input")
draw_confounder_network(axes[0][1], G, pos, edge_list_1, edge_list_2, arc_rad,
                        [0, 0, 0], [1, 1, 0],
                        {('y1', 'y0'): 1, ('y2', 'y1'): 1, ('y2', 'y0'): 0},
                        "Indirect connections")
draw_confounder_network(axes[0][2], G, pos, edge_list_1, edge_list_2, arc_rad,
                        [0, 0, 0], [0, 0, 1],
                        {('y1', 'y0'): 0, ('y2', 'y1'): 0, ('y2', 'y0'): 1},
                        "Volume conduction")

plot_bar_from_dict("common_input", axes[1][0], mse_dict, 'zero_1_0',
                   directed_methods, ["gPDC", "dDTF", "pSGP"], "(zero)")
plot_bar_from_dict("common_input", axes[2][0], mse_dict, 'zero_0_1',
                   directed_methods, ["gPDC", "dDTF", "pSGP"], "(zero)")
plot_bar_from_dict("common_input", axes[3][0], mse_dict, 'zero_0_2',
                   directed_methods, ["gPDC", "dDTF", "pSGP"], "(zero)")
plot_bar_from_dict("common_input", axes[4][0], mse_dict, 'non-zero_2_0',
                   directed_methods, ["gPDC", "dDTF", "pSGP"], "(non-zero)")
plot_bar_from_dict("common_input", axes[5][0], mse_dict, 'zero_1_2',
                   directed_methods, ["gPDC", "dDTF", "pSGP"], "(zero)")
plot_bar_from_dict("common_input", axes[6][0], mse_dict, 'non-zero_2_1',
                   directed_methods, ["gPDC", "dDTF", "pSGP"], "(non-zero)")

plot_bar_from_dict("indirect_connections", axes[1][1], mse_dict, 'non-zero_1_0',
                   directed_methods, ["gPDC", "dDTF", "pSGP"],
                   "Node y1 → Node y0 \n (non-zero)")
plot_bar_from_dict("indirect_connections", axes[2][1], mse_dict, 'zero_0_1',
                   directed_methods, ["gPDC", "dDTF", "pSGP"],
                   "Node y0 → Node y1 \n (zero)")
plot_bar_from_dict("indirect_connections", axes[3][1], mse_dict, 'zero_0_2',
                   directed_methods, ["gPDC", "dDTF", "pSGP"],
                   "Node y0 → Node y2 \n (zero)")
plot_bar_from_dict("indirect_connections", axes[4][1], mse_dict, 'zero_2_0',
                   directed_methods, ["gPDC", "dDTF", "pSGP"],
                   "Node y2 → Node y0 \n (zero)")
plot_bar_from_dict("indirect_connections", axes[5][1], mse_dict, 'zero_1_2',
                   directed_methods, ["gPDC", "dDTF", "pSGP"],
                   "Node y1 → Node y2 \n (zero)")
plot_bar_from_dict("indirect_connections", axes[6][1], mse_dict, 'non-zero_2_1',
                   directed_methods, ["gPDC", "dDTF", "pSGP"],
                   "Node y2 → Node y1 \n (non-zero)")

plot_bar_from_dict("volume_conduction", axes[1][2], mse_dict, 'zero_1_0',
                   directed_methods, ["gPDC", "dDTF", "pSGP"], "(zero)")
plot_bar_from_dict("volume_conduction", axes[2][2], mse_dict, 'zero_0_1',
                   directed_methods, ["gPDC", "dDTF", "pSGP"], "(zero)")
plot_bar_from_dict("volume_conduction", axes[3][2], mse_dict, 'zero_0_2',
                   directed_methods, ["gPDC", "dDTF", "pSGP"], "(zero)")
plot_bar_from_dict("volume_conduction", axes[4][2], mse_dict, 'non-zero_2_0',
                   directed_methods, ["gPDC", "dDTF", "pSGP"], "(non-zero)")
plot_bar_from_dict("volume_conduction", axes[5][2], mse_dict, 'zero_1_2',
                   directed_methods, ["gPDC", "dDTF", "pSGP"], "(zero)")
plot_bar_from_dict("volume_conduction", axes[6][2], mse_dict, 'zero_2_1',
                   directed_methods, ["gPDC", "dDTF", "pSGP"], "(zero)")

fig.suptitle("          Dynamic system \n", fontweight="bold", fontsize=14)
fig.supylabel("Mean-squared error (MSE)", fontsize=12, fontweight="bold")
fig.tight_layout()
plt.savefig(FIGURES_DIR / "mse_eff_dyn.png", dpi=300)


# ## GRAPHS - CORR COEF (FUNCTIONAL)

fig, axes = plt.subplots(5, 1, figsize=(4, 9))

plot_bar_from_dict("common_input", axes[0], corr_coef_dict, 'non-zero_2_0',
                   undirected_methods, ["Coh", "ciPLV", "imCoh", "dwPLI"],
                   "Node y2 ↔ Node y0 \n (non-zero)")
plot_bar_from_dict("common_input", axes[1], corr_coef_dict, 'non-zero_2_1',
                   undirected_methods, ["Coh", "ciPLV", "imCoh", "dwPLI"],
                   "Node y2 ↔ Node y1 \n (non-zero)")
plot_bar_from_dict("indirect_connections", axes[2], corr_coef_dict, 'non-zero_1_0',
                   undirected_methods, ["Coh", "ciPLV", "imCoh", "dwPLI"],
                   "Node y1 ↔ Node y0 \n (non-zero)")
plot_bar_from_dict("indirect_connections", axes[3], corr_coef_dict, 'non-zero_2_1',
                   undirected_methods, ["Coh", "ciPLV", "imCoh", "dwPLI"],
                   "Node y2 ↔ Node y1 \n(non-zero)")
plot_bar_from_dict("volume_conduction", axes[4], corr_coef_dict, 'non-zero_2_0',
                   undirected_methods, ["Coh", "ciPLV", "imCoh", "dwPLI"],
                   "Node y2 ↔ Node y0 \n (non-zero)")

axes[0].set_ylabel('Common Input', rotation='horizontal', fontsize=10, loc='top')
axes[2].set_ylabel('Indirect connections', rotation='horizontal', fontsize=10, loc='top')
axes[4].set_ylabel('Volume conduction', rotation='horizontal', fontsize=10, loc='top')

fig.suptitle("                            Dynamic system", fontweight="bold", fontsize=14)
fig.tight_layout()
plt.savefig(FIGURES_DIR / "corr_func_dyn.png", dpi=300)


# ## GRAPHS - CORR COEF (EFFECTIVE)

fig, axes = plt.subplots(5, 1, figsize=(4, 9))

plot_bar_from_dict("common_input", axes[0], corr_coef_dict, 'non-zero_2_0',
                   directed_methods, ["gPDC", "dDTF", "pSGP"],
                   "Node y2 → Node y0 \n (non-zero)")
plot_bar_from_dict("common_input", axes[1], corr_coef_dict, 'non-zero_2_1',
                   directed_methods, ["gPDC", "dDTF", "pSGP"],
                   "Node y2 → Node y1 \n (non-zero)")
plot_bar_from_dict("indirect_connections", axes[2], corr_coef_dict, 'non-zero_1_0',
                   directed_methods, ["gPDC", "dDTF", "pSGP"],
                   "Node y1 → Node y0 \n (non-zero)")
plot_bar_from_dict("indirect_connections", axes[3], corr_coef_dict, 'non-zero_2_1',
                   directed_methods, ["gPDC", "dDTF", "pSGP"],
                   "Node y2 → Node y1 \n(non-zero)")
plot_bar_from_dict("volume_conduction", axes[4], corr_coef_dict, 'non-zero_2_0',
                   directed_methods, ["gPDC", "dDTF", "pSGP"],
                   "Node y2 → Node y0 \n (non-zero)")

axes[0].set_ylabel('Common Input', rotation='horizontal', fontsize=10, loc='top')
axes[2].set_ylabel('Indirect connections', rotation='horizontal', fontsize=10, loc='top')
axes[4].set_ylabel('Volume conduction', rotation='horizontal', fontsize=10, loc='top')

fig.suptitle("                            Dynamic system", fontweight="bold", fontsize=14)
fig.tight_layout()
plt.savefig(FIGURES_DIR / "corr_eff_dyn.png", dpi=300)


# ### 95% CONFIDENCE INTERVALS

ci_dict_mse, ci_dict_corr = compute_ci_dict(
    mse_dict, corr_coef_dict, overall_estimate_dict, confounder_list, comparison_pairs
)

# Functional pairs (indices 0:6)
df_list = build_ci_dataframe(ci_dict_mse, confounder_list, slice(0, 6))
df_list[0] = df_list[0].rename(columns={
    "zero_1_0": "Node 1 ↔ Node 0",
    "non-zero_2_0": "Node 2 ↔ Node 0",
    "non-zero_2_1": "Node 2 ↔ Node 1",
})
df_list[1] = df_list[1].rename(columns={
    "non-zero_1_0": "Node 1 ↔ Node 0",
    "zero_2_0": "Node 2 ↔ Node 0",
    "non-zero_2_1": "Node 2 ↔ Node 1",
})
df_list[2] = df_list[2].rename(columns={
    "zero_1_0": "Node 1 ↔ Node 0",
    "non-zero_2_0": "Node 2 ↔ Node 0",
    "zero_2_1": "Node 2 ↔ Node 1",
})

# Effective pairs (indices 6:)
df_list = build_ci_dataframe(ci_dict_mse, confounder_list, slice(6, None))
directed_rename = {
    "zero_1_0": "Node 1 → Node 0",
    "non-zero_2_0": "Node 2 → Node 0",
    "non-zero_2_1": "Node 2 → Node 1",
    "zero_0_1": "Node 0 → Node 1",
    "zero_0_2": "Node 0 → Node 2",
    "zero_1_2": "Node 1 → Node 2",
}
df_list[0] = df_list[0].rename(columns=directed_rename)
df_list[1] = df_list[1].rename(columns={
    **directed_rename,
    "non-zero_1_0": "Node 1 → Node 0",
    "zero_2_0": "Node 2 → Node 0",
})
df_list[2] = df_list[2].rename(columns={
    **directed_rename,
    "zero_2_1": "Node 2 → Node 1",
})
