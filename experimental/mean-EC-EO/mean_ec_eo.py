# -*- coding: utf-8 -*-

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from utils.experimental import bootstrap_ci, remove_outliers


def error_bars(data, n_iter=1000):
    n = len(data)
    means = sorted(
        np.mean(np.array(data)[np.random.choice(n, n, replace=True)])
        for _ in range(n_iter)
    )
    return means[24], means[974]


methods = ['coh', 'ciplv', 'imcoh', 'wpli2']
# methods = ['gpdc', 'ddtf', 'psgp']

path_nc_files = '/scratch/srishyla'
vn = ['fusiform-rh', 'fusiform-lh', 'lingual-rh', 'lingual-lh',
      'cuneus-rh', 'cuneus-lh', 'lateraloccipital-rh', 'lateraloccipital-lh']


def _load_mean_per_subject(methods, condition):
    means_dict = {}
    for method in methods:
        xarrays = glob.glob(f"{path_nc_files}/{method}/*_{method}_{condition}.nc")
        subjects = [arr.split('/')[4].split('_')[0] for arr in xarrays]
        combined = xr.concat(
            [xr.open_dataarray(a) for a in xarrays],
            pd.Index(subjects, name="subjects"),
        )
        combined_vn = combined.sel(region1=vn, region2=vn)
        means_dict[method] = [
            float(combined_vn.sel(subjects=s).mean(
                dim=["bootstrap_samples", "region1", "region2", "frequencies"]
            ).values)
            for s in subjects
        ]
    return means_dict


means_dict_ec = _load_mean_per_subject(methods, 'EC')
means_dict_eo = _load_mean_per_subject(methods, 'EO')


# CALCULATING EFFECT SIZES (COHEN'S D)

cohens_d_dict = {}
for method in methods:
    n_ec = len(means_dict_ec[method])
    std_ec = np.std(means_dict_ec[method])
    mean_ec = np.mean(means_dict_ec[method])
    n_eo = len(means_dict_eo[method])
    std_eo = np.std(means_dict_eo[method])
    mean_eo = np.mean(means_dict_eo[method])

    pooled_sd = np.sqrt(
        ((n_ec - 1) * std_ec ** 2 + (n_eo - 1) * std_eo ** 2) / (n_ec + n_eo - 2)
    )
    cohens_d_dict[method] = (mean_ec - mean_eo) / pooled_sd

print(cohens_d_dict)


# REMOVE OUTLIERS

for method in methods:
    means_dict_ec[method] = remove_outliers(means_dict_ec[method])
    means_dict_eo[method] = remove_outliers(means_dict_eo[method])


# 95% CONFIDENCE INTERVALS

ci_dict = {method: bootstrap_ci(means_dict_ec[method], means_dict_eo[method])
           for method in methods}
print(ci_dict)


# ERROR BARS FOR GRAPHING

graph_ci_eo_dict = {}
graph_ci_ec_dict = {}
for method in methods:
    lo, hi = error_bars(means_dict_eo[method])
    graph_ci_eo_dict[method] = hi - lo
    lo, hi = error_bars(means_dict_ec[method])
    graph_ci_ec_dict[method] = hi - lo


# GRAPHING

fig, axes = plt.subplots(2, 2, figsize=(10, 6))

for method, ax in zip(methods, axes.ravel()):
    ax.bar(0, np.mean(means_dict_ec[method]), yerr=graph_ci_ec_dict[method])
    ax.bar(1, np.mean(means_dict_eo[method]), yerr=graph_ci_eo_dict[method])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(('EC', 'EO'), fontsize=12)

axes[0][0].set_title('Coh', fontweight='bold')
axes[0][1].set_title('ciPLV', fontweight='bold')
axes[1][0].set_title('imCoh', fontweight='bold')
axes[1][1].set_title('dwPLI', fontweight='bold')

fig.supylabel('Mean connectivity', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig('/scratch/srishyla/figures/real_func_means.png', dpi=300)
