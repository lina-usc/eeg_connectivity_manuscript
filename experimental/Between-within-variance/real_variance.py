#!/usr/bin/env python
# coding: utf-8

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from utils.experimental import bootstrap_ci, remove_outliers

vn = ['fusiform-rh', 'fusiform-lh', 'lingual-lh', 'lingual-rh',
      'cuneus-rh', 'cuneus-lh', 'lateraloccipital-rh', 'lateraloccipital-lh']

# methods = ['coh', 'ciplv', 'imcoh', 'wpli2']
methods = ['gpdc', 'ddtf', 'psgp']  # effective measures


# ASSEMBLING VARIANCE DICTIONARIES

inter_variance_dict = {}
intra_variance_dict = {}

for method in methods:
    subject_files = glob.glob(f'/scratch/srishyla/{method}/*_EC.nc')

    # intra-subject variance
    mean_per_subject = []
    for file in subject_files:
        xarray = xr.open_dataarray(file)
        var_list = []
        for sample in range(100):
            std = xarray.sel(bootstrap_samples=sample, region1=vn, region2=vn).values.std()
            var_list.append(std ** 2)
        mean_per_subject.append(np.mean(var_list))
    intra_variance_dict[method] = mean_per_subject

    # inter-subject variance
    all_bootstraps = []
    for file in subject_files:
        xarray = xr.open_dataarray(file)
        for sample in range(100):
            bootstrap = xarray.sel(bootstrap_samples=sample, region1=vn, region2=vn).values
            all_bootstraps.append(bootstrap)

    n_subjects = len(subject_files)
    random_bootstraps = []
    for i in range(n_subjects):
        index = np.random.choice(range(n_subjects * 100), 100, replace=False)
        random_bootstraps.append(np.array(all_bootstraps)[index, :])

    mean_per_bootstrap = []
    for sample in random_bootstraps:
        var_list = [sample[i].std() ** 2 for i in range(100)]
        mean_per_bootstrap.append(np.mean(var_list))

    inter_variance_dict[method] = mean_per_bootstrap


# REMOVE OUTLIERS

for method in methods:
    inter_variance_dict[method] = remove_outliers(inter_variance_dict[method])
    intra_variance_dict[method] = remove_outliers(intra_variance_dict[method])

for method in methods:
    print(method, len(inter_variance_dict[method]))
for method in methods:
    print(method, len(intra_variance_dict[method]))


# CALCULATING RATIOS

ratio_dict = {}
for method in methods:
    ratio_list = [
        inter_v / intra_v
        for inter_v, intra_v in zip(inter_variance_dict[method], intra_variance_dict[method])
    ]
    ratio_dict[method] = ratio_list


# 95% CONFIDENCE INTERVALS

comparison_pairs = [('gpdc', 'psgp'), ('gpdc', 'ddtf'), ('ddtf', 'psgp')]

ci_dict = {}
for pair in comparison_pairs:
    ci_dict[pair] = bootstrap_ci(ratio_dict[pair[0]], ratio_dict[pair[1]])

print(ci_dict)


# GRAPHING

ratio_df_long = {method: ratio_dict[method] for method in methods}

fig, axes = plt.subplots(1, 1, figsize=(10, 6))
ratio_df = pd.DataFrame(
    [ratio_dict['gpdc'], ratio_dict['ddtf'], ratio_dict['psgp']]
).transpose()
ratio_df.columns = ['gPDC', 'dDTF', 'pSGP']
sns.violinplot(ratio_df, color="orange")
axes.set(ylabel="Between-to-within subject variance")
plt.savefig("/scratch/srishyla/figures/variance_eff_fig.png", dpi=300)
