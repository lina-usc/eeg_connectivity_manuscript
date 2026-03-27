# EEG Connectivity Manuscript

Code accompanying the manuscript evaluating functional and effective EEG connectivity methods under common confounding scenarios.

## Overview

This repository contains analysis scripts for benchmarking spectral EEG connectivity measures. Methods are assessed on their ability to accurately recover ground-truth connection strengths in the presence of three confounders: **common input**, **indirect connections**, and **volume conduction**.

Performance is evaluated using mean squared error (MSE) and Spearman correlation, with 95% bootstrap confidence intervals for pairwise method comparisons.

## Methods Evaluated

**Functional (undirected) connectivity:**
- Coherence (Coh)
- Corrected imaginary Phase Locking Value (ciPLV)
- Imaginary Coherence (imCoh)
- Debiased weighted Phase Lag Index (dwPLI)

**Effective (directed) connectivity:**
- Generalized Partial Directed Coherence (gPDC)
- Direct Directed Transfer Function (dDTF)
- Pairwise Spectral Granger Prediction (pSGP)

## Repository Structure

```
├── simulated/          # Simulation analyses
│   ├── simulations_non_dynamic.py     # Non-dynamic system simulations
│   ├── simulations_dynamic.py         # Dynamic system simulations
│   ├── topologies_non-dynamic.ipynb   # Topology figures (non-dynamic)
│   ├── topologies_dynamic.py          # Topology figures (dynamic)
│   ├── copy_topology_non-dyn.py       # Copy topology analysis
│   └── combined_mse_corr_figs.py      # Combined MSE and correlation figures
│
├── experimental/       # Real EEG data analyses (MPI-LEMON dataset)
│   ├── metrics-py-file/                      # Connectivity metric computation scripts
│   │   ├── coh.py, ciplv.py                  # Functional connectivity computation
│   │   ├── gpdc_revised.ipynb                # gPDC computation
│   │   ├── ddtf_revised.ipynb                # dDTF computation
│   │   ├── psgp_revised.ipynb                # pSGP computation
│   │   ├── imcoh_kramer.ipynb                # imCoh computation
│   │   └── gpu_job_script_template.sh        # HPC job submission template
│   ├── Between-within-variance/              # Between- vs. within-subject variance
│   ├── epoch-length-assessments/             # Effect of epoch length on estimates
│   └── mean-EC-EO/                           # Mean connectivity: Eyes Closed vs. Eyes Open
│
├── introduction/       # Introductory figures
│   ├── confounders_fig.py             # Confounder schematic diagrams
│   └── corr_matrices.py               # Correlation matrix figures
│
└── figures/            # Generated output figures
```

## Simulations

Simulated three-node networks (100 s, 250 Hz, sinusoidal signals with additive noise) are constructed for each confounder scenario. Connectivity methods are estimated over 100 repetitions per condition and evaluated against randomized ground-truth connection strengths. Bootstrap resampling (1000 iterations) is used to compute MSE and Spearman correlation distributions for each method and connection pair.

## Experimental Data

Real EEG analyses use the [MPI-LEMON](https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html) resting-state dataset (Eyes Closed and Eyes Open conditions). Source-space connectivity is computed using sLORETA inverse solutions and parcellated using the Desikan-Killiany atlas.

## Dependencies

- [MNE-Python](https://mne.tools/)
- [mne-connectivity](https://mne.tools/mne-connectivity/)
- [spectral_connectivity](https://github.com/Eden-Kramer-Lab/spectral_connectivity)
- NumPy, SciPy, pandas, seaborn, matplotlib
- NetworkX, xarray
