#!/usr/bin/env python
# coding: utf-8

import gc
import time

import mne
import numpy as np
import pandas as pd
from mne.coreg import Coregistration
from mne.minimum_norm import apply_inverse_epochs
from mne_connectivity import spectral_connectivity_epochs
from spectral_connectivity import Connectivity, Multitaper

from utils.experimental import filter_labels_with_vertices

methods_kramer = [
    'pairwise_spectral_granger_prediction',
    'generalized_partial_directed_coherence',
    'direct_directed_transfer_function',
    'directed_transfer_function',
    'directed_coherence',
    'partial_directed_coherence',
]

subjects_dir = "/scratch/MPI-LEMON/freesurfer/subjects/inspected"
subjects = ['sub-032304', 'sub-032302', 'sub-032306', 'sub-032307', 'sub-032310']

time_start = time.time()

all_dat_dict = {}

for subject in subjects:
    src = mne.setup_source_space(subject, add_dist="patch", subjects_dir=subjects_dir)
    raw = mne.io.read_raw_eeglab(
        f"/scratch/MPI-LEMON/freesurfer/subjects/eeg/{subject}/{subject}_EC.set"
    )

    coreg = Coregistration(raw.info, subject, subjects_dir, fiducials="estimated")
    conductivity = (0.3, 0.006, 0.3)
    model = mne.make_bem_model(subject=subject, conductivity=conductivity,
                               subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)

    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, preload=False)
    epochs.set_eeg_reference(projection=True)
    epochs.apply_baseline((None, None))

    fwd = mne.make_forward_solution(epochs.info, trans=coreg.trans, src=src,
                                     bem=bem, verbose=True)
    cov = mne.compute_covariance(epochs)
    inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, verbose=True)

    stc = apply_inverse_epochs(
        epochs, inv, 1.0 / 3.0 ** 2, method="sLORETA",
        pick_ori=None, verbose=True, return_generator=False,
    )

    labels_parc = mne.read_labels_from_annot(subject, parc='aparc',
                                              subjects_dir=subjects_dir)
    filtered_labels = filter_labels_with_vertices(labels_parc, src)
    label_ts = mne.extract_label_time_course(
        stc, filtered_labels, src, mode='auto',
        return_generator=False, allow_empty=False,
    )

    kramer_dat = []
    for method in methods_kramer:
        m = Multitaper(time_series=np.array(label_ts).transpose(2, 0, 1),
                       sampling_frequency=250)
        c = Connectivity(fourier_coefficients=m.fft(), frequencies=m.frequencies,
                         time=m.time)
        con = getattr(c, method)()
        kramer_dat.append(np.nan_to_num(con.squeeze().mean(0), 0))
        del m, c, con
        gc.collect()

    all_dat_dict[subject] = kramer_dat

time_end = time.time()
print(time_end - time_start)


# EFFECTIVE CONNECTIVITY: mean across subjects

mean_arrays = [
    np.array([all_dat_dict[s][m].ravel() for s in subjects]).mean(axis=0)
    for m in range(len(methods_kramer))
]

corr_df = np.round(
    pd.DataFrame(np.array(mean_arrays).T, columns=methods_kramer).corr(), 3
)
corr_df.to_csv('/scratch/figures/corr_df_eff.csv')
