#!/usr/bin/env python
# coding: utf-8

import os
import time

import numpy as np
from mne_connectivity import spectral_connectivity_epochs

from utils.experimental import run_source_pipeline, save_connectivity_xarray

subjects = [s for s in os.listdir('/scratch/coh_subset') if s.startswith('sub')]
subjects_dir = "/scratch/coh_subset"
conditions = ['EC', 'EO']

start_time = time.time()

for subject in subjects:
    try:
        for condition in conditions:
            label_ts, filtered_labels, epochs, src = run_source_pipeline(
                subject, subjects_dir,
                f"/scratch/MPI-LEMON/freesurfer/subjects/eeg/{subject}/{subject}_{condition}.set",
            )

            n = len(epochs)
            coh_mats = []
            for i in range(100):
                inds = np.random.choice(range(n), int(n / 2), replace=False)
                mne_con = spectral_connectivity_epochs(
                    np.array(label_ts)[inds],
                    method="coh", sfreq=250, mode='multitaper',
                    fmin=8, fmax=13, fskip=0, faverage=False,
                    tmin=None, tmax=None, mt_bandwidth=None, mt_adaptive=False,
                    mt_low_bias=True, block_size=1000, n_jobs=1, verbose=None,
                )
                coh_mats.append(np.real(mne_con.get_data(output="dense")))

            save_connectivity_xarray(coh_mats, filtered_labels, mne_con.freqs,
                                     subject, "coh", condition)
    except Exception:
        continue

print(time.time() - start_time)
