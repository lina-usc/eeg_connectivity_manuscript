#!/usr/bin/env python
# coding: utf-8

import time

import numpy as np
from mne_connectivity import spectral_connectivity_epochs

from utils.experimental import run_source_pipeline, save_connectivity_xarray

subjects = ['sub-032304', 'sub-032302', 'sub-032307', 'sub-032310', 'sub-032312']
subjects_dir = "/work/erikc/EEG_CON/inspected"
conditions = ['18']
conditions2 = ['EC', 'EO']

start_time = time.time()

for subject in subjects:
    for condition in conditions:
        for condition2 in conditions2:
            label_ts, filtered_labels, epochs, src = run_source_pipeline(
                subject, subjects_dir,
                f"/work/erikc/EEG_CON/eeg/{subject}/{subject}_{condition2}.set",
                epoch_duration=float(condition),
            )

            n = len(epochs)
            coh_mats = []
            for i in range(100):
                inds = np.random.choice(range(n), int(n / 2), replace=False)
                mne_con = spectral_connectivity_epochs(
                    np.array(label_ts)[inds],
                    method="ciplv", sfreq=250, mode='multitaper',
                    fmin=8, fmax=13, fskip=0, faverage=False,
                    tmin=None, tmax=None, mt_bandwidth=None, mt_adaptive=False,
                    mt_low_bias=True, block_size=1000, n_jobs=1, verbose=None,
                )
                coh_mats.append(np.real(mne_con.get_data(output="dense")))

            save_connectivity_xarray(
                coh_mats, filtered_labels, mne_con.freqs,
                subject, "ciplv", f"{condition}_{condition2}",
            )

print(time.time() - start_time)
