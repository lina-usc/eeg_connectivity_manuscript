#!/usr/bin/env python
# coding: utf-8

import gc
import time

import mne
import numpy as np
import xarray as xr
from mne.coreg import Coregistration
from mne.minimum_norm import apply_inverse_epochs
from spectral_connectivity import Connectivity, Multitaper

from utils.experimental import filter_labels_with_vertices

subjects = ['sub-032304', 'sub-032302', 'sub-032307', 'sub-032310', 'sub-032312']
subjects_dir = "/work/erikc/EEG_CON/inspected"
conditions = ['14', '16', '18', '20']
conditions2 = ['EC']

start_time = time.time()

for subject in subjects:
    src = mne.setup_source_space(subject, add_dist="patch", subjects_dir=subjects_dir)
    conductivity = (0.3, 0.006, 0.3)
    model = mne.make_bem_model(subject=subject, conductivity=conductivity,
                               subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    labels_parc = mne.read_labels_from_annot(subject, parc='aparc',
                                              subjects_dir=subjects_dir)
    filtered_labels = filter_labels_with_vertices(labels_parc, src)

    for condition2 in conditions2:
        raw = mne.io.read_raw_eeglab(
            f"/work/erikc/EEG_CON/eeg/{subject}/{subject}_{condition2}.set"
        )
        coreg = Coregistration(raw.info, subject, subjects_dir, fiducials="estimated")

        for condition in conditions:
            epochs = mne.make_fixed_length_epochs(raw, duration=float(condition),
                                                  preload=False)
            epochs.set_eeg_reference(projection=True)
            epochs.apply_baseline((None, None))

            fwd = mne.make_forward_solution(
                epochs.info, trans=coreg.trans, src=src, bem=bem, verbose=True
            )
            cov = mne.compute_covariance(epochs)
            inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov,
                                                          verbose=True)

            stc = apply_inverse_epochs(
                epochs, inv, 1.0 / 3.0 ** 2, method="sLORETA",
                pick_ori=None, verbose=True, return_generator=False,
            )
            label_ts = mne.extract_label_time_course(
                stc, filtered_labels, src, mode='auto',
                return_generator=False, allow_empty=False,
            )

            n = len(epochs)
            mats = []
            for i in range(100):
                inds = np.random.choice(range(n), int(n / 2), replace=False)
                m = Multitaper(
                    time_series=np.array(label_ts)[inds].transpose(2, 0, 1),
                    sampling_frequency=250,
                )
                c = Connectivity(fourier_coefficients=m.fft(),
                                  frequencies=m.frequencies, time=m.time)
                con = getattr(c, 'direct_directed_transfer_function')()
                mat = np.nan_to_num(con.squeeze())
                mats.append(mat[64:112, :, :])
                frequencies = c.frequencies
                del m, c, con
                gc.collect()

            region = [label.name for label in filtered_labels]
            freqs = list(frequencies[64:112])
            bootstrap_samples = list(range(100))

            xarray = xr.DataArray(
                np.array(mats),
                dims=["bootstrap_samples", "frequencies", "region1", "region2"],
                coords={
                    "bootstrap_samples": bootstrap_samples,
                    "frequencies": freqs,
                    "region1": region,
                    "region2": region,
                },
            )
            xarray.to_netcdf(f'{subject}_array_ddtf_{condition}_{condition2}.nc')

print(time.time() - start_time)
