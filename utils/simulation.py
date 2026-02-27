import random

import mne
import numpy as np
from mne_connectivity import spectral_connectivity_epochs
from spectral_connectivity import Connectivity, Multitaper

from utils.constants import directed_methods


def get_ground_truth_dict():
    c_2_0 = np.random.uniform(0.2, 1)
    c_2_1 = np.random.uniform(0.2, 1)
    c_1_0 = np.random.uniform(0.2, 1)

    return {
        'common_input': np.array([[0, 0, 0],
                                   [0, 0, 0],
                                   [c_2_0, c_2_1, 0]]),
        'indirect_connections': np.array([[0, 0, 0],
                                          [c_1_0, 0, 0],
                                          [0, c_2_1, 0]]),
        'volume_conduction': np.array([[0, 0, 0],
                                       [0, 0, 0],
                                       [c_2_0, 0, 0]]),
    }


def simulate_confounder(confounder, dynamic=False):
    sfreq = 250
    time = np.arange(0, 100, 1 / sfreq)

    f0 = np.random.randint(1, 41)
    f1 = np.random.randint(1, 41)
    f2 = np.random.randint(1, 41)

    noise0 = np.random.randn(len(time))
    noise1 = np.random.randn(len(time))
    noise2 = np.random.randn(len(time))

    alpha0 = np.random.uniform(0.01, 0.5)
    alpha1 = np.random.uniform(0.01, 0.5)
    alpha2 = np.random.uniform(0.01, 0.5)

    snr0 = 1 / alpha0
    snr1 = 1 / alpha1
    snr2 = 1 / alpha2

    c_2_0 = get_ground_truth_dict()[confounder][2][0]
    c_2_1 = get_ground_truth_dict()[confounder][2][1]
    c_0_1 = get_ground_truth_dict()[confounder][1][0]

    if dynamic:
        rho0 = round(random.uniform(0.5, 1.0), 2)
        rho1 = round(random.uniform(0.5, 1.0), 2)
        rho2 = round(random.uniform(0.5, 1.0), 2)

        y0 = np.zeros(len(time))
        y1 = np.zeros(len(time))
        y2 = np.zeros(len(time))

        for i in np.arange(2, len(time)):
            y2[i] = rho2 * y2[i-1] - (rho2 ** 2) * y2[i-2] + noise2[i] * snr2
            y0[i] = rho0 * y0[i-1] - (rho0 ** 2) * y0[i-2] - c_2_0 * y2[i-1] + noise0[i] * snr0
            y1[i] = rho1 * y1[i-1] - (rho1 ** 2) * y1[i-2] - c_0_1 * y0[i-3] - c_2_1 * y2[i-2] + noise1[i] * snr1
    else:
        signal0 = np.sin(2 * np.pi * f0 * time)
        signal1 = np.sin(2 * np.pi * f1 * time)
        signal2 = np.sin(2 * np.pi * f2 * time)

        phase_2_0 = np.random.uniform(0, 2) * np.pi
        phase_2_1 = np.random.uniform(0, 2) * np.pi
        phase_0_1 = np.random.uniform(0, 2) * np.pi

        lag2_0 = np.int64(np.round(phase_2_0 / (2 * np.pi * f0) * f0))
        lag2_1 = np.int64(np.round(phase_2_1 / (2 * np.pi * f1) * f1))
        lag0_1 = np.int64(np.round(phase_0_1 / (2 * np.pi * f2) * f2))

        y2 = signal2 + noise2 * snr2
        y0 = signal0 + noise0 * snr0 + c_2_0 * np.roll(y2, lag2_0)
        y1 = signal1 + noise1 * snr1 + c_2_1 * np.roll(y2, lag2_1) + c_0_1 * np.roll(y0, lag0_1)

    if confounder == 'volume_conduction':
        L = np.random.uniform(-1, 1, size=(3, 3))
    else:
        L = np.identity(3)

    z0, z1, z2 = L @ np.array([y0, y1, y2])
    return {"f0": f0, "f1": f1, "f2": f2, "signals": [z0, z1, z2]}


def make_mne_raw(signals):
    sim_data = np.array(signals)
    info = mne.create_info(ch_names=["ch_0", "ch_1", "ch_2"], ch_types=["eeg"] * 3, sfreq=250)
    return mne.io.RawArray(sim_data, info)


def normalize(mat):
    return (mat - mat.min()) / (mat.max() - mat.min())


def estimate_connectivity(method, confounder, dynamic=False):
    confounder_simulated = simulate_confounder(confounder, dynamic=dynamic)
    f0 = confounder_simulated["f0"]
    f1 = confounder_simulated["f1"]
    f2 = confounder_simulated["f2"]
    delta = 1

    epochs = mne.make_fixed_length_epochs(make_mne_raw(confounder_simulated["signals"]), duration=1)

    if method == "imaginary_coherence" or method in directed_methods:
        m = Multitaper(time_series=np.array(epochs.get_data()).transpose(2, 0, 1), sampling_frequency=250)
        c = Connectivity(fourier_coefficients=m.fft(), frequencies=m.frequencies, time=m.time)
        con = getattr(c, method)()

        if dynamic:
            con_mat = np.nan_to_num(con.squeeze())[1:41].transpose(1, 2, 0)
            return normalize(con_mat).mean(axis=2)
        else:
            f0_con = np.nan_to_num(con.squeeze())[f0-delta:f0+delta].transpose(1, 2, 0)
            f1_con = np.nan_to_num(con.squeeze())[f1-delta:f1+delta].transpose(1, 2, 0)
            f2_con = np.nan_to_num(con.squeeze())[f2-delta:f2+delta].transpose(1, 2, 0)
            con_mat = np.array([f0_con, f1_con, f2_con])
            return normalize(con_mat).mean(axis=3).mean(axis=2)
    else:
        if dynamic:
            mne_con = spectral_connectivity_epochs(
                epochs, method=method, sfreq=250, fmin=1, fmax=41, fskip=0,
                faverage=True, mt_low_bias=True, block_size=1000, n_jobs=1, verbose=None)
        else:
            mne_con = spectral_connectivity_epochs(
                epochs, method=method, sfreq=250,
                fmin=(f0-delta, f1-delta, f2-delta),
                fmax=(f0+delta, f1+delta, f2+delta),
                fskip=0, faverage=True, mt_low_bias=True, block_size=1000, n_jobs=1, verbose=None)

        con_mat = mne_con.get_data(output="dense")
        return normalize(con_mat).mean(2)
