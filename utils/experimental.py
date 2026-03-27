import mne
import numpy as np
import xarray as xr
from mne.coreg import Coregistration
from mne.minimum_norm import apply_inverse_epochs


def filter_labels_with_vertices(labels_parc, src):
    src_vertices = [src[0]['vertno'], src[1]['vertno']]
    valid_labels = []
    for label in labels_parc:
        hemi_idx = 0 if label.hemi == 'lh' else 1
        if any(v in src_vertices[hemi_idx] for v in label.vertices):
            valid_labels.append(label)
    return valid_labels


def outlier_indices(data):
    q1 = np.percentile(data, 25, method='midpoint')
    q3 = np.percentile(data, 75, method='midpoint')
    iqr = q3 - q1
    upper_outliers = np.where(data >= q3 + 1.5 * iqr)[0]
    lower_outliers = np.where(data <= q1 - 1.5 * iqr)[0]
    return list(upper_outliers) + list(lower_outliers)


def remove_outliers(data):
    outlier_idx = outlier_indices(data)
    return [data[i] for i in range(len(data)) if i not in outlier_idx]


def bootstrap_ci(group1, group2, n_iter=1000):
    n1, n2 = len(group1), len(group2)
    samples1 = [np.array(group1)[np.random.choice(n1, n1, replace=True)] for _ in range(n_iter)]
    samples2 = [np.array(group2)[np.random.choice(n2, n2, replace=True)] for _ in range(n_iter)]
    diff_list = sorted(np.mean(s2) - np.mean(s1) for s1, s2 in zip(samples1, samples2))
    return diff_list[24], diff_list[974]


def run_source_pipeline(subject, subjects_dir, raw_path, epoch_duration=1.0):
    src = mne.setup_source_space(subject, add_dist="patch", subjects_dir=subjects_dir)
    raw = mne.io.read_raw_eeglab(raw_path)

    coreg = Coregistration(raw.info, subject, subjects_dir, fiducials="estimated")

    conductivity = (0.3, 0.006, 0.3)
    model = mne.make_bem_model(subject=subject, conductivity=conductivity, subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)

    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=False)
    epochs.set_eeg_reference(projection=True)
    epochs.apply_baseline((None, None))

    fwd = mne.make_forward_solution(epochs.info, trans=coreg.trans, src=src, bem=bem, verbose=True)
    cov = mne.compute_covariance(epochs)
    inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, verbose=True)

    stc = apply_inverse_epochs(
        epochs, inv, 1.0 / 3.0 ** 2, method="sLORETA",
        pick_ori=None, verbose=True, return_generator=False
    )

    labels_parc = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)
    filtered_labels = filter_labels_with_vertices(labels_parc, src)
    label_ts = mne.extract_label_time_course(
        stc, filtered_labels, src, mode='auto', return_generator=False, allow_empty=False
    )

    return label_ts, filtered_labels, epochs, src


def save_connectivity_xarray(mats, labels, freqs, subject, conn_method, condition, output_path=None):
    region = [label.name for label in labels]
    da = xr.DataArray(
        np.array(mats),
        dims=["bootstrap_samples", "region1", "region2", "frequencies"],
        coords={
            "bootstrap_samples": list(range(len(mats))),
            "region1": region,
            "region2": region,
            "frequencies": list(freqs),
        },
    )
    fname = output_path or f'{subject}_array_{conn_method}_{condition}.nc'
    da.to_netcdf(fname)
