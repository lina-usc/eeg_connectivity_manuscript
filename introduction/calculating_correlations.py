#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mne
from mne.coreg import Coregistration
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
import numpy as np
from pathlib import Path
import pandas as pd
import os
from pathlib import Path
from mne_connectivity import spectral_connectivity_epochs
import xarray as xr
from spectral_connectivity import Multitaper, Connectivity
import gc
import seaborn as sns
from matplotlib import pyplot as plt
import time


# In[2]:


def filter_labels_with_vertices(labels_parc, src):
    # Get the vertices from both hemispheres in the source space
    src_vertices = [src[0]['vertno'], src[1]['vertno']]
    
    # Initialize an empty list to hold valid labels
    valid_labels = []
    
    for label in labels_parc:
        # Determine the hemisphere index: 0 for 'lh' and 1 for 'rh'
        hemi_idx = 0 if label.hemi == 'lh' else 1
        
        # Check if any of the label's vertices are in the source space for that hemisphere
        if any(v in src_vertices[hemi_idx] for v in label.vertices):
            valid_labels.append(label)
            
    return valid_labels


# In[10]:


methods_kramer = ['pairwise_spectral_granger_prediction',
                    'generalized_partial_directed_coherence', 
                    'direct_directed_transfer_function', 
                    'directed_transfer_function',
                    'directed_coherence', 
                    'partial_directed_coherence']

#methods = ['coh', 'imcoh', 'cohy', 'plv', 'ciplv', 'ppc', 'pli', 'pli2_unbiased', 'wpli', 'wpli2_debiased']

subjects_dir = "/scratch/MPI-LEMON/freesurfer/subjects/inspected"

subjects = ['sub-032304','sub-032302', 'sub-032306', 'sub-032307', 'sub-032310']


# In[11]:


time_start = time.time()

all_dat_dict = {}

for subject in subjects:
    
    src = mne.setup_source_space(subject, add_dist="patch", subjects_dir=subjects_dir)
    raw = mne.io.read_raw_eeglab(f"/scratch/MPI-LEMON/freesurfer/subjects/eeg/{subject}/{subject}_EC.set")
    
    info = raw.info
    fiducials = "estimated"
    coreg = Coregistration(info, subject, subjects_dir, fiducials=fiducials)
    
    conductivity = (0.3, 0.006, 0.3)
    model = mne.make_bem_model(subject=subject, conductivity=conductivity, subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    
    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, preload=False)
    epochs.set_eeg_reference(projection=True)
    epochs.apply_baseline((None,None))
    fwd = mne.make_forward_solution(
        epochs.info, trans=coreg.trans, src=src, bem=bem, verbose=True
    )
    
    cov = mne.compute_covariance(epochs)
    
    inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, verbose=True)
    
    method = "sLORETA"
    snr = 3.0
    lambda2 = 1.0 / snr**2
    stc = apply_inverse_epochs(
        epochs,
        inv,
        lambda2,
        method=method,
        pick_ori=None,
        verbose=True,
        return_generator=False
    )
    
    labels_parc = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)
    
    filtered_labels = filter_labels_with_vertices(labels_parc, src)
    label_ts = mne.extract_label_time_course(stc, filtered_labels, src, mode='auto', return_generator=False, allow_empty=False)

    #EFFECTIVE CONNECTIVITY METHODS
    
    kramer_dat = []
    for method in methods_kramer:
 
        m = Multitaper(time_series=np.array(label_ts).transpose(2, 0, 1),
                        sampling_frequency=250)
                    
        c = Connectivity(fourier_coefficients=m.fft(),
                frequencies=m.frequencies,
                time=m.time)    
        
        con = getattr(c, method)()
        mat = np.nan_to_num(con.squeeze())
        kramer_dat.append(np.nan_to_num(con.squeeze().mean(0), 0))
        frequencies = c.frequencies
        del m
        del c
        del con
        gc.collect()

    all_dat_dict[subject] = kramer_dat


    """
    #FUNCTIONAL CONNECTIVITY METHODS
    mne_all_dat = []
    for method in methods:
        mne_con = spectral_connectivity_epochs(np.array(label_ts), 
                                           method=method, sfreq=250, mode='multitaper', 
                                           fmin=8, fmax=13, fskip=0, faverage=False,
                                           tmin=None, tmax=None, mt_bandwidth=None, mt_adaptive=False,
                                           mt_low_bias=True, block_size=1000, n_jobs=1, verbose=None)
        mat = np.real(mne_con.get_data(output="dense"))
        mne_all_dat.append((mat.mean(2) + mat.mean(2).T).ravel())

    all_dat_dict[subject] = mne_all_dat

    """

time_end = time.time()
print(time_end - time_start)


# In[ ]:

#EFFECTIVE CONNECTIVITY METHODS
mean_arrays = []
for method in range(len(methods_kramer)):
    mean_array = np.array([all_dat_dict['sub-032304'][method].ravel(), all_dat_dict['sub-032302'][method].ravel(), all_dat_dict['sub-032306'][method].ravel(), all_dat_dict['sub-032307'][method].ravel(), all_dat_dict['sub-032310'][method].ravel()]).mean(axis=0)
    mean_arrays.append(mean_array)



"""
#FUNCTIONAL CONNECTIVITY METHODS
mean_arrays = []
for method in range(len(methods)):
    mean_array = np.array([all_dat_dict['sub-032304'][method], all_dat_dict['sub-032302'][method], all_dat_dict['sub-032306'][method], all_dat_dict['sub-032307'][method], all_dat_dict['sub-032310'][method]]).mean(axis=0)
    mean_arrays.append(mean_array)

"""


# In[26]:

corr_df = np.round(pd.DataFrame(np.array(mean_arrays).T,columns=methods_kramer).corr(),3)
corr_df.to_csv('/scratch/figures/corr_df_eff.csv')


# In[ ]:





# In[ ]:




