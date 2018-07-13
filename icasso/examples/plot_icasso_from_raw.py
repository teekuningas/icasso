"""
.. _tut_icasso:

Use Icasso to compute and validate ICA on MEG data
============================================

ICA is fit to MEG raw data multiple times, and the performance is then visually inspected. Finally the components are retrieved as centroids of the most robust clusters.
"""
# Authors: Erkka Heinila <erkka.heinila@jyu.fi>
#
# License: BSD (3-clause)

import numpy as np

import mne
import logging

from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.datasets import sample

from icasso import Icasso

###############################################################################
# Set up logging
logging.basicConfig(level=logging.INFO)

###############################################################################
# Setup paths and prepare raw data.

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.filter(1, None, fir_design='firwin')  # already lowpassed @ 40

random_state = 10

###############################################################################
# Define parameters for mne's ICA-object and create a Icasso object.

ica_params = {
    'n_components': 20,
    'method': 'fastica',
    'max_iter': 1000,
}
icasso = Icasso(ICA, ica_params=ica_params, iterations=40, 
                bootstrap=True, vary_init=True)

###############################################################################
# Pick good gradiometer channels and set up params for ICA.fit method.
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=False,
                       stim=False, exclude='bads')

fit_params = {
    'picks': picks,
    'decim': 3,
    'reject': dict(mag=4e-12, grad=4000e-13),
    'verbose': 'warning'
}

###############################################################################
# Set up function for getting bootstrapped versions of Raw object.
def bootstrap_fun(raw, generator):
    raw = raw.copy()
    raw._data = np.apply_along_axis(
        func1d=generator.choice,
        axis=1,
        arr=raw._data,
        size=raw._data.shape[1],
        replace=True)
    return raw

###############################################################################
# Set up function to get unmixing matrix from mne's ICA object after fitting.
def unmixing_fun(ica):
    unmixing_matrix = np.dot(ica.unmixing_matrix_, 
                             ica.pca_components_[:ica.n_components_])
    return unmixing_matrix

###############################################################################
# Set up function to store information about individual runs. 
# We do this to get pca mean and pre_whiten information.
def store_fun(ica):
    data = {'pre_whitener': ica.pre_whitener_.T,
            'pca_mean': ica.pca_mean_[np.newaxis, :]}
    return data

###############################################################################
# Fit icasso to raw data.
icasso.fit(data=raw, fit_params=fit_params, random_state=random_state, 
           bootstrap_fun=bootstrap_fun, unmixing_fun=unmixing_fun)

###############################################################################
# Plot a dendogram
icasso.plot_dendrogram()

###############################################################################
# Plot the components in 2D space.
icasso.plot_mds(distance=0.5, random_state=random_state)

###############################################################################
# Unmix using the centroids

# unmixing = icasso.get_centroid_unmixing()
# Must remember here to sort the mean and whitener properly
# pca_mean, pre_whitener = icasso.store[0]['pca_mean'], icasso.store[0]['pre_whitener']
# sources = np.dot(unmixing, (raw._data * pre_whitener) - pca_mean)

###############################################################################
# Create raw object using the icasso centroid sources and plot it

# info = None
# components = mne.io.RawArray(sources, info)
# components.plot()

