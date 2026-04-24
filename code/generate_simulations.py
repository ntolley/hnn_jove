# Authors
# Nicholas Tolley <nicholas_tolley@brown.edu>

import sys
sys.path.append('../code')

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA

from hnn_core.batch_simulate import BatchSimulate
from hnn_core import jones_2009_model, pick_connection, calcium_model, read_params, read_network_configuration
from hnn_core.network_models import add_erp_drives_to_jones_model
from sbi import utils

import pandas as pd
import seaborn as sns
import torch
import os
from scipy.stats import norm

# Set random seed for reproducibility
seed = 123
torch.manual_seed(seed)

def set_params(param_values, net):
    """
    Set parameters for the network drives.

    Parameters
    ----------
    param_values : dict
        Dictionary of parameter values.
    net : instance of Network
        Network to be updated
    """

    # Feedforward synchronization
    net.external_drives['evprox1']['dynamics']['sigma'] *= param_values['ff_sync_scale']

    # Dendritic Km
    for sec_name, sec in net.cell_types['L5_pyramidal']['cell_object'].sections.items():
        if sec_name != 'soma':
            sec.mechs['km']['gbar_km'] *= (10 ** param_values['km_scale'])

    # Inhibitory gain
    conn_indices = pick_connection(net, receptor=['gabab'])
    for conn_idx in conn_indices:
        net.connectivity[conn_idx]['nc_dict']['A_weight'] *= 10 ** param_values['inh_gain_scale']

    # FB gain
    conn_indices = pick_connection(net, src_gids='evdist1')
    for conn_idx in conn_indices:
        net.connectivity[conn_idx]['nc_dict']['A_weight'] *= 10 ** param_values['fb_gain_scale']


if __name__ == "__main__":
    """This script is associated with Step 6 of the JoVE protocol and is used to generate
    a dataset of parameter samples and simulations from a prior distribution.
    
    This dataset is then used in `notebooks/drug_moa_sbi_ppc.ipynb` to train a neural
    density estimator that produces predictions on parameters that can account
    for 2 different ERP waveforms.
    """

    # Set the number of jobs to be equal to or less than the number of cores on your machine.
    n_jobs = 50

    # Replace with save path specific to your file system.
    save_path = '/oscar/data/sjones/ntolley/hnn_jove/jones_2009_jove'

    # Set random seed for reproducibility, this will be used to generate parameter samples
    rng = np.random.default_rng(seed=123)

    num_sims = 10000 # Number of sample from prior distribution
    tstop = 250 # Simulation run time in milliseconds
    dt = 0.025 # Simulation time step


    # Define a prior distribution over the hypothesed post-treatment mechanisms
    # ff_sync_scale is parameterized as a multiplicative scaling factor of the pre-treatment parameter value.
    # All other parameters are are parameterized as a multiplicative scaling factor on a log scale.
    min_val, max_val = -1, 1
    theta_train_dict = {
        'ff_sync_scale': rng.uniform(0, 5, num_sims), # Thalamocortical Synchony
        'km_scale': rng.uniform(min_val, max_val, num_sims), # Dendritic potassium conductance
        'inh_gain_scale': rng.uniform(min_val, max_val, num_sims), # Local GABAB connectivity
        'fb_gain_scale': rng.uniform(min_val, max_val, num_sims) # Corticocortical strength
    }

    # Initialize the network model from the optimized pre-treatment parameter set
    net_base = read_network_configuration('../data/opt_baseline_config_correlation_best.json')

    # Instantiate the BatchSimulate object to handle running and saving large trainng batches
    batch_simulation = BatchSimulate(net=net_base,
                                    set_params=set_params,
                                    save_outputs=True,
                                    save_dpl=True,
                                    tstop=tstop,
                                    dt=dt,
                                    save_folder=save_path,
                                    overwrite=True,
                                    clear_cache=True)

    # Simulate the samples from the prior distribution and save the results.
    _ = batch_simulation.run(theta_train_dict,
                            n_jobs=n_jobs,
                            combinations=False,
                            backend='loky',
                            verbose=False)

