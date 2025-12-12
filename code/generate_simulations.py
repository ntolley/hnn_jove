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

    seed_rng = np.random.default_rng(123)
    seed_array = seed_rng.integers(0, 1e5, size=3)

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

    # The number of cores may need modifying depending on your current machine.
    n_jobs = 50

    save_path = '/oscar/data/sjones/ntolley/hnn_jove/jones_2009_jove'

    rng = np.random.default_rng(seed=123)

    num_sims = 10000
    tstop = 250
    dt = 0.025


    min_val, max_val = -1, 1
    theta_train_dict = {
        'ff_sync_scale': rng.uniform(0, 5, num_sims),
        'km_scale': rng.uniform(min_val, max_val, num_sims),
        'inh_gain_scale': rng.uniform(min_val, max_val, num_sims),
        'fb_gain_scale': rng.uniform(min_val, max_val, num_sims)
    }

    # Initialize the network model and run the batch simulation.
    # net_base = jones_2009_model()
    net_base = read_network_configuration('../data/opt_baseline_config_correlation_best.json')

    # Class to handle running and saving large trainng batches
    batch_simulation = BatchSimulate(net=net_base,
                                    set_params=set_params,
                                    save_outputs=True,
                                    save_dpl=True,
                                    tstop=tstop,
                                    dt=dt,
                                    save_folder=save_path,
                                    overwrite=True,
                                    clear_cache=True)

    _ = batch_simulation.run(theta_train_dict,
                            n_jobs=n_jobs,
                            combinations=False,
                            backend='loky')

