import sys
sys.path.append('../code')

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA

from hnn_core.batch_simulate import BatchSimulate
from hnn_core import jones_2009_model, pick_connection, calcium_model, read_params
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

    # Kohl et al. ERP drives
    # ----------------------
    # # Proximal 1 drive
    # weights_ampa_p1 = {'L2_basket': 0.997291, 'L2_pyramidal':0.990722,'L5_basket':0.614932, 'L5_pyramidal': 0.004153}
    # weights_nmda_p1 = {'L2_basket': 0.984337, 'L2_pyramidal':1.714247,'L5_basket':0.061868, 'L5_pyramidal': 0.010042}
    # synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,'L5_basket': 1., 'L5_pyramidal': 1.}
    # net.add_evoked_drive('evprox1', mu=54.897936, sigma=5.401034 * param_values['ff_sync_scale'], numspikes=1, weights_ampa=weights_ampa_p1, weights_nmda=weights_nmda_p1, location='proximal',synaptic_delays=synaptic_delays_prox, event_seed=seed_array[0])
    # # Distal drive
    # weights_ampa_d1 = {'L2_basket': 0.624131, 'L2_pyramidal': 0.606619, 'L5_pyramidal':0.258}
    # weights_nmda_d1 = {'L2_basket': 0.95291, 'L2_pyramidal': 0.242383, 'L5_pyramidal': 0.156725}
    # synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1, 'L5_pyramidal': 0.1}
    # net.add_evoked_drive('evdist1', mu=82.9915, sigma=13.208408, numspikes=1, weights_ampa=weights_ampa_d1,weights_nmda=weights_nmda_d1, location='distal',synaptic_delays=synaptic_delays_d1, event_seed=seed_array[1]) 
    # # Second proximal evoked drive.
    # weights_ampa_p2 = {'L2_basket': 0.758537, 'L2_pyramidal': 0.854454,'L5_basket': 0.979846, 'L5_pyramidal': 0.012483}
    # weights_nmda_p2 = {'L2_basket': 0.851444, 'L2_pyramidal':0.067491 ,'L5_basket': 0.901834, 'L5_pyramidal': 0.003818}
    # net.add_evoked_drive('evprox2', mu=161.306837, sigma=19.843986, numspikes=1, weights_ampa=weights_ampa_p2,weights_nmda= weights_nmda_p2, location='proximal',synaptic_delays=synaptic_delays_prox, event_seed=seed_array[2])

    # Deafault HNN drives
    weights_ampa_d1 = {"L2_basket": 0.006562, "L2_pyramidal": 7e-6, "L5_pyramidal": 0.142300,
    }
    weights_nmda_d1 = {"L2_basket": 0.019482, "L2_pyramidal": 0.004317, "L5_pyramidal": 0.080074,
    }
    synaptic_delays_d1 = {"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1}
    net.add_evoked_drive("evdist1", mu=63.53, sigma=3.85 * param_values['ff_sync_scale'], numspikes=1, weights_ampa=weights_ampa_d1,
        weights_nmda=weights_nmda_d1, location="distal", synaptic_delays=synaptic_delays_d1, event_seed=seed_array[0])

    # Add proximal drives
    weights_ampa_p1 = {"L2_basket": 0.08831, "L2_pyramidal": 0.01525, "L5_basket": 0.19934, "L5_pyramidal": 0.00865}
    synaptic_delays_prox = {"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5_basket": 1.0, "L5_pyramidal": 1.0,}
    net.add_evoked_drive("evprox1", mu=26.61, sigma=2.47, numspikes=1, weights_ampa=weights_ampa_p1,
        weights_nmda=None, location="proximal", synaptic_delays=synaptic_delays_prox, event_seed=seed_array[1])

    weights_ampa_p2 = {"L2_basket": 0.000003, "L2_pyramidal": 1.438840, "L5_basket": 0.008958, "L5_pyramidal": 0.684013,}
    net.add_evoked_drive("evprox2", mu=137.12, sigma=8.33, numspikes=1, weights_ampa=weights_ampa_p2, location="proximal",
        synaptic_delays=synaptic_delays_prox, event_seed=seed_array[2])

    # Dendritic Km
    for sec_name, sec in net.cell_types['L5_pyramidal']['cell_object'].sections.items():
        if sec_name != 'soma':
            sec.mechs['km']['gbar_km'] *= (10 ** param_values['km_scale'])

    # Inhibitory gain sweep
    conn_indices = pick_connection(net, receptor=['gabab'])
    for conn_idx in conn_indices:
        net.connectivity[conn_idx]['nc_dict']['A_weight'] *= 10 ** param_values['inh_gain_scale']

def set_params_with_nmda(param_values, net):
    """
    Set parameters for the network drives.

    Parameters
    ----------
    param_values : dict
        Dictionary of parameter values.
    net : instance of Network
        Network to be updated
    """

    set_params(param_values, net)

    # Inhibitory gain sweep
    conn_indices = pick_connection(net, src_gids=['L2_pyramidal', 'L5_pyramidal'], receptor=['nmda'])
    for conn_idx in conn_indices:
        net.connectivity[conn_idx]['nc_dict']['A_weight'] *= 10 ** param_values['nmda_gain_scale']


def set_params_with_fbgain(param_values, net):
    """
    Set parameters for the network drives.

    Parameters
    ----------
    param_values : dict
        Dictionary of parameter values.
    net : instance of Network
        Network to be updated
    """

    set_params(param_values, net)

    # FF gain sweep
    conn_indices = pick_connection(net, src_gids='evdist1')
    for conn_idx in conn_indices:
        net.connectivity[conn_idx]['nc_dict']['A_weight'] *= 10 ** param_values['fb_gain_scale']

if __name__ == "__main__":

    # The number of cores may need modifying depending on your current machine.
    n_jobs = 50

    save_path = '/oscar/data/sjones/ntolley/hnn_jove/jones_2009_drug_with_fbgain'


    rng = np.random.default_rng(seed=123)

    num_sims = 10000
    tstop = 200


    min_val, max_val = -1, 1
    theta_train_dict = {
        'ff_sync_scale': rng.uniform(0, 5, num_sims),
        'km_scale': rng.uniform(min_val, max_val, num_sims),
        'inh_gain_scale': rng.uniform(min_val, max_val, num_sims),
        'fb_gain_scale': rng.uniform(min_val, max_val, num_sims)
    }

    # Initialize the network model and run the batch simulation.
    # l_contra_params = read_params('../data/L_Contra.param')
    # net_base = calcium_model(params=l_contra_params, add_drives_from_params=False)
    net_base = jones_2009_model()

    # Class to handle running and saving large trainng batches
    batch_simulation = BatchSimulate(net=net_base,
                                    set_params=set_params_with_fbgain,
                                    save_outputs=True,
                                    save_dpl=True,
                                    tstop=tstop,
                                    dt=0.5,
                                    save_folder=save_path,
                                    overwrite=True,
                                    clear_cache=True)

    _ = batch_simulation.run(theta_train_dict,
                            n_jobs=n_jobs,
                            combinations=False,
                            backend='loky')

