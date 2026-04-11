import numpy as np
import os
from scipy.integrate import odeint
from scipy import signal
from scipy.stats import wasserstein_distance
import torch
import glob
from functools import partial
# from dask_jobqueue import SLURMCluster
# import dask
# from distributed import Client
from sbi import utils as sbi_utils
from sbi import analysis as sbi_analysis
from sbi import inference as sbi_inference
from sklearn.decomposition import PCA
import scipy

from pyknos.nflows.flows import Flow
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

from hnn_core import jones_2009_model, simulate_dipole, pick_connection
from hnn_core.cells_default import _linear_g_at_dist, _exp_g_at_dist, pyramidal
from hnn_core.params import _short_name
rng_seed = 123
rng = np.random.default_rng(rng_seed)
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

device = 'cpu'
num_cores = 256

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

    # Proximal 1 drive
    weights_ampa_p1 = {'L2_basket': 0.997291, 'L2_pyramidal':0.990722,'L5_basket':0.614932, 'L5_pyramidal': 0.004153}
    weights_nmda_p1 = {'L2_basket': 0.984337, 'L2_pyramidal':1.714247,'L5_basket':0.061868, 'L5_pyramidal': 0.010042}
    synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1,'L5_basket': 1., 'L5_pyramidal': 1.}
    net.add_evoked_drive('evprox1', mu=54.897936, sigma=5.401034 * param_values['ff_gain_scale'], numspikes=1, weights_ampa=weights_ampa_p1, weights_nmda=weights_nmda_p1, location='proximal',synaptic_delays=synaptic_delays_prox, event_seed=seed_array[0])
    # Distal drive
    weights_ampa_d1 = {'L2_basket': 0.624131, 'L2_pyramidal': 0.606619, 'L5_pyramidal':0.258}
    weights_nmda_d1 = {'L2_basket': 0.95291, 'L2_pyramidal': 0.242383, 'L5_pyramidal': 0.156725}
    synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1, 'L5_pyramidal': 0.1}
    net.add_evoked_drive('evdist1', mu=82.9915, sigma=13.208408, numspikes=1, weights_ampa=weights_ampa_d1,weights_nmda=weights_nmda_d1, location='distal',synaptic_delays=synaptic_delays_d1, event_seed=seed_array[1]) 
    # Second proximal evoked drive.
    weights_ampa_p2 = {'L2_basket': 0.758537, 'L2_pyramidal': 0.854454,'L5_basket': 0.979846, 'L5_pyramidal': 0.012483}
    weights_nmda_p2 = {'L2_basket': 0.851444, 'L2_pyramidal':0.067491 ,'L5_basket': 0.901834, 'L5_pyramidal': 0.003818}
    net.add_evoked_drive('evprox2', mu=161.306837, sigma=19.843986, numspikes=1, weights_ampa=weights_ampa_p2,weights_nmda= weights_nmda_p2, location='proximal',synaptic_delays=synaptic_delays_prox, event_seed=seed_array[2])
    

    # Dendritic Km
    for sec_name, sec in net.cell_types['L5_pyramidal']['cell_object'].sections.items():
        if sec_name != 'soma':
            sec.mechs['km']['gbar_km'] *= (10 ** param_values['km_scale'])

    # Inhibitory gain sweep
    conn_indices = pick_connection(net, receptor=['gabab'])
    for conn_idx in conn_indices:
        net.connectivity[conn_idx]['nc_dict']['A_weight'] *= 10 ** param_values['inh_gain_scale']


def run_hnn_sim(net, param_function, prior_dict, theta_samples, tstop, save_path, save_suffix):
    """Run parallel HNN simulations using Dask distributed interface
    
    Parameters
    ----------
    net: Network object
    
    param_function: function definition
        Function which accepts theta_dict and updates simulation parameters 
    prior_dict: dict 
        Dictionary storing information to map uniform sampled parameters to prior distribution.
        Form of {'param_name': {'bounds': (lower_bound, upper_bound), 'scale_func': callable}}.
    theta_samples: array-like
        Unscaled paramter values in range of (0,1) sampled from prior distribution
    tstop: int
        Simulation stop time (ms)
    save_path: str
        Location to store simulations. Must have subdirectories 'sbi_sims/' and 'temp/'
    save_suffix: str
        Name appended to end of output files
    """
    
    # create simulator object, rescale function transforms (0,1) to range specified in prior_dict    
    simulator = partial(simulator_hnn, prior_dict=prior_dict, param_function=param_function,
                        network_model=net, tstop=tstop, return_objects=True)
    # Generate simulations
    seq_list = list()
    num_sims = theta_samples.shape[0]
    step_size = num_cores
    
    for i in range(0, num_sims, step_size):
        seq = list(range(i, i + step_size))
        if i + step_size < theta_samples.shape[0]:
            batch(simulator, seq, theta_samples[i:i + step_size, :], save_path)
        else:
            seq = list(range(i, theta_samples.shape[0]))
            batch(simulator, seq, theta_samples[i:, :], save_path)
        seq_list.append(seq)
        
    # Load simulations into single array, save output, and remove small small files
    dpl_files = [f'{save_path}/temp/dpl_temp{seq[0]}-{seq[-1]}.npy' for seq in seq_list]
    spike_times_files = [f'{save_path}/temp/spike_times_temp{seq[0]}-{seq[-1]}.npy' for seq in seq_list]
    spike_gids_files = [f'{save_path}/temp/spike_gids_temp{seq[0]}-{seq[-1]}.npy' for seq in seq_list]
    theta_files = [f'{save_path}/temp/theta_temp{seq[0]}-{seq[-1]}.npy' for seq in seq_list]

    dpl_orig, spike_times_orig, spike_gids_orig, theta_orig = load_prerun_simulations(
        dpl_files, spike_times_files, spike_gids_files, theta_files)
    
    dpl_name = f'{save_path}/sbi_sims/dpl_{save_suffix}.npy'
    spike_times_name = f'{save_path}/sbi_sims/spike_times_{save_suffix}.npy'
    spike_gids_name = f'{save_path}/sbi_sims/spike_gids_{save_suffix}.npy'
    theta_name = f'{save_path}/sbi_sims/theta_{save_suffix}.npy'
    
    np.save(dpl_name, dpl_orig)
    np.save(spike_times_name, spike_times_orig)
    np.save(spike_gids_name, spike_gids_orig)
    np.save(theta_name, theta_orig)

    files = glob.glob(str(save_path) + '/temp/*')
    for f in files:
        os.remove(f) 

def start_cluster():
    """Reserve SLURM resources using Dask Distributed interface"""
     # Set up cluster and reserve resources
    cluster = SLURMCluster(
        cores=32, processes=32, queue='compute', memory="256GB", walltime="10:00:00",
        job_extra=['-A csd403', '--nodes=1'], log_directory=os.getcwd() + '/slurm_out')

    client = Client(cluster)
    client.upload_file('utils.py')
    print(client.dashboard_link)
    
    client.cluster.scale(num_cores)


def linear_scale_forward(value, bounds, constrain_value=True):
    """Scale value in range (0,1) to range bounds"""
    if constrain_value:
        assert np.all(value >= 0.0) and np.all(value <= 1.0)
        
    assert isinstance(bounds, tuple)
    assert bounds[0] < bounds[1]
    
    return (bounds[0] + (value * (bounds[1] - bounds[0]))).astype(float)

def linear_scale_array(value, bounds, constrain_value=True):
    """Scale columns of array according to bounds"""
    assert value.shape[1] == len(bounds)
    return np.vstack(
        [linear_scale_forward(value[:, idx], bounds[idx], constrain_value) for 
         idx in range(len(bounds))]).T

def log_scale_forward(value, bounds, constrain_value=True):
    """log scale value in range (0,1) to range bounds in base 10"""
    rescaled_value = linear_scale_forward(value, bounds, constrain_value)
    
    return 10**rescaled_value

def log_scale_array(value, bounds, constrain_value=True):
    """log scale columns of array according to bounds in base 10"""
    assert value.shape[1] == len(bounds)
    return np.vstack(
        [log_scale_forward(value[:, idx], bounds[idx], constrain_value) for 
         idx in range(len(bounds))]).T