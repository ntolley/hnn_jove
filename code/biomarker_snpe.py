import sys
sys.path.append('../code')

from hnn_core.optimization import Optimizer, add_opt_drives, set_params_opt_drives

import os
import os.path as op

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import hnn_core
from hnn_core import (MPIBackend, jones_2009_model, simulate_dipole,
                      read_dipole, JoblibBackend, pick_connection)
from hnn_core.dipole import average_dipoles
from hnn_core.viz import plot_dipole
from hnn_core.batch_simulate import BatchSimulate

import pickle
import numpy as np
import sys

import torch

from sbi.analysis import pairplot
from sbi.inference import NRE_A
from sbi.utils import BoxUniform
from sbi import utils

from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns

import argparse

from scipy.stats import norm

def set_params_drug(param_values, net):
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

# Combine drive and drug update params into one function
def set_params(a, b):
    set_params_opt_drives(b, a)  # Need to flip arguments for now, will be fixed in later HNN-core version
    set_params_drug(a, b)


def get_sims(fpath):
    scale_factor = 1000
    smooth_window = 40
    downsample = 20

    # Load batch simulations used for SBI training
    dpl_list, theta_train = list(), list()
    for fname in os.listdir(fpath):
        if '.npz' in fname:
            res = np.load(f'{fpath}/{fname}', allow_pickle=True)
            for dpl in res['dpl']:
                dpl_list.append(dpl[0].copy().smooth(smooth_window).scale(scale_factor).data['agg'][::downsample])

            for param_dict in res['param_values']:
                theta_train.append(np.array(list(param_dict.values())))

    dpl_list = np.array(dpl_list)
    theta_train = np.array(theta_train)
    return dpl_list, theta_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["baseline", "biomarker"], required=True)

    args = parser.parse_args()

    print(f'Running SNPE on {args.target} ERP')


    # Simulation and inference hyperparameters
    num_sims = 10_000
    n_jobs = 50
    tstop = 250
    dt = 0.025
    num_rounds = 10
    n_components = 10

    # Load in target waveform and optimized parameter set
    dipole_experimental = read_dipole('/users/ntolley/Jones_Lab/hnn_jove/data/L_Contra.txt')

    round = 0
    # save_path = f'/oscar/data/sjones/ntolley/hnn_jove/jones_2009_snpe/{args.target}'
    save_path = f'/oscar/scratch/ntolley/hnn_jove/jones_2009_snpe/{args.target}'
    data_save_path = f'{save_path}/round_{round}'

    figure_path = f'{save_path}/figures'
    posterior_path = f'{save_path}/posteriors'
    os.makedirs(figure_path, exist_ok=True)
    os.makedirs(posterior_path, exist_ok=True)

    net_base = jones_2009_model()
    constraints_wide, initial_params = add_opt_drives(net_base, n_prox=2, n_dist=1)

    opt_fname = '../data/baseline_optimization/opt_baseline_object_correlation_10.pkl'
    with open(opt_fname, 'rb') as file:
        opt_run = pickle.load(file)

    opt_params = opt_run.opt_params_

    percent_change = 0.5
    delta = np.abs(opt_params) * percent_change
    lower_bounds = opt_params - delta
    upper_bounds = opt_params + delta

    # Widen drive parameter bounds centered on optimized baseline
    constraints = {name: (lower_bounds[idx], upper_bounds[idx]) for idx, name in enumerate(constraints_wide.keys())}

    # Add drug mechanism parameters
    min_val, max_val = -1, 1
    constraints['ff_sync_scale'] =  (0, 5)
    constraints['km_scale'] = (min_val, max_val)
    constraints['inh_gain_scale'] = (min_val, max_val)
    constraints['fb_gain_scale'] = (min_val, max_val)

    # Generate initial round to fit PCA:
    bounds = np.array(list(constraints.values()))
    prior = utils.torchutils.BoxUniform(
        low=torch.as_tensor(bounds[:, 0]), high=torch.as_tensor(bounds[:, 1]))

    theta = prior.sample((num_sims,))

    batch_simulation = BatchSimulate(net=net_base,
                                    set_params=set_params,
                                    save_outputs=True,
                                    save_dpl=True,
                                    tstop=tstop,
                                    dt=dt,
                                    save_folder=data_save_path,
                                    overwrite=True,
                                    clear_cache=True)

    theta_dict = {name: theta[:, idx] for idx, name in enumerate(constraints.keys())}

    print('Running initial PCA simulations...', end=' ')
    _ = batch_simulation.run(theta_dict,
                            n_jobs=n_jobs,
                            combinations=False,
                            backend='loky',
                            verbose=False)

    print ('Done!')

    dpl_train, theta_train = get_sims(save_path)

    times = np.linspace(0, tstop, dpl_train.shape[1])

    # Load experimental biomarker
    baseline_cs = CubicSpline(dipole_experimental.times, dipole_experimental.data['agg'])
    baseline_dpl = baseline_cs(times)

    # Create artificial biomarker
    biomarker_scale = 0.5
    gauss = norm.pdf(times, loc=20, scale=200)

    gauss = (gauss / np.max(gauss)) * (biomarker_scale - 1)
    gauss += 1
    biomarker_dpl = baseline_dpl * gauss

    # Select x_cond target for posterior estimation
    if args.target == "baseline":
        target_dpl = baseline_dpl
        target_color = 'C0'
    elif args.target == "biomarker":
        target_dpl = biomarker_dpl
        target_color = 'C3'

    # Fit PCA to training simulations
    print(f'\nRunning PCA')
    pca = PCA(n_components=n_components)
    x_train = pca.fit_transform(dpl_train)
    dpl_transform = pca.inverse_transform(x_train)

    num_dim = len(constraints)
    # The specific observation we want to focus the inference on.
    x_cond = pca.transform(target_dpl.reshape(1,-1))

    inference = NRE_A(prior)

    posteriors = []
    proposal = prior

    for round in range(1, num_rounds + 1):
        print(f'\nSNPE Round {round}/{num_rounds}')
        theta_train = proposal.sample((num_sims,))

        data_save_path = f'{save_path}/round_{round}'

        batch_simulation = BatchSimulate(net=net_base,
                                    set_params=set_params,
                                    save_outputs=True,
                                    save_dpl=True,
                                    tstop=tstop,
                                    dt=dt,
                                    save_folder=data_save_path,
                                    overwrite=True,
                                    clear_cache=True)

        theta_dict = {name: theta_train[:, idx] for idx, name in enumerate(constraints.keys())}
        
        _ = batch_simulation.run(theta_dict,
                                n_jobs=n_jobs,
                                combinations=False,
                                backend='loky',
                                verbose=False)

        dpl_train, theta_train = get_sims(save_path)

        x_train = pca.fit_transform(dpl_train)

        density_estimator = inference.append_simulations(
            torch.tensor(theta_train).float(), torch.tensor(x_train).float()).train()

        posterior = inference.build_posterior(density_estimator)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(x_cond)

        plt.figure()
        _ = plt.plot(dpl_train[:100, :].T, color='k', linewidth=0.5, alpha=0.5)
        _ = plt.plot(target_dpl, color=target_color, linewidth=3)
        plt.xlabel('Samples')
        plt.savefig(f'{figure_path}/ppc_waveforms_round_{round}.png')
        plt.close()

        with open(f'{posterior_path}/posterior_round{round}.pkl', "wb") as handle:
            pickle.dump(posterior, handle)


    # Generate posterior distribution plot
    num_samples = 1000
    label_names = [args.target]
    sample_list, label_list = list(), list()

    for cond_idx in range(x_cond.shape[0]):
        samples = posterior.sample((num_samples,), x=x_cond)
        sample_list.append(samples.numpy())

        label_list.extend(np.repeat(label_names[cond_idx], num_samples))

    sample_list = np.concatenate(sample_list)
    param_labels = list(constraints.keys())

    sample_list = np.array(sample_list)
    df = pd.DataFrame(sample_list)
    df.columns = param_labels
    df['cond'] = label_list

    labelsize = 15

    color_palette = [target_color]

    print('Generating posterior plot')
    g = sns.PairGrid(df, diag_sharey=False, corner=False, hue='cond', palette=color_palette, height=2.5)
    g.map_lower(sns.kdeplot, fill=True, common_norm=False)
    g.map_diag(sns.kdeplot, fill=True)

    for idx in range(len(constraints)):
        g.axes[idx,0].set_ylabel(param_labels[idx], fontsize=labelsize)
        g.axes[idx,0].set_ylim(list(constraints.values())[idx])

        g.axes[len(constraints)-1,idx].set_xlabel(param_labels[idx], fontsize=labelsize)
        g.axes[len(constraints)-1,idx].set_xlim(list(constraints.values())[idx])

    plt.tight_layout()
    plt.savefig(f'{figure_path}/posterior.png')