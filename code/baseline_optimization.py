# Authors
# Nicholas Tolley <nicholas_tolley@brown.edu>

from hnn_core.optimization import Optimizer
import os.path as op

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import hnn_core
from hnn_core import (MPIBackend, jones_2009_model, simulate_dipole,
                      read_dipole, JoblibBackend, pick_connection)
from hnn_core.dipole import average_dipoles
from hnn_core.viz import plot_dipole

import pickle
import numpy as np
import sys


from sklearn.decomposition import PCA

def add_opt_drives(net, tstop=200, n_prox=2, n_dist=1):
    prox_cell_type = ["L5_pyramidal", "L5_basket", "L2_pyramidal", "L2_basket"]
    dist_cell_type = ["L5_pyramidal", "L2_pyramidal", "L2_basket"]
    default_range = {
        "mu": (0, tstop),
        "sigma": (0, 20),
        "numspikes": (0, 1),
        "ampa": (-5, 1),
        "nmda": (-5, 1),
    }

    prox_weights = {cell_type: 0.0 for cell_type in prox_cell_type}
    dist_weights = {cell_type: 0.0 for cell_type in dist_cell_type}

    prox_delays = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }
    dist_delays = {"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1}

    constraints, initial_params = dict(), dict()
    # Add proximal drives
    for idx in range(n_prox):
        name = f"evprox{idx + 1}"
        constraints[f"{name}_mu"] = default_range["mu"]
        initial_params[f"{name}_mu"] = np.random.uniform(*default_range["mu"])

        constraints[f"{name}_sigma"] = default_range["sigma"]
        initial_params[f"{name}_sigma"] = np.random.uniform(*default_range["sigma"])

        constraints[f"{name}_numspikes"] = default_range["numspikes"]
        initial_params[f"{name}_numspikes"] = 1

        for cell_type in prox_cell_type:
            constraints[f"{name}_{cell_type}_ampa"] = default_range["ampa"]
            initial_params[f"{name}_{cell_type}_ampa"] = np.random.uniform(
                *default_range["ampa"]
            )

            constraints[f"{name}_{cell_type}_nmda"] = default_range["nmda"]
            initial_params[f"{name}_{cell_type}_nmda"] = np.random.uniform(
                *default_range["nmda"]
            )

        net.add_evoked_drive(
            name,
            mu=0.0,
            sigma=1.0,
            numspikes=1,
            location="proximal",
            weights_ampa=prox_weights,
            weights_nmda=prox_weights,
            synaptic_delays=prox_delays,
        )

    # Add distal drives
    for idx in range(n_dist):
        name = f"evdist{idx + 1}"
        constraints[f"{name}_mu"] = default_range["mu"]
        initial_params[f"{name}_mu"] = np.random.uniform(*default_range["mu"])

        constraints[f"{name}_sigma"] = default_range["sigma"]
        initial_params[f"{name}_sigma"] = np.random.uniform(*default_range["sigma"])

        constraints[f"{name}_numspikes"] = default_range["numspikes"]
        initial_params[f"{name}_numspikes"] = 1

        for cell_type in prox_cell_type:
            constraints[f"{name}_{cell_type}_ampa"] = default_range["ampa"]
            initial_params[f"{name}_{cell_type}_ampa"] = np.random.uniform(
                *default_range["ampa"]
            )

            constraints[f"{name}_{cell_type}_nmda"] = default_range["nmda"]
            initial_params[f"{name}_{cell_type}_nmda"] = np.random.uniform(
                *default_range["nmda"]
            )

        net.add_evoked_drive(
            name,
            mu=0.0,
            sigma=1.0,
            numspikes=1,
            location="distal",
            weights_ampa=dist_weights,
            weights_nmda=dist_weights,
            synaptic_delays=dist_delays,
        )

    return constraints, initial_params


def set_params_opt_drives(net, param_values):
    drive_names = list(net.external_drives.keys())
    for name in drive_names:
        target_cell_types = net.external_drives[name]["target_types"]

        net.external_drives[name]["dynamics"]["mu"] = param_values[f"{name}_mu"]
        net.external_drives[name]["dynamics"]["sigma"] = param_values[f"{name}_sigma"]
        net.external_drives[name]["dynamics"]["numspikes"] = max(
            1, int(np.round(param_values[f"{name}_numspikes"]))
        )

        # reinstate external drives to be able to fill in below
        for receptor in ["ampa", "nmda"]:
            net.external_drives[name][f"weights_{receptor}"] = {
                ct: 0.0 for ct in target_cell_types
            }

        for cell_type in target_cell_types:
            for receptor in ["ampa", "nmda"]:
                conn_idx = pick_connection(
                    net, src_gids=name, target_gids=cell_type, receptor=receptor
                )
                assert len(conn_idx) == 1

              
                net.connectivity[conn_idx[0]]["nc_dict"]["A_weight"] = (
                    10 ** param_values[f"{name}_{cell_type}_{receptor}"]
                )
                net.external_drives[name][f"weights_{receptor}"][cell_type] = (
                    10 ** param_values[f"{name}_{cell_type}_{receptor}"]
                )



if __name__ == "__main__":
    """This script is associated with Step 4 of the JoVE protocol and optimizes parameters of
    extrinsic drives with the default HNN model to produce an empirical
    ERP waveform (i.e. the pre-treatment waveform.)"""

    # 1) Load ERP waveform from CSV file
    # ----------------------------------
    # 1st column must be time in ms, 2nd column must be neural signal (e.g. nAm or or mV)
    dipole_experimental = read_dipole('/users/ntolley/Jones_Lab/hnn_jove/data/pre-treatment.txt')

    # 2) Define optimization hyperparameters
    # --------------------------------------
    n_trials = 5  # Trials run per simulation, optimization loss is calculated on trial averaged waveform
    scaling_factor = 1000  # Relevant for plotting, does not impact optimization with correlations objective function
    window_length = 40  # Window size for smoothing (units as number of samples)
    tstop = 250  # Simulation time in ms
    dt = 0.025  # Time step of differential equation solver
    max_iter = 200  # Maximum iterations of optimizer
    popsize = 16  # Number of parameter sets per epoch (only relevant for CMA-ES)

    obj_fun = "dipole_corr"  # Objective function to maximize correlation coefficient
    solver = "cma"  # Covariance matrix adaption evolutionary strategy (CMA-ES)


    # 2) Build network to be optimized and instantiate optimizer
    # ----------------------------------------------------------
    net_base = jones_2009_model()  # Variable to store the "base" network to be simulated

    # This function adds extrinsic drives to the network in place (2 proximal and 1 distal drive).
    # `constraints` is a dictionary with a unique key for every parameter in the drives that are added.
    # The entries of constraints are tuples (low_val, high_val) defining the optimization bounds.
    # `initial_params` is a dictionary with the same keys as `constraints`
    constraints, initial_params = add_opt_drives(net_base, n_prox=2, n_dist=1)

    # Instantiate the optimization class with the network objects
    # Note: `set_param_opt_drives()` is a function which complements `add_opt_drives()`. It uses the
    # parameter keys in `constraints` to update the corresponding values in the network.
    optim = Optimizer(net_base, tstop=tstop, constraints=constraints, solver=solver,
                      set_params=set_params_opt_drives, initial_params=initial_params, max_iter=max_iter, obj_fun=obj_fun)
                    
    # Run optimization
    optim.fit(target=dipole_experimental, n_trials=n_trials, scale_factor=scaling_factor,
              smooth_window_len=window_length, dt=dt, popsize=popsize)

    # 3) Evalute and save optimization results
    # -----------------------------------------
    job_id = 0
    if len(sys.argv) > 1:
        job_id = int(sys.argv[1])
    fpath = "/users/ntolley/Jones_Lab/hnn_jove/data/baseline_optimization"

    # Save parameters of best fit network
    optim.net_.write_configuration(f'{fpath}/opt_baseline_config_correlation_{job_id}.json')

    # Save optimizer class
    with open(f'{fpath}/opt_baseline_object_correlation_{job_id}.pkl', 'wb') as file:
        pickle.dump(optim, file)


    # Run best fit dipole:
    with JoblibBackend(n_jobs=10):
        dipoles_optimized = simulate_dipole(
                optim.net_, tstop=tstop, n_trials=n_trials, dt=dt)

    # Smooth and scale
    for dipole in dipoles_optimized:
        dipole.smooth(window_length).scale(scaling_factor)

    # Make figures
    labelsize = 13
    ticksize = 10

    # Loss figure
    plt.figure(figsize=(6,4.5))
    plt.plot(optim.obj_, color='k')
    plt.xlabel('Epochs', fontsize=labelsize)
    plt.ylabel('Loss', fontsize=labelsize)
    plt.xticks(fontsize=ticksize)
    plt.xticks(fontsize=ticksize)
    plt.savefig(f'/users/ntolley/Jones_Lab/hnn_jove/figures/baseline_optimization/opt_baseline_loss_{job_id}.png')

    # Dipole figure
    fig, ax = plt.subplots(sharex=True, figsize=(6,4))
    plot_dipole(dipoles_optimized.copy(), ax=ax, layer='agg',
                show=False, color='tab:blue', average=True)
    dipole_experimental.plot(ax=ax, layer='agg', show=False,
                            color='tab:orange')
    # Legend
    legend_handles = [Line2D([0], [0], color='tab:blue', lw=1.0),
                    Line2D([0], [0], color='tab:orange', lw=1.0),
                    Line2D([0], [0], color='tab:green', lw=1.0)]
    ax.legend(legend_handles, ['optimized', 'baseline'])
    plt.title(f'Best loss: {optim.obj_[-1]:.2f}')

    plt.savefig(f'/users/ntolley/Jones_Lab/hnn_jove/figures/baseline_optimization/opt_baseline_dipole_{job_id}.png')

