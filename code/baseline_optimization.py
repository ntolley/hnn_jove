from hnn_core.optimization import Optimizer, add_opt_drives, set_params_opt_drives
import os.path as op

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import hnn_core
from hnn_core import (MPIBackend, jones_2009_model, simulate_dipole,
                      read_dipole, JoblibBackend)
from hnn_core.dipole import average_dipoles
from hnn_core.viz import plot_dipole

import pickle
import numpy as np
import sys


if __name__ == "__main__":
    dipole_experimental = read_dipole('/users/ntolley/Jones_Lab/hnn_jove/data/L_Contra.txt')

    n_trials = 5
    scaling_factor = 1000
    window_length = 40
    tstop = 250
    dt = 0.025
    max_iter = 100
    popsize = 16

    net_base = jones_2009_model()
    constraints, initial_params = add_opt_drives(net_base, n_prox=2, n_dist=1)

    optim = Optimizer(net_base, tstop=tstop, constraints=constraints, solver='cma',
                    set_params=set_params_opt_drives, initial_params=initial_params, max_iter=max_iter, obj_fun='dipole_corr')
                    

    optim.fit(target=dipole_experimental, n_trials=n_trials, scale_factor=scaling_factor,
            smooth_window_len=window_length, dt=dt, popsize=popsize)

        
    job_id = int(sys.argv[1])
    fpath = "/users/ntolley/Jones_Lab/hnn_jove/data/baseline_optimization"

    optim.net_.write_configuration(f'{fpath}/opt_baseline_config_correlation_{job_id}.json')


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

