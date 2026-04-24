# Uncovering putative neural mechanisms of neurotherapeutic impacts on EEG using the Human Neocortical Neurosolver

This repository contains code and examples to reproduce Tolley et al., 2026 (link TBD).
<img width="924" height="318" alt="image" src="https://github.com/user-attachments/assets/31eedc41-ef48-4524-a21b-190b799b5d2b" />

### Motivation
A key barrier to developing effective drugs for disorders of the central nervous system (CNS) is understanding their impact on neural circuits. This protocol demonstrates how physics-based neural simulations can be used to interpret electrophysiological biomarkers, providing a mechanistically grounded approach to the development of neurotherapeutics.

The examples in this protocol primarily concern the combination of the python packages [HNN-core](https://github.com/jonescompneurolab/hnn-core) and [SBI](https://github.com/sbi-dev/sbi)

### Installation
We reccomend [pixi]() for the installation of this repository. After installing pixi, enter the following prompts into the terminal
```bash
git clone https://github.com/ntolley/hnn_jove
cd hnn_jove
pixi shell
```

These steps will install 1) install all dependencies for the pixi environment, 2) create a `pixi.lock` file specific to your operating system containing a list of all python depenencies, and 3) activate the newly installed environment.

### GUI examples
Steps 1-4 of the protocol in the paper can be completed entirely in the HNN graphical user interface (GUI). In a terminal with the newly created pixi environment activated (this is indicated by `(hnn_jove)` being visible on the prompt), enter the following to activate the GUI:
```
hnn-gui
```

This command will automatically open a tab in your default browser where you can interact with the GUI.

Several steps in the protocol involve fitting simulations to empirical data. These files can be found in the `data/` folder and are titled `pre-treatment.txt` and `post-treatment.txt`.

### Python API examples
Jupyter notebooks are provided  in `notebooks/` to reproduce the main data figures of Tolley et al. 2026 (link TBD). These notebooks depend on outputs of 2 computationally expensive python scripts found in the `code/`. Specifically `code/baseline_optimization.py` runs parameter optimization to fit the model to the empirical ERP waveform `pre-treatment.txt` (Step 4-5 of the JoVE protocol). This repository is distributed with a parameter set that was output from this script in `data/opt_baseline_config_correlation_best.json`. The notebook files `notebooks/baseline_optimization_figure.ipynb` and `notebooks/posttreatment_optimization_figure.ipynb` both use these optimized files to generate the manuscript figures.

Then `code/generate_simulations.py` uses the optized parameter set to generate a dataset of samples from a prior distribution over "post-treatment parameters of interest" (Step 6 of the JoVE protocol). Data generated from this script is available for download at the associated [Open Science Framework repository](https://osf.io/q4udw). The notebook `notebooks/drug_moa_sbi_ppc.ipynb` uses this dataset to 1) train an neural density estimator with the `sbi` package, and 2) produce a posterior distribution over the parameters of interest for the pre-treatment and hypothetical post-treatment ERP waveforms.


