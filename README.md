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
Steps 5-7 of the protocol are reccomended to be completed using the Python interface to the HNN model.

Jupyter notebooks are provided  in `/notebooks` to reproduce the main data figures of Tolley et al. 2026 (link TBD). These notebooks depend on outputs of 2 computationally expensive python scripts found in the `/code`. Data generated for both of these scripts are available for download at the associated [Open Science Framework repository](https://osf.io/q4udw)
