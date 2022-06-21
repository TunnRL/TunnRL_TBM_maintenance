# Tunnel automation with Reinforcement Learning - TunnRL-TBM

This repository contains code for the ongoing project to use RL for optimization of cutter maintenance in hardrock tunnel boring machines.

The first paper on this will be:

_Towards optimized TBM cutter changing policies with reinforcement learning_

by Georg H. Erharter and Tom F. Hansen

published in __Geomechanics and Tunnelling (Vol. 15; Iss 5; October 2022)__

DOI: XXXXXXXXXXXXXXXXXXXXXXXX

## Requirements and folder structure

Use the `requirements.txt` or `environment.yaml` file to download the required packages to run the code. We recommend using a package management system like conda or pipenv for this purpose. Using conda:

1. Create an environment called `rl_cutter` using `environment.yaml` with the help of `conda`. If you get pip errors, install pip libraries manually, e.g. `pip install pandas`

   ```bash
   conda env create --file environment.yaml
   ```

2. Activate the new environment with:

   ```bash
   conda activate rl_cutter
   ```

The code framework depends on a certain folder structure. The python files should be placed in the main directory. The set up should be done in the following way:
```
TunnRL_TBM_maintenance
├── src
│   ├── checkpoints
│   ├── graphics
│   ├── results
│   ├── A_main.py
│   ├── B_optimization_analyzer.py
│   ├── XX_maintenance_lib.py
```
Either set up the folder structure manually or on Linux run:
```bash
bash make_folder_structure.sh
```
