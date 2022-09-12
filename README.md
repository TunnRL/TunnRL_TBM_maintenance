# Tunnel automation with Reinforcement Learning - TunnRL-TBM

[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains code for the ongoing project to use RL for optimization of cutter maintenance in hardrock tunnel boring machines.

The first paper on this will be:

_Towards optimized TBM cutter changing policies with reinforcement learning_

by Georg H. Erharter and Tom F. Hansen

published in __Geomechanics and Tunnelling (Vol. 15; Iss 5; October 2022)__

DOI: XXXXXXXXXXXXXXXXXXXXXXXX

## Directory structure

The code framework depends on a certain folder structure. The python files should be placed in the main directory. The set up should be done in the following way:
```
TunnRL_TBM_maintenance
├── checkpoints
├── graphics
├── optimization
├── results
├── src
│   ├── A_main.py
│   ├── B_optimization_analyzer.py
│   ├── C_action_analyzer.py
│   ├── D_training_path_analyzer.py
│   ├── XX_hyperparams.py
│   ├── XX_maintenance_lib.py
├── .gitignore
├── poetry.lock
├── pyproject.toml
├── README.md
```

To clone the repository and have everything set up for you, run:

```bash
git clone <url to repository>
```

## Requirements

We have organized 2 ways of setting up the environment, downloading and installing all required pacakages, using the same package versions as have been used in development. In this way it is possible to repeat the experiments as close as possible.

1. The recommended way is to use the `poetry` system set up the environment and install all dependencies. Poetry is stricter on depedencies than conda and define all depedencies in a human readable way through the categorized `pyproject.toml`file.

   Set up environment and install all depedencies:
   
   ```bash
   poetry install
   ```

   Running this will install all dependencies defined in `poetry.lock`.

   Activate the environment with
   
   ```bash
   poetry shell
   ```

2. Another way is to use `conda`.

   Create an environment called `rl_cutter` using `environment.yaml` with the help of `conda`. If you get pip errors, install pip libraries manually, e.g. `pip install pandas`

   ```bash
   conda env create --file environment.yaml
   ```

   Activate the new environment with:

   ```bash
   conda activate rl_cutter
   ```

**NOTE**: To use hyperparameter functionality you need to have the database engine
`SQlite` installed. This is by default installed in Linux, but not in Windows.

## Principles for training an RL-agent

We use the quality controlled implementation of RL-agents in Stable Baselines 3 (implemented in Pytorch). In setting up the customized RL-environment we follow the 
API from Open AI gym by inheriting our custom environment from `gym.env`.

The basic principles for training an agent follow these steps:

1. Instantiate the environment: `env = CustomEnv(...)`
2. Instantiate the agent with: `agent = PPO(...)`
3. Instantiate callbacks with: `callback = CallbackList([eval_cb, custom_callback]`.
Callbacks are not a part of the actual training process but provides functionality for 
logging metrics, early stopping etc.
4. Train the agent by looping over episodes, thereby making new actions and changin the state, each time with a new version of the 
environment. This functionality is wrapped into the learn function in Stable Baselines3.
`agent.learn(total_timesteps=self.EPISODES * self.MAX_STROKES, callback=callback)`.

In every episode (say 10 000 episodes), the agent takes (loops over) a number of steps
(which is TBM-strokes in this environment, eg. 1000 strokes of 1.8 meter). In each step a MLP-neural network is trained to match the best actions to the given state, ie. that maximize the reward.
This training session is a classic NN-machine learning session looping over a number of
epochs (eg 10 epochs) in order to minimize the loss-function.


## How to use the functionality

1. Choose an agent architecture (PPO, DDPG, TD3 etc.) and run an optimization process with Optuna to optimize hyperparameters to achieve the highest reward for that architecture.
   - Optimization data is saved in the `optimization`directory and a subdirectory for each model run. Data is updated in this subdirectory for every chosen episode interval (eg. every 100 episode in a 10 000 episode study).
   - Each time one model run is completed, common data-files for all experiments are saved into the `results`directory. Run `B_optimization_analyzer.py` to visualize this data.
2. Train an agent for a number of episodes for a certain architecture and parameters given from an Optuna optimization for that architecture.
   - Metrics are saved into the `checkpoint`directory.
3. Execute to execute the actions for a trained agent.


## Hydra functionality
Invoke tab completion with hydra by running:

```bash
python src/A_main_hydra.py -sc install=bash
```
