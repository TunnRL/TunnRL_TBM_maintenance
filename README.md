# Tunnel automation with Reinforcement Learning - TunnRL-TBM

[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains code for the ongoing project to use RL for optimization of cutter maintenance in hardrock tunnel boring machines.

The first paper on this will be:

_Towards optimized TBM cutter changing policies with reinforcement learning_

by Georg H. Erharter and Tom F. Hansen

published in __Geomechanics and Tunnelling (Vol. 15; Iss 5; October 2022)__

DOI: XXXXXXXXXXXXXXXXXXXXXXXX

## Directory structure

The code framework depends on a certain folder structure. The main functionality is 
in the src folder. Here are mainly two types of files:

- "<A, B, C, D>"_description - scripts to be run
- XX_description - functionality provided to run scripts. 
The set up should be done in the following way:

```
TunnRL_TBM_maintenance
├── checkpoints                           - files from training models
├── experiments                           - logged metrics and config for each experiment using hydra and mlflow
├── graphics                              - saved graphics from running scripts in src
├── install                               - shell scripts to set up environment and Python version with Pyenv and Poetry
├── optimization                          - files from optimization of hyperparameters
├── results                               - study-db files and parameters
│   ├── algorithm_parameters              - optimized hyperparameters for agents
├── src
│   ├── A_main_hydra.py                   - main script to call for optimization, training, execution
│   ├── B_optimization_analyzer.py        - analysing the optuna optimization study
│   ├── C_training_path_analyzer.py       
│   ├── D_recommender.py                  - recommend the next action from a policy (based on a trained agent)
│   ├── XX_config_schemas.py              - schemas for pydantic check of config
│   ├── XX_experiment_factory.py
│   ├── XX_hyperparams.py
│   ├── XX_plotting.py
│   ├── XX_TBM_environment.py             - defining the RL environment and reward function
│   ├── config                            - hierarchical system of config-files utilizing the hydra system
├── .gitignore
├── .flake8                               - setup for linting with flake8
├── .pre-commit-config.yaml               - autoformatting and checks upon commit
├── makefile                              - covenience functionality for file logistics
├── poetry.lock                           - exact version of all dependencies
├── pyproject.toml                        - rules for dependencies and div. settings
├── README.md
```

To clone the repository and have everything set up for you, run:

```bash
git clone https://github.com/TunnRL/TunnRL_TBM_maintenance.git
```

## Requirements

We have organized 2 ways of setting up the environment, downloading and installing all required pacakages, using the same package versions as have been used in development. In this way it is possible to repeat the experiments as close as possible.

1. The recommended way is to use the `poetry` system to set up the environment and install all dependencies. Poetry is stricter on depedencies than conda and define all depedencies in a human readable way through the categorized `pyproject.toml` file. The 
`poetry.lock`defines exact version of all dependencies.

   Make sure you have installed `pyenv` to control your python version. Install the python version and continue. If you don't have poetry and pyenv installed, we have made bash-scripts that installs these in your linux system. **NOTE**: If you haven't got linux you can run linux from windows by activating Window Subsystem for Linux:
   https://learn.microsoft.com/en-us/windows/wsl/install
   
   Run these scripts in your terminal to install:

   ```bash
   install_pyenv.sh
   install_poetry.sh
   ```

   Install the Python version used in this environment. This will take some time:

   ```bash
   pyenv install -v 3.10.5
   ```

   cd into your project dir and activate the Python version:

   ```bash
   pyenv local 3.10.5
   ```

   Check your python version:

   ```bash
   python -V
   ```

   Set up environment and install all depedencies:
   
   ```bash
   poetry install
   ```

   Running this will install all dependencies defined in `poetry.lock`.

   Activate the environment with
   
   ```bash
   poetry shell
   ```

   Then you are ready to run your Python scripts in the exact same system setup as it 
   has been developed!

2. Another way is to use `conda`.

   Create an environment called `rl_cutter` using `environment.yaml` with the help of `conda`. If you get pip errors, install pip libraries manually, e.g. `pip install pandas`

   ```bash
   conda env create --file environment.yaml
   ```

   Activate the new environment with:

   ```bash
   conda activate rl_cutter
   ```

## Sqlite for Optuna optimization of parameters in parallell

To use hyperparameter functionality you need to have the database engine
`SQlite` installed. This is by default installed in Linux, but not in Windows.

Sqlite make it possible to have one common study-file for optimization that a number of terminal-sessions (utilizing all the cores on a computer) or computers can access at the same time. This makes it possible to run optimization of hyperparameters in parallell, greatly speeding up the process, which in reinforcement learning is computationally demanding.

Simply kick off a number of similar runs with the same study-name and all processes will update the same study-db.

## Principles for training an RL-agent

We use the quality controlled implementation of RL-agents in Stable Baselines 3 (implemented in Pytorch). In setting up the customized RL-environment we follow the 
API from Open AI gym by inheriting our custom environment from `gym.env`.

The basic principles for training an agent follow these steps (functionality included in scripts):

1. Instantiate the environment: `env = CustomEnv(...)`. This lays out the state of the
cutters defined in a state vector of cutter life for all cutters, initially assigned with a max life of 1. Another vector defines the penetration value for all steps in the episode.
2. Instantiate the agent with: `agent = PPO(...)` (a number of different agents are defined)
3. Instantiate callbacks with: `callback = CallbackList([eval_cb, custom_callback]`.
Callbacks are not a part of the actual training process but provides functionality for 
logging metrics, early stopping etc.
4. Train the agent by looping over episodes, thereby making new actions and changin the state, each time with a new version of the 
environment. This functionality is wrapped into the learn function in Stable Baselines3.
`agent.learn(total_timesteps=self.EPISODES * self.MAX_STROKES, callback=callback)`.

In every episode (say 10 000 episodes), the agent takes (loops over) a number of steps
(which is TBM-strokes in this environment, eg. 1000 strokes of 1.8 meter). In each step a MLP-neural network is trained to match the best actions to the given state, ie. that maximize the reward. The MLP's are used in different ways for the different agent 
architectures: PPO, A2C, DDPG, TD3, SAC.
This training session is a classic NN-machine learning session looping over a number of
epochs (eg 10 epochs) in order to minimize the loss-function.


## How to use the functionality - in general terms

1. Choose an agent architecture (PPO, DDPG, TD3 etc.) and run an optimization process with Optuna to optimize hyperparameters to achieve the highest reward for that architecture.
   - Optimization data is saved in the `optimization`directory and a subdirectory for each model run. Data is updated in this subdirectory for every chosen episode interval (eg. every 100 episode in a 10 000 episode study).
   - Each time one model-run is completed, common data-files for all experiments are saved into the `results`directory. Run `B_optimization_analyzer.py` to visualize this data.
2. Train an agent for a number of episodes for a certain architecture and parameters given from an Optuna optimization for that architecture.
   - Metrics are saved into the `checkpoint`directory.
   - Visualize the training process with `C_training_paty_analyzer.py`
3. Execute to execute the actions for a trained agent.
   - To recommend the actions (cutter maintenance) for the next step (stroke) use the policy from a trained agent and run `D_recommender.py`.


## Hydra functionality

The hydra system logs and organize all config values, without touching the code itself. Config values can be altered in the yaml-files directly in the config-dir, or overridden in the terminal as described below.

You can now run different reinforcement learning functions by defining different
inputs to A_main_hydra.py in the terminal. Eg.

```bash
python src/A_main_hydra.py EXP.MODE=optimization EXP.DEBUG=True
```

or

```bash
python src/A_main_hydra.py agent=ppo EXP.MODE=training EXP.EPISODES=5000
```


To see all options, run:

```bash
python src/A_main_hydra.py --help
```

With hydra you can also kick of multiruns with different configs. Eg.
using  `--multirun` or shorter `-m` to run models with different cutterheads of 4 and 2 meters.

```bash
python src/A_main_hydra.py -m agent=ppo TBM.CUTTERHEAD_RADIUS=4,2
```


Invoke tab completion with hydra by running:

```bash
python src/A_main_hydra.py -sc install=bash
```

## MLflow

Use mlflow to track all parameters and metrics for each experiment-runs, thereby making it possible to easily compare results from different experiments.

Invoke the MLflow webinterface by (click the link that appears):

```bash
cd experiments
mlflow ui
```
