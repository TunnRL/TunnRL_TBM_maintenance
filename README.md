# Tunnel automation with Reinforcement Learning - TunnRL-TBM

This repository contains code for the ongoing project to use RL for optimization of cutter maintenance in hardrock tunnel boring machines.

The first paper on this will be:

_Towards optimized TBM cutter changing policies with reinforcement learning_

by Georg H. Erharter and Tom F. Hansen

published in __Geomechanics and Tunnelling (Vol. 15; Iss 5; October 2022)__

DOI: XXXXXXXXXXXXXXXXXXXXXXXX

## Two different repo versions

There are implemented two different versions of the repo:

- main branch
- industry_advanced branch

Industry_advanced __has the same RL-functionality__ and plots as the main_branch but has extended functionality and structure for reporting, config and Q&A, mainly with functionality from:
- Mlflow    - for tracking and visualization of all parameters and metrics of all training experiments
- Hydra     - defines, structures and saves all configuration parameters
- Pydantic  - defines schemas and validations for quality checking config-inputs
- Rich      - better visualisation of terminal output
- Pytest    - unity testing of code

This repo implements more advanced programming techniques, and includes software principles such as testing, input-checks and code-formatting, all by facilitating easy runs of code using the terminal.

Switch between the repos by choosing the branch at the top left or by clicking: 
https://github.com/TunnRL/TunnRL_TBM_maintenance/tree/industry_advanced.

## Directory structure

The code framework depends on a certain folder structure. The main functionality is 
in the src folder. Here are mainly two types of files:

- "<A, B, C, D>"_description - scripts to be run
- XX_description - functionality provided to run scripts. 
The set up should be done in the following way:

```
TunnRL_TBM_maintenance
├── checkpoints                           - files from training models
├── graphics                              - saved graphics from running scripts in src
├── install                               - shell scripts to set up environment and Python version with Pyenv and Poetry
├── optimization                          - files from optimization of hyperparameters
├── results                               - study-db files and parameters
│   ├── algorithm_parameters              - optimized hyperparameters for agents
├── src
│   ├── A_main.py                         - main script to call for optimization, training, execution
│   ├── B_optimization_analyzer.py        - analysing the optuna optimization study
│   ├── C_training_path_analyzer.py       
│   ├── D_recommender.py                  - recommend the next action from a policy (based on a trained agent)
│   ├── XX_experiment_factory.py
│   ├── XX_hyperparams.py
│   ├── XX_plotting.py
│   ├── XX_TBM_environment.py             - defining the RL environment and reward function
├── .gitignore
├── makefile                              - covenience functionality for file logistics
├── poetry.lock                           - exact version of all dependencies
├── pyproject.toml                        - rules for dependencies and div. settings
├── environment.yaml                      - dependency file to use with conda
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

   Make sure you have installed `pyenv` to control your python version. Install the python version and continue. If you don't have poetry and pyenv installed, we have made bash-scripts that installs these in your linux system. __NOTE__: If you haven't got linux you can run linux from windows by activating Window Subsystem for Linux:
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

**NOTE**: To use hyperparameter functionality you need to have the database engine
`SQlite` installed. This is by default installed in Linux, but not in Windows.

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

1. In `A_main.py` choose an agent architecture (PPO, DDPG, TD3 etc.) and run an optimization process with Optuna to optimize hyperparameters to achieve the highest reward for that architecture.
   - Optimization data is saved in the `optimization`directory and a subdirectory for each model run. Data is updated in this subdirectory for every chosen episode interval (eg. every 100 episode in a 10 000 episode study).
   - Each time one model-run is completed, common data-files for all experiments are saved into the `results`directory. Run `B_optimization_analyzer.py` to visualize this data.
2. Train an agent for a number of episodes for a certain architecture and parameters given from an Optuna optimization for that architecture.
   - Metrics are saved into the `checkpoint`directory.
   - Visualize the training process with `C_training_paty_analyzer.py`
3. Execute to execute the actions for a trained agent.
   - To recommend the actions (cutter maintenance) for the next step (stroke) use the policy from a trained agent and run `D_recommender.py`.
