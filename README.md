# Tunnel automation with Reinforcement Learning - TunnRL-TBM

#TODO: legg til info om begge os + mer om installasjonsskript

[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains code for the ongoing project to use RL for optimization of cutter maintenance in hardrock tunnel boring machines. As mentioned in the Readme for the main branch, this branch contains the __same RL-functionality__ and plots as the main_branch but has extended functionality and structure for reporting, reproducability, config and quality control, mainly with functionality from:

- Mlflow - for tracking and visualization of all parameters and metrics of all training experiments
- Hydra - defines, structures and saves all configuration parameters for all experiments
- Pydantic - defines schemas and validations for quality checking config-inputs
- Rich - enhanced visualisation of terminal output
- Pytest - unity testing of code
- Docker - to run experiments in a reproducible way on a High Performance Computer (HPC) or just to run the experiments in an easy way 😀
- Code formating, checking and linting with: Black and Ruff, Pre-commit

Industry_advanced implements more advanced programming techniques, and includes software principles such as testing, input-checks and code-formatting, all by facilitating easy runs of code using the terminal.


## Directory structure

The code framework depends on a certain folder structure. The main functionality is
in the src folder. Here are mainly two types of files:

- "<A, B, C, D>"_description - scripts to be run
- An `utils` package with library files to be used by scripts


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
│   ├── utils                             - utility package
│        ├── XX_config_schemas.py         - schemas for pydantic check of config
│        ├── XX_experiment_factory.py
│        ├── XX_hyperparams.py
│        ├── XX_plotting.py
│        ├── XX_TBM_environment.py        - defining the RL environment and reward function
│   ├── config                            - hierarchical system of config-files utilizing the hydra system
├── .gitignore
├── Dockerfile                            - Instructions to make a Docker image >>> run the training process using a docker container.
├── .pre-commit-config.yaml               - autoformatting and checks upon commit
├── .python-versions                      - Python version used in development
├── makefile                              - covenience functionality for file logistics
├── poetry.lock                           - exact version of all dependencies
├── pyproject.toml                        - rules for dependencies and div. settings
├── requirements.txt                      - dependency requirements for use by pip or conda
├── README.md
```


## Run

You can run the functionality in this repo in two ways.

1. Clone the repo, install the right Python version, set up the environment, and then finally run scripts the standard way. This way you can also change and inspect the code underway.
2. A quicker way to run the functionality is to pull a built `docker image` from `docker hub`, start a container from the image and run the implemented functionality. A docker container has similiraties with an executable application file, and contain a filesystem, all dependencies, code and hardware settings. If you use this docker image to run the functionality you will be able to reproduce our results, also taking care of operation system and hardware drivers. The docker image has got its own DOI. This is a good way to run experiments on a HPC machine.

Below we describe how you can setup and run the functionality in these two ways, but first we give you a general overview of the principles of training this RL-agent.

## Principles for training an RL-agent

We use the quality controlled implementation of RL-agents in `Stable Baselines 3` (implemented in Pytorch). In setting up the customized RL-environment we follow the
API from Open AI gym (more correct the package `gymnasium` which is a fork of gym made for maintenance) by inheriting for our custom environment for tunnelling from `gym.env`.

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
(which is TBM-strokes in this environment, eg. 1000 strokes of 1.8 meter). In each step a MLP-neural network is trained to match the best actions to the given state, ie. that maximizes the reward. The MLP's are used in different ways for the different agent
architectures: PPO, A2C, DDPG, TD3, SAC.
This training session is a classic NN-machine learning session looping over a number of
epochs (eg 10 epochs) in order to minimize the loss-function.

## Setup and run - the Docker way

Start the `Docker desktop` application.
If you haven't installed `Docker desktop` install it from here: https://www.docker.com/products/docker-desktop/

Start a terminal, ie. Powershell for Windows, Linux terminal etc.
cd to where you want to save experiment data, eg. `P:/2022/00/2022043/Calculations/exp_results`

Run:

```bash
docker run -it --rm --ipc=host --gpus all --name running_tbm_rl -v "$(PWD)":/exp_results tomfh/rock_classify:09.01-base bash
```

You will now be able to run 4 different scripts from the terminal.

- `python scripts/A_main_hydra.py`
- `python scripts/B_optimization_analyzer.py`
- `python scripts/C_training_path_analyzer.py`
- `python scripts/D_recommender.py`

For each of these scripts you can change configuration from the terminal setting using different flags. More info about this in a chapter below about the `Hydra configuration system`. If you start each run with the `--help` flag (eg. `python src/A_main_hydra.py --help`) you will see available config settings.

The experiment data produced will now be saved in the mounted experiment directory.

__A special note on running parameter optimization with Optuna.__ This normally involves running 100's of experiments. RL is computationally demanding, so this takes time. You can greatly speed up this process by spinning up several containers, for instance of different nodes and then run optimization. By binding the same experiment directory to the containers, all containers access the same optimization-study object.

If you access one node with several powerful CPU's you can spin up several containers on the same node using `docker compose'. By binding a shared volume, all containers can access the same optimization-study object.

### Singularity version

If you plan to run the RL-training on a HPC computer, the HPC machine will probably have Singularity (Apptainer) installed, not Docker. Here is how you can do the same in Singularity.

Login to the HPC machine, normally something like: `ssh <user-name>@<url-to-machine>` e.g.
`ssh tfh@odin.oslo.ngi.no`

Then make sure that you have mounted `P` into the HPC machine with: `ngi-mount P`.

In the example below we mount a directory for saving experiment data in a project directory and a config directory, to be able to change config outside the container. This is a sustainable setup that avoid large experiment data directories to be stored locally. **NOTE**: we mount the subdirectories in the mount directory into the target directory.

The `/projects/tbm-rl` could be any directory, but it is then also important to run the scripts from that directory, since that is the current working directory.

```bash
singularity shell --writable-tmpfs --nv --pwd / -B /home/tfh/NGI/P/2022/00/20220043/Calculations/exp_results:/home/tfh/projects/tbm-rl/ -B /home/tfh/NGI/P/2022/00/20220043/Calculations/config:/scripts/config docker://tomfh/tbm-rl:26.11-train

cd ~/projects/tbml-rl
python /scripts/A_main_hydra.py --help
# or directly to train a TD3 model on the best parameters
python /scripts/A_main_hydra.py agent=td3_best.yaml TRAIN.N_DUPLICATES=1
# or to load params from study object. nohup >>> experiment don't terminate when terminal session breaks
nohup python /scripts/A_main_hydra.py TRAIN.LOAD_PARAMS_FROM_STUDY=True EXP.STUDY=TD3_2022_09_27_study agent=td3_best.yaml
```

Then you run the scripts in the same way as described for Docker above.

**A NOTE on potential errors**.

- In running optuna studies placing the study.db object on a local NFS-system (like P:) you might encounter database locking problems using the default sqlite database engine. A countermeasure can be to start a mysql or postgres database for saving the study data.
- Mounting a local directory for saving experimentation data might give you speed issues or errors.

Both errors might be fixed by saving the study object locally on the HPC (e.g. mount experimentation directories from a project directory into the container). After every completed run a python function that copy the experimentation data to a local storage (in Singularity P: is accessible)


## Setup - the standard way

To clone the repository the code in the repository run:

```bash
git clone https://github.com/TunnRL/TunnRL_TBM_maintenance.git
```

### Dependencies

We have organized 2 ways of setting up the environment, downloading and installing all required pacakages, using the same package versions as have been used in development. In this way it is possible to repeat the experiments as close as possible.

1. The recommended way is to use the `poetry` system to set up the environment and install all dependencies. Poetry is stricter on depedencies than conda and define all depedencies in a human readable way through the categorized `pyproject.toml` file. The
`poetry.lock` defines exact version of all dependencies.

   Before you start, make sure you have installed `pyenv` to control your python version and `poetry` for environment and package handling. Install the python version defined in `.python-version` and continue. **NOTE**: the code in this branch is made to run in Linux, but will most likely also run in Windows. If you haven't got linux you can run linux from windows by activating Window Subsystem for Linux:
   https://learn.microsoft.com/en-us/windows/wsl/install


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

   Set up environment and install all depedencies (this command will install all packages defined in `poetry.lock`):

   ```bash
   poetry install
   ```

   Activate the environment with

   ```bash
   poetry shell
   ```

   Then you are ready to run your Python scripts in the exact same system setup as it
   has been developed!

1. Another way is to use `conda`.

   Create an environment called `rl_cutter` using `requirements.txt` with the help of `conda`. If you get pip errors, install pip libraries manually, e.g. `pip install pandas`

   ```bash
   conda env create --name rl_cutter --file requirements.txt
   ```

   Activate the new environment with:

   ```bash
   conda activate rl_cutter
   ```

## How to use the functionality - in general terms

1. **Optimization**. Choose an agent architecture (PPO, DDPG, TD3 etc.) and run an optimization process with Optuna to optimize hyperparameters to achieve the highest reward for that architecture. Run `A_main_hydra.py EXP.MODE=optimization`.
   - Optimization data is saved in the `optimization` directory and a subdirectory for each model run. Data is updated in this subdirectory for every chosen episode interval (eg. every 100 episode in a 10 000 episode study).
   - Each time one model-run is completed, common data-files for all experiments are saved into the `results` directory. Run `B_optimization_analyzer.py` to visualize this data.
2. **Training**. Train an agent for a number of episodes for a certain architecture and parameters given from an Optuna optimization for that architecture by running `A_main_hydra.py EXP.MODE=training`.
   - Metrics and trained models are saved into the `checkpoints` directory.
   - Visualize the training process with `C_training_path_analyzer.py`
3. **Execute**. To execute the actions for a trained agent.
   - To recommend the actions (cutter maintenance) for the next step (stroke) use the policy from a trained agent and run `D_recommender.py`.

All scripts B.., C.., D.. have flags to set. Run the scripts with the `--help` flag to see them.

## Hydra functionality

The hydra system logs and organize all config values, without touching the code itself. A hierarchical system of yaml files makes configuration easy. Config values can be altered in the yaml-files directly in the config-dir, or overridden in the terminal as described below. In this project, hydra functionality can be controlled in standard CLI in `A_main_hydra.py`.

Benefits of hydra logging

- Make experiments reproducible by saving exact config values for each experiment in an organized way.
- Change values in user friendly terminal setup (tab completion) without touching the code
- One organized place to change all config values, without the need to traverse the code
   for hard coded values. This makes code use easier for team
- Change a config value one place and not in a number of scripts (where you hopefully find all)
- Remove several lines of comments in the code.
- A hierachical configuration make it possible to easy change different parameters for different agents and reward-experiments

You can now run different reinforcement learning functions by defining different
inputs to `A_main_hydra.py` in the terminal. Eg.

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


Invoke tab completion with hydra for a certain script by running:

```bash
eval "$(python src/C_training_path_analyzer.py -sc install=bash)"
```

## MLflow

Mlflow is utilized to track all parameters and metrics upon completion of each experiment, thereby making it possible to easily compare results from different experiments.

All optimization and training runs are logged to mlflow.

Invoke the MLflow webinterface by (click the link that appears):

```bash
cd experiments
mlflow ui
```

## Tensorboard

Tensorboard is utilized to inspect the development of details in loss, learning_rate, episode reward, mean_reward etc. in each experiment. This is useful for an eventual stop of an experiment with unsatisfactory development.

```bash
# cd to directory above subdirectories with tensorboard files
tensorboard --logdir <directory with tensorboard event files>
# eg.
cd /mnt/P/2022/00/20220043/Calculations/
tensorboard --logdir SAC_2022_10_05_study
```


## Sqlite for Optuna optimization of parameters in parallell

To use hyperparameter functionality you need to have the database engine
`SQlite` installed. This is by default installed in Linux, but not in Windows.

Sqlite make it possible to have one common study-file for optimization that a number of terminal-sessions (utilizing all the cores on a computer) or computers can access at the same time. This makes it possible to run optimization of hyperparameters in parallell, greatly speeding up the process, which in reinforcement learning is computationally demanding.

Simply kick off a number of similar runs with the same study-name and all processes will update the same study-db. You can also continue updating the same study-db in a later optimization session.


## Pydantic for quality control of config parameters

Schemas set up in `XX_config_schemas.py` validate and parse config values from Hydra on runtime. Config values that violates the rules defined in the schemas will cause an error.