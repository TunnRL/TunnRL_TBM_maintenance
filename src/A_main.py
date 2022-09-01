"""
Towards optimized TBM cutter changing policies with reinforcement learning
G.H. Erharter, T.F. Hansen
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Main code that either runs a hyperparameter optimization study with OPTUNA or a
"main run" of just one study with fixed hyperparameters

Created on Sat Oct 30 12:46:42 2021
code contributors: Georg H. Erharter, Tom F. Hansen
"""

import warnings
from pathlib import Path

import hydra
import numpy as np
import optuna
import torch.nn as nn  # used in evaluation of yaml file
import yaml
from joblib import Parallel, delayed
from numpy.typing import NDArray
from omegaconf import DictConfig
from rich.console import Console
from rich.traceback import install
from stable_baselines3.common.env_checker import check_env

from XX_experiment_factory import Optimization, load_best_model
from XX_hyperparams import Hyperparameters
from XX_plotting import Plotter
from XX_TBM_environment import CustomEnv


install()


###############################################################################
# CONSTANTS AND FIXED VARIABLES
###############################################################################

# TBM EXCAVATION PARAMETERS
######################
CUTTERHEAD_RADIUS = 3  # cutterhead radius [m]
TRACK_SPACING = 0.11  # cutter track spacing [m]
LIFE = 400000  # theoretical durability of one cutter [m]
STROKE_LENGTH = 1.8  # length of one stroke [m]
MAX_STROKES = 1000  # number of strokes per episode
BROKEN_CUTTERS_THRESH = 0.5  # minimum required % of functional cutters

# MAIN EXPERIMENT INFO
######################
# MODE determines if either an optimization should run = "Optimization", or a
# new agent is trained with prev. optimized parameters = "Training", or an
# already trained agent is executed = "Execution"
MODE = "optimization"  # 'optimization', 'training', 'execution'
# set to run SB3 environment check function
# Checks if env is a suitable gym environment
CHECK_ENV = False
DEBUG = True
# name of the study if MODE == 'Optimization' or 'Training'
# the Study name must start with the name of the agent that needs to be one of
# 'PPO', 'A2C', 'DDPG', 'SAC', 'TD3'
STUDY = "PPO_2022_08_27_study"  # DDPG_2022_07_27_study 'PPO_2022_08_03_study'
# evaluations in optimization and checkpoints in training every X episodes
CHECKPOINT_INTERVAL = 100
EPISODES = 12_000  # max episodes to train for

# SETUP FOR OPTIMIZATION AND TRAINING
######################
PLOT_PROGRESS = True  # wether to make progress plots during training or not. Plotting takes time
DETERMINISTIC = False  # fixed environment used in evaluation or not

# OPTIMIZATION SPECIAL SETUP
######################
DEFAULT_TRIAL = False  # first run a trial with default parameters.
MAX_NO_IMPROVEMENT_EVALS = (
    2  # maximum number of evaluations without improvement
)
# n optuna trials to run in total (including eventual default trial)
N_SINGLE_RUN_OPTUNA_TRIALS = 2
# NOTE: memory can be an issue for many parallell processes. Size of neural network and
# available memory will be limiting factors
N_CORES_PARALLELL = -1
N_PARALLELL_PROCESSES = 5
N_EVAL_EPISODES_OPTIMIZATION = 3  # n eval episodes in stop training cb
N_EVAL_EPISODES_REWARD = (
    10  # n eval episodes for computing mean reward in end of training
)

# TRAINING SPECIAL SETUP
######################
# load best parameters from study object in training. Alternative: load from yaml
LOAD_PARAMS_FROM_STUDY = False
N_EVAL_EPISODES_TRAINING = 10

# EXECUTION SPECIAL SETUP
######################
EXECUTION_MODEL = "DDPG_a3536e69-a501-421d-b6d6-51f152554660"
NUM_TEST_EPISODES = 3

###############################################################################
# WARNINGS AND ERROR CHECKING INPUT VARIABLES
###############################################################################
if DEBUG:
    EPISODES = 10
    CHECKPOINT_INTERVAL = 5
    N_PARALLELL_PROCESSES = 2
    N_CORES_PARALLELL = 2

warnings.filterwarnings(
    "ignore", category=optuna.exceptions.ExperimentalWarning
)

if DEFAULT_TRIAL is True:
    warnings.warn(
        "Optimization runs are started with default parameter values"
    )

assert (
    N_CORES_PARALLELL >= N_PARALLELL_PROCESSES or N_CORES_PARALLELL == -1
), "Num cores must be >= num parallell processes."
if N_PARALLELL_PROCESSES > 1 and (MODE == "training" or MODE == "execution"):
    warnings.warn("No parallellization in training and execution mode")

if LOAD_PARAMS_FROM_STUDY is True and MODE == "training":
    assert Path(
        f"./results/{STUDY}.db"
    ).exists(), "The study object does not exist"

if LOAD_PARAMS_FROM_STUDY is False and MODE == "training":
    assert Path(
        f'results/algorithm_parameters/{STUDY.split("_")[0]}.yaml'
    ).exists(), "a yaml file with pararameter does not exist."

###############################################################################
# COMPUTED/DERIVED VARIABLES AND INSTANTIATIONS
###############################################################################
rich_console = Console()

n_c_tot = (
    int(round((CUTTERHEAD_RADIUS - TRACK_SPACING / 2) / TRACK_SPACING, 0)) + 1
)

cutter_positions = (
    np.cumsum(np.full((n_c_tot), TRACK_SPACING)) - TRACK_SPACING / 2
)

cutter_pathlenghts = cutter_positions * 2 * np.pi  # [m]

env = CustomEnv(
    n_c_tot,
    LIFE,
    MAX_STROKES,
    STROKE_LENGTH,
    cutter_pathlenghts,
    CUTTERHEAD_RADIUS,
    BROKEN_CUTTERS_THRESH,
)
if CHECK_ENV:
    check_env(env)

agent_name = STUDY.split("_")[0]
assert agent_name in [
    "PPO",
    "A2C",
    "DDPG",
    "SAC",
    "TD3",
], f"{agent_name} is not a valid agent."


hparams = Hyperparameters()
optim = Optimization(
    n_c_tot,
    env,
    STUDY,
    EPISODES,
    CHECKPOINT_INTERVAL,
    MODE,
    MAX_STROKES,
    agent_name,
    DEFAULT_TRIAL,
    DEBUG,
    PLOT_PROGRESS,
    DETERMINISTIC,
    MAX_NO_IMPROVEMENT_EVALS,
    N_EVAL_EPISODES_OPTIMIZATION,
    N_EVAL_EPISODES_REWARD,
    N_EVAL_EPISODES_TRAINING,
    rich_console,
)

###############################################################################
# run one of the three modes: optimization, training, execution

if MODE == "optimization":  # study
    # print(
    #     f"{N_SINGLE_RUN_OPTUNA_TRIALS * N_PARALLELL_PROCESSES} optuna trials are processed in {N_PARALLELL_PROCESSES} processes.\n"
    # )

    db_path = f"results/{STUDY}.db"
    db_file = f"sqlite:///{db_path}"
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY,
        storage=db_file,
        load_if_exists=True,
        sampler=sampler,
    )

    optim.optimize(study, N_SINGLE_RUN_OPTUNA_TRIALS)

    # Parallel(n_jobs=N_CORES_PARALLELL, verbose=10, backend="loky")(
    #     delayed(optim.optimize)(N_SINGLE_RUN_OPTUNA_TRIALS) for _ in range(N_PARALLELL_PROCESSES))

elif MODE == "training":
    print(f"New {agent_name} training run with optimized parameters started.")
    print(f" - total number of cutters: {n_c_tot}\n")

    if LOAD_PARAMS_FROM_STUDY is True:
        print(f"loading parameters from the study object: {STUDY}")
        db_path = f"results/{STUDY}.db"
        db_file = f"sqlite:///{db_path}"
        study = optuna.load_study(study_name=STUDY, storage=db_file)
        best_trial = study.best_trial
        print(f"Highest reward from best trial: {best_trial.value}")

        best_params_dict = hparams.remake_params_dict(
            algorithm=agent_name,
            raw_params_dict=best_trial.params,
            env=env,
            n_actions=n_c_tot * n_c_tot,
        )
    else:
        print("loading parameters from yaml file...")
        with open(f"results/algorithm_parameters/{agent_name}.yaml") as file:
            best_params_dict: dict = yaml.safe_load(file)
        with open(
            f"results/algorithm_parameters/{agent_name}_sub.yaml"
        ) as file:
            sub_params: dict = yaml.safe_load(file)

        best_params_dict["learning_rate"] = hparams.parse_learning_rate(
            best_params_dict["learning_rate"]
        )
        best_params_dict["policy_kwargs"] = eval(
            best_params_dict["policy_kwargs"]
        )
        best_params_dict.update(dict(env=env, n_steps=MAX_STROKES))

    optim.train_agent(best_params_dict, sub_params)

elif MODE == "execution":
    agent_name = EXECUTION_MODEL.split("_")[0]
    agent = load_best_model(
        agent_name, main_dir="checkpoints", agent_dir=EXECUTION_MODEL
    )

    # test agent throughout multiple episodes
    for test_ep_num in range(NUM_TEST_EPISODES):
        print(f"Episode num: {test_ep_num}")
        state = env.reset()  # reset new environment
        terminal = False  # reset terminal flag

        actions: list[NDArray] = []  # collect actions per episode
        states: list[NDArray] = [state]  # collect states per episode
        rewards = []  # collect rewards per episode
        broken_cutters = []  # collect number of broken cutters per stroke
        replaced_cutters = []  # collect n of replaced cutters per stroke
        moved_cutters = []  # collect n of moved_cutters cutters per stroke

        # one episode loop
        i = 0
        while not terminal:
            print(f"Stroke (step) num: {i}")
            # collect number of broken cutters in curr. state
            broken_cutters.append(len(np.where(state == 0)[0]))
            # agent takes an action -> tells which cutters to replace
            action = agent.predict(state, deterministic=DETERMINISTIC)[0]
            # environment gives new state signal, terminal flag and reward
            state, reward, terminal, info = env.step(action)
            # collect actions, states and rewards for later analyses
            actions.append(action)
            states.append(state)
            rewards.append(reward)
            replaced_cutters.append(env.replaced_cutters)
            moved_cutters.append(env.moved_cutters)
            i += 1

        Plotter.state_action_plot(
            states,
            actions,
            n_strokes=300,
            n_c_tot=n_c_tot,
            show=False,
            savepath=f"checkpoints/_sample/{EXECUTION_MODEL}{test_ep_num}_state_action.svg",
        )
        Plotter.environment_parameter_plot(
            test_ep_num,
            env,
            show=False,
            savepath=f"checkpoints/_sample/{EXECUTION_MODEL}{test_ep_num}_episode.svg",
        )
        Plotter.sample_ep_plot(
            states,
            actions,
            rewards,
            ep=test_ep_num,
            savepath=f"checkpoints/_sample/{EXECUTION_MODEL}{test_ep_num}_sample.svg",
            replaced_cutters=replaced_cutters,
            moved_cutters=moved_cutters,
        )

else:
    raise ValueError(f"{MODE} is not a valid mode")