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

from joblib import Parallel, delayed
import numpy as np
from numpy.typing import NDArray
import optuna
from stable_baselines3.common.env_checker import check_env
import torch.nn as nn  # used in evaluation of yaml file
import yaml

from XX_experiment_factory import Optimization, load_best_model, ExperimentAnalysis
from XX_hyperparams import Hyperparameters
from XX_plotting import Plotter
from XX_TBM_environment import CustomEnv

###############################################################################
# CONSTANTS AND FIXED VARIABLES
###############################################################################

# TBM EXCAVATION PARAMETERS
######################
CUTTERHEAD_RADIUS = 4  # cutterhead radius [m]
TRACK_SPACING = 0.1  # cutter track spacing [m]
LIFE = 400000  # theoretical durability of one cutter [m]
STROKE_LENGTH = 1.8  # length of one stroke [m]
MAX_STROKES = 1000  # number of strokes per episode

# REWARD FUNCTION PARAMETERS
######################
BROKEN_CUTTERS_THRESH = 0.85  # minimum required % of functional cutters
ALPHA = 0.1  # weighting factor for replacing cutters
BETA = 0.65  # weighting factor for moving cutters
GAMMA = 0.1  # weighting factor for cutter distance
DELTA = 0.15  # weighting factor for entering cutterhead
CHECK_BEARING_FAILURE = True  # if True should check cutter bearing failures
BEARING_FAILURE_PENALTY = 0

# MAIN EXPERIMENT INFO
######################
# MODE determines if either an optimization should run = "optimization", or a
# new agent is trained with prev. optimized parameters = "training", or an
# already trained agent is executed = "execution"
MODE = "optimization"  # 'optimization', 'training', 'execution'
# set to run SB3 environment check function
# Checks if env is a suitable gym environment
CHECK_ENV = False
DEBUG = False  # sets test values for quicker response

# PARAMETERS FOR MODES OPTIMIZATION AND TRAINING
#####################
# name of the study if MODE == 'Optimization' or 'Training'
# the Study name must start with the name of the agent that needs to be one of
# 'PPO', 'A2C', 'DDPG', 'SAC', 'TD3'
STUDY = "PPO_2022_09_27_study"
# evaluations in optimization and checkpoints in training every X episodes
CHECKPOINT_INTERVAL = 100
EPISODES = 12_000  # max episodes to train for

# OPTIMIZATION SPECIAL SETUP
######################
DEFAULT_TRIAL = True  # first run a trial with default parameters.
MAX_NO_IMPROVEMENT = 1  # maximum number of evaluations without improvement
# n optuna trials to run in total (including eventual default trial)
N_SINGLE_RUN_OPTUNA_TRIALS = 250
# NOTE: memory can be an issue for many parallell processes. Size of neural
# network and available memory will be limiting factors
N_CORES_PARALLELL = 4
N_PARALLELL_PROCESSES = 4

# TRAINING SPECIAL SETUP
######################
# load best parameters from study object in training. Alternative: load from yaml
LOAD_PARAMS_FROM_STUDY = False

# EXECUTION SPECIAL SETUP
######################
EXECUTION_MODEL = "TD3_58c8ef65-de89-4b64-ab26-c5ddae4f0d06"
NUM_TEST_EPISODES = 10
VISUALIZE_EPISODES = False  # if the episodes should be visualized or not

###############################################################################
# WARNINGS AND ERROR CHECKING INPUT VARIABLES
###############################################################################

if DEBUG is True:
    EPISODES = 20
    CHECKPOINT_INTERVAL = 10
    N_PARALLELL_PROCESSES = 1
    N_CORES_PARALLELL = 1

warnings.filterwarnings("ignore",
                        category=optuna.exceptions.ExperimentalWarning)

if DEFAULT_TRIAL is True:
    warnings.warn(
        "Optimization runs are started with default parameter values"
    )

assert N_CORES_PARALLELL >= N_PARALLELL_PROCESSES or N_CORES_PARALLELL == -1, "Num cores must be >= num parallell processes."
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

n_c_tot = (
    int(round((CUTTERHEAD_RADIUS - TRACK_SPACING / 2) / TRACK_SPACING, 0)) + 1
)
print(f"\ntotal number of cutters: {n_c_tot}\n")

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
    ALPHA, BETA, GAMMA, DELTA,
    CHECK_BEARING_FAILURE,
    BEARING_FAILURE_PENALTY
)
if CHECK_ENV:
    check_env(env)

agent_name = STUDY.split('_')[0]
assert agent_name in ["PPO", "A2C", "DDPG", "SAC", "TD3"], f"{agent_name} is not a valid agent."

optim = Optimization(n_c_tot, env, STUDY, EPISODES, CHECKPOINT_INTERVAL, MODE,
                     MAX_STROKES, agent_name, DEFAULT_TRIAL,
                     MAX_NO_IMPROVEMENT)

ea = ExperimentAnalysis()
hparams = Hyperparameters()
plotter = Plotter()

###############################################################################
# run one of the three modes: optimization, training, execution

if MODE == "optimization":  # study
    print(
        f"{N_SINGLE_RUN_OPTUNA_TRIALS * N_PARALLELL_PROCESSES} optuna trials are processed in {N_PARALLELL_PROCESSES} processes.\n"
    )

    db_path = f"results/{STUDY}.db"
    db_file = f"sqlite:///{db_path}"
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        direction='maximize', study_name=STUDY, storage=db_file,
        load_if_exists=True, sampler=sampler)

    Parallel(n_jobs=N_CORES_PARALLELL, verbose=10, backend="loky")(
        delayed(optim.optimize)(N_SINGLE_RUN_OPTUNA_TRIALS) for _ in range(N_PARALLELL_PROCESSES))

elif MODE == 'training':
    print(f'New {agent_name} training run with optimized parameters started.')

    if LOAD_PARAMS_FROM_STUDY is True:
        print(f"loading parameters from the study object: {STUDY}")
        db_path = f"results/{STUDY}.db"
        db_file = f"sqlite:///{db_path}"
        study = optuna.load_study(study_name=STUDY, storage=db_file)
        best_trial = study.best_trial
        print(f"Highest reward from best trial: {best_trial.value}")

        best_params_dict = hparams.remake_params_dict(
            algorithm=agent_name, raw_params_dict=best_trial.params, env=env,
            n_actions=n_c_tot * n_c_tot)
    else:
        print("loading parameters from yaml file...")
        with open(f"results/algorithm_parameters/{agent_name}.yaml") as file:
            best_params_dict: dict = yaml.safe_load(file)

        best_params_dict["learning_rate"] = hparams.parse_learning_rate(best_params_dict["learning_rate"])
        best_params_dict["policy_kwargs"] = eval(best_params_dict["policy_kwargs"])
        best_params_dict.update(dict(env=env, n_steps=MAX_STROKES))

    optim.train_agent(best_parameters=best_params_dict)

elif MODE == 'execution':
    agent_name = EXECUTION_MODEL.split('_')[0]
    agent = load_best_model(agent_name, main_dir="optimization",
                            agent_dir=EXECUTION_MODEL)

    all_actions = []
    all_states = []
    all_rewards = []
    all_broken_cutters = []
    all_replaced_cutters = []
    all_moved_cutters = []

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
            # print(f"Stroke (step) num: {i}")
            # collect number of broken cutters in curr. state
            broken_cutters.append(len(np.where(state == 0)[0]))
            # agent takes an action -> tells which cutters to replace
            action = agent.predict(state, deterministic=False)[0]
            # environment gives new state signal, terminal flag and reward
            state, reward, terminal, info = env.step(action)
            # collect actions, states and rewards for later analyses
            actions.append(action)
            states.append(state)
            rewards.append(reward)
            replaced_cutters.append(env.replaced_cutters)
            moved_cutters.append(env.moved_cutters)
            i += 1

        all_actions.append(actions)
        all_states.append(states[:-1])
        all_rewards.append(rewards)
        all_broken_cutters.append(broken_cutters)
        all_replaced_cutters.append([len(c) for c in replaced_cutters])
        all_moved_cutters.append([len(c) for c in moved_cutters])

        if VISUALIZE_EPISODES is True:
            plotter.state_action_plot(states, actions, n_strokes=300,
                                      rewards=rewards, n_c_tot=n_c_tot,
                                      show=False,
                                      savepath=f'checkpoints/_sample/{EXECUTION_MODEL}{test_ep_num}_state_action.svg')
            plotter.environment_parameter_plot(test_ep_num, env, show=False,
                                               savepath=f'checkpoints/_sample/{EXECUTION_MODEL}{test_ep_num}_episode.svg')
            plotter.sample_ep_plot(states, actions, rewards,
                                   replaced_cutters=replaced_cutters,
                                   moved_cutters=moved_cutters,
                                   n_cutters=n_c_tot, show=False,
                                   savepath=f'checkpoints/_sample/{EXECUTION_MODEL}{test_ep_num}_sample.svg')

    df_reduced = ea.dimensionality_reduction(all_actions,
                                             all_states,
                                             all_rewards,
                                             all_broken_cutters,
                                             all_replaced_cutters,
                                             all_moved_cutters,
                                             perplexity=200)

    plotter.action_analysis_scatter_plotly(df_reduced,
                                           savepath=f"checkpoints/_sample/{EXECUTION_MODEL}_TSNE_scatter_plotly.html")

    plotter.action_analysis_scatter(df_reduced,
                                    savepath=f'checkpoints/_sample/{EXECUTION_MODEL}_TSNE_scatter.svg')

else:
    raise ValueError(f"{MODE} is not a valid mode")
