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

import numpy as np
import optuna
from stable_baselines3.common.env_checker import check_env
from joblib import Parallel, delayed

from XX_maintenance_lib import CustomEnv, Optimization
from XX_plotting import Plotter
from XX_utility import load_best_model


warnings.filterwarnings("ignore",
                        category=optuna.exceptions.ExperimentalWarning)


###############################################################################
# Constants and fixed variables

CUTTERHEAD_RADIUS = 3  # cutterhead radius [m]
TRACK_SPACING = 0.11  # cutter track spacing [m]
LIFE = 400000  # theoretical durability of one cutter [m]

STROKE_LENGTH = 1.8  # length of one stroke [m]
MAX_STROKES = 1000  # number of strokes per episode
BROKEN_CUTTERS_THRESH = 0.5  # minimum required percentage of functional cutters

EPISODES = 10_000  # max episodes to train for 10_000
# evaluations in optimization and checkpoints in training every X episodes

CHECKPOINT_INTERVAL = 100

# MODE determines if either an optimization should run = "Optimization", or a
# new agent is trained with prev. optimized parameters = "Training", or an
# already trained agent is executed = "Execution"

MODE = 'Execution'  # 'Optimization', 'Training', 'Execution'
DEFAULT_TRIAL = False  # first run a trial with default parameters.
N_OPTUNA_TRIALS = 2  # n optuna trials to run in total (including eventual default trial)
# NOTE: memory can be an issue for many parallell processes. Size of neural network and 
# available memory will be limiting factors
N_CORES_PARALLELL = -1
N_PARALLELL_PROCESSES = 5
# assert N_PARALLELL_PROCESSES <= N_OPTUNA_TRIALS, "Num. parallell processes cannot be higher than number of trials"
# name of the study if MODE == 'Optimization' or 'Training'
# the Study name must start with the name of the agent that needs to be one of
# 'PPO', 'A2C', 'DDPG', 'SAC', 'TD3'
STUDY = 'PPO_2022_08_17_study'  # DDPG_2022_07_27_study 'PPO_2022_08_03_study'

#load best parameters from study object in training. Alt: load from yaml
#TODO: do an automatic save of parameters upon completing an optuna-experiment session
LOAD_PARAMS_FROM_STUDY = True

EXECUTION_MODEL = "PPO20220817-115856"
NUM_TEST_EPISODES = 3

###############################################################################
# computed variables and instantiations

n_c_tot = int(round((CUTTERHEAD_RADIUS - TRACK_SPACING / 2) / TRACK_SPACING, 0)) + 1
print(f'total number of cutters: {n_c_tot}\n')

cutter_positions = np.cumsum(np.full((n_c_tot), TRACK_SPACING)) - TRACK_SPACING / 2

cutter_pathlenghts = cutter_positions * 2 * np.pi  # [m]

env = CustomEnv(n_c_tot, LIFE, MAX_STROKES, STROKE_LENGTH, cutter_pathlenghts,
                CUTTERHEAD_RADIUS, BROKEN_CUTTERS_THRESH)
# check_env(env)  # check if env is a suitable gym environment

agent = STUDY.split('_')[0]
assert agent in ["PPO", "A2C", "DDPG", "SAC", "TD3"], f"{agent} is not a valid agent."

optim = Optimization(n_c_tot, env, EPISODES, CHECKPOINT_INTERVAL, MODE,
                     MAX_STROKES, agent, DEFAULT_TRIAL)

plotter = Plotter()

###############################################################################
# run one of the three modes: Optimization, Training, Execution


def optimize(n_trials: int):
    """Optimize-function to be called in parallell process.

    Args:
        n_trials (int): Number of trials in each parallell process.
    
    TODO: move this to Optimization class
    """
    db_path = f"results/{STUDY}.db"
    db_file = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=STUDY, storage=db_file)
    study.optimize(optim.objective, n_trials=n_trials,
                   catch=(ValueError,))

if MODE == 'Optimization':  # study
    db_path = f"results/{STUDY}.db"
    db_file = f"sqlite:///{db_path}"
    sampler = optuna.samplers.TPESampler() #TODO: play around with sampling configs
    study = optuna.create_study(
        direction='maximize', study_name=STUDY, storage=db_file,
        load_if_exists=True, sampler=sampler)
    
    Parallel(n_jobs=N_CORES_PARALLELL, verbose=10, backend="loky")(
        delayed(optimize)(N_OPTUNA_TRIALS) for _ in range(N_PARALLELL_PROCESSES))
    
    study = optuna.load_study(study_name=STUDY, storage=db_file)
    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

elif MODE == 'Training':
    print('new main training run with optimized parameters started')
    db_path = f"results/{STUDY}.db"
    db_file = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=STUDY, storage=db_file)

    best_trial = study.best_trial
    print(f"Highest reward: {best_trial.value}")
    if LOAD_PARAMS_FROM_STUDY: 
        print(f"loading parameters from the study object: {STUDY}")
        best_params_dict = optim.remake_params_dict(
            algorithm=agent, raw_params_dict=best_trial.params)
    else:
        print("loading parameters from yaml file")
        
    print(f"Best hyperparameters: {best_params_dict}")
    optim.train_agent(agent_name=agent, best_parameters=best_params_dict)

elif MODE == 'Execution':
    agent_name = EXECUTION_MODEL[0:3]
    agent = load_best_model(agent_name, main_dir="checkpoints",
                            agent_dir=EXECUTION_MODEL)

    # test agent throughout multiple episodes
    for test_ep_num in range(NUM_TEST_EPISODES):
        print(f"Episode num: {test_ep_num}")
        state = env.reset()  # reset new environment
        terminal = False  # reset terminal flag

        actions = []  # collect actions per episode
        states = [state]  # collect states per episode
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

        plotter.state_action_plot(states, actions, n_strokes=200, n_c_tot=n_c_tot,
                                  savepath=f'checkpoints/sample/{EXECUTION_MODEL}{test_ep_num}_state_action.svg')
        plotter.environment_parameter_plot(test_ep_num, env, savepath=f'checkpoints/sample/{EXECUTION_MODEL}{test}_episode.svg')
        plotter.sample_ep_plot(states, actions, rewards, ep=test_ep_num,
                               savepath=f'checkpoints/sample/{EXECUTION_MODEL}{test_ep_num}_sample.svg',
                               replaced_cutters=replaced_cutters,
                               moved_cutters=moved_cutters)
