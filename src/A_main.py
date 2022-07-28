# -*- coding: utf-8 -*-
"""
Towards optimized TBM cutter changing policies with reinforcement learning
G.H. Erharter, T.F. Hansen
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Main code that either runs a hyperparameter optimization study with OPTUNA or a
"main run" of just one study with fixed hyperparameters

Created on Sat Oct 30 12:46:42 2021
code contributors: Georg H. Erharter, Tom F. Hansen
"""

import joblib
import numpy as np
import optuna
from pathlib import Path
from stable_baselines3 import PPO, DDPG, TD3
from stable_baselines3.common.env_checker import check_env
import warnings

from XX_maintenance_lib import CustomEnv, optimization

warnings.filterwarnings("ignore",
                        category=optuna.exceptions.ExperimentalWarning)

###############################################################################
# Constants and fixed variables

R = 3  # cutterhead radius [m]
D = 0.11  # cutter track spacing [m]
LIFE = 400000  # theoretical durability of one cutter [m]

STROKE_LENGTH = 1.8  # length of one stroke [m]
MAX_STROKES = 1000  # number of strokes per episode

EPISODES = 10_000  # max episodes to train for
# evaluations in optimization and checkpoints in training every X episodes
CHECKPOINT = 100

t_C_max = 75  # maximum time to change one cutter [min]

# MODE determines if either an optimization should run = "Optimization", or a
# new agent is trained with prev. optimized parameters = "Training", or an
# already trained agent is executed = "Execution"
MODE = 'Execution'  # 'Optimization', 'Training', 'Execution'
N_DEFAULT_TRIALS = 0  # n trials with default parameters to insert in study
# name of the study if MODE == 'Optimization' or 'Training'
# the Study name must start with the name of the agent that needs to be one of
# 'PPO', 'A2C', 'DDPG', 'SAC', 'TD3'
STUDY = 'DDPG_2022_07_27_study'  # DDPG_2022_07_27_study

###############################################################################
# computed variables and instantiations

# total number of cutters
n_c_tot = int(round((R-D/2) / D, 0)) + 1
print(f'total number of cutters: {n_c_tot}\n')

cutter_positions = np.cumsum(np.full((n_c_tot), D)) - D/2

cutter_pathlenghts = cutter_positions * 2 * np.pi  # [m]

env = CustomEnv(n_c_tot, LIFE, MAX_STROKES, STROKE_LENGTH, cutter_pathlenghts,
                R, t_C_max)
# check_env(env)  # check if env is a suitable gym environment

agent = STUDY.split('_')[0]

# Instantiate optimization function
optim = optimization(n_c_tot, env, EPISODES, CHECKPOINT, MODE, MAX_STROKES,
                     agent)

###############################################################################
# run one of the three modes: Optimization, Training, Execution

if MODE == 'Optimization':  # study
    try:  # evtl. load already existing study if one exists
        study = joblib.load(Path(f'results/{STUDY}.pkl'))
        print('prev. optimization study loaded')
    except FileNotFoundError:  # or create a new study
        study = optuna.create_study(direction='maximize', study_name=STUDY)
        print('new optimization study created')
    # the OPTUNA study is then run in a loop so that intermediate results are
    # saved and can be checked every trials
    study = optim.enqueue_defaults(study, agent, n_trials=N_DEFAULT_TRIALS)
    for _ in range(200):  # save every second study
        study.optimize(optim.objective, n_trials=1, catch=(ValueError,))
        joblib.dump(study, Path(f'results/{STUDY}.pkl'))
        df = study.trials_dataframe()
        df.to_csv(Path(f'results/{STUDY}.csv'))

elif MODE == 'Training':
    print('new main training run with optimized parameters started')
    study = joblib.load(Path(fr'results\{STUDY}.pkl'))

    # print results of study
    trial = study.best_trial
    print('\nhighest reward: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

    optim.objective(study.best_trial)

elif MODE == 'Execution':
    model = 'DDPG20220728-092315'  # name of the model to load
    tests = 3  # number of test episodes

    agent = DDPG.load(Path(fr'checkpoints\{model}\best_model'))

    # test agent throughout multiple episodes
    for test in range(tests):
        state = env.reset()  # reset new environment
        terminal = False  # reset terminal flag

        actions = []  # collect actions per episode
        states = [state]  # collect states per episode
        rewards = []  # collect rewards per episode
        broken_cutters = []  # collect number of broken cutters per stroke
        replaced_cutters = []  # collect n of replaced cutters per stroke
        moved_cutters = []  # collect n of moved_cutters cutters per stroke

        # one episode loop
        while not terminal:
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

        env.state_action_plot(states, actions, n_strokes=200,
                              savepath=fr'checkpoints\sample\{model}{test}_state_action.svg')
        env.environment_parameter_plot(fr'checkpoints\sample\{model}{test}_episode.svg', test)
        env.sample_ep_plot(states, actions, rewards, ep=test,
                           savepath=fr'checkpoints\sample\{model}{test}_sample.svg',
                           replaced_cutters=replaced_cutters,
                           moved_cutters=moved_cutters)

