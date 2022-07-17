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

from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
from pathlib import Path
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from XX_maintenance_lib import maintenance, CustomEnv, plotter, optimization

###############################################################################
# Constants and fixed variables

R = 3  # cutterhead radius [m]
D = 0.11  # cutter track spacing [m]
LIFE = 400000  # theoretical durability of one cutter [m]

STROKE_LENGTH = 1.8  # length of one stroke [m]
MAX_STROKES = 1000  # number of strokes per episode

EPISODES = 1000  # episodes to train for
CHECKPOINT = 100  # checkpoints every X episodes
REWARD_MODE = 1  # one of three different reward modes needs to be chosen

t_I = 25  # time for inspection of cutterhead [min]
t_C_max = 75  # maximum time to change one cutter [min]
K = 1.25  # factor controlling change time of cutters

# MODE determines if either an optimization should run = "Optimization", or a
# new agent is trained with prev. optimized parameters = "Training", or an
# already trained agent is executed = "Execution"
MODE = 'Optimization'  # 'Optimization', 'Training', 'Execution'
# name of the study if MODE == 'Optimization' or 'Training'
# the Study name must start with the name of the agent that needs to be one of
# 'PPO', 'A2C', 'DDPG', 'SAC', 'TD3'
STUDY = 'SAC_2022_07_17_study'

###############################################################################
# computed variables and instantiations

# Instantiate modules
m = maintenance()
pltr = plotter()

# total number of cutters
n_c_tot = int(round((R-D/2) / D, 0)) + 1
print(f'total number of cutters: {n_c_tot}\n')

cutter_positions = np.cumsum(np.full((n_c_tot), D)) - D/2

cutter_pathlenghts = cutter_positions * 2 * np.pi  # [m]

t_maint_max = m.maintenance_cost(t_I, t_C_max, n_c_tot, K)[0]

env = CustomEnv(n_c_tot, LIFE, t_I, t_C_max, K, t_maint_max, MAX_STROKES,
                STROKE_LENGTH, cutter_pathlenghts, REWARD_MODE, R)
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
    # saved and can be checked every 2 trials
    for _ in range(200):  # save every second study
        study.optimize(optim.objective, n_trials=2, catch=(ValueError,))
        joblib.dump(study, Path(f"results/{STUDY}.pkl"))

        df = study.trials_dataframe()
        df.to_csv(Path(f'results/{STUDY}.csv'))

elif MODE == 'Training':
    print('new main training run with optimized parameters started')
    study = joblib.load(fr'results\{STUDY}.pkl')

    # print results of study
    trial = study.best_trial
    print('\nhighest reward: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

    optim.objective(study.best_trial)

    df_log = pd.read_csv(r'checkpoints\progress.csv')
    fig, ax = plt.subplots()
    ax.plot(df_log[r'time/time_elapsed'], df_log[r'rollout/ep_rew_mean'])
    plt.savefig(r'checkpoints\progress.jpg')

elif MODE == 'Execution':
    # initialize test episodes
    ep, train_start = 0, datetime.now().strftime("%Y%m%d-%H%M%S")
    # dataframe to collect episode recordings for later analyses
    df = pd.DataFrame({'episode': [], 'min_rewards': [], 'max_rewards': [],
                       'avg_rewards': [], 'min_brokens': [],
                       'avg_brokens': [], 'max_brokes': [],
                       'avg_changes_per_interv': []})

    agent = PPO.load(r'checkpoints\PPO__200000_steps')

    summed_actions = []  # list to collect number of actions per episode

    # main loop that trains the agent throughout multiple episodes
    for ep in range(3):
        state = env.reset()  # reset new environment
        terminal = False  # reset terminal flag

        actions = []  # collect actions per episode
        states = [state]  # collect states per episode
        rewards = []  # collect rewards per episode
        broken_cutters = []  # collect number of broken cutters per episode
        # loop that trains agent in "act-observe" style on one episode
        while not terminal:
            # collect number of broken cutters in curr. state
            broken_cutters.append(len(np.where(state == 0)[0]))
            # agent takes an action -> tells which cutters to replace
            action, _states = agent.predict(state, deterministic=True)
            # environment gives new state signal, terminal flag and reward
            state, reward, terminal, info = env.step(action)
            # collect actions, states and rewards for later analyses
            actions.append(action)
            states.append(state)
            rewards.append(reward)
        # compute and collect average statistics per episode for later analyses
        avg_reward = np.mean(rewards)
        avg_broken = np.mean(broken_cutters)
        avg_changed = np.mean(np.sum(np.where(np.vstack(actions) > .5, 1, 0), 1))
        df_temp = pd.DataFrame({'episode': [ep], 'min_rewards': [min(rewards)],
                                'max_rewards': [max(rewards)],
                                'avg_rewards': [avg_reward],
                                'min_brokens': [min(broken_cutters)],
                                'avg_brokens': [avg_broken],
                                'max_brokes': [max(broken_cutters)],
                                'avg_changes_per_interv': [avg_changed]})
        df = pd.concat([df, df_temp])
        summed_actions.append(np.sum(actions, axis=0))

        env.save('PPO', train_start, ep, states, actions, rewards, df,
                 summed_actions, agent)
