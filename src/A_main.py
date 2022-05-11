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
import numpy as np
import optuna
import pandas as pd
from tensorforce.agents import Agent
from tensorforce.environments import Environment

from XX_maintenance_lib import maintenance, CustomEnv, plotter


###############################################################################
# Constants and fixed variables

R = 3  # cutterhead radius [m]
D = 0.11  # cutter track spacing [m]
LIFE = 400000  # theoretical durability of one cutter [m]

STROKE_LENGTH = 1.8  # length of one stroke [m]
MAX_STROKES = 1000  # number of strokes per episode

EPISODES = 20_000  # episodes to train for
CHECKPOINT = 200  # checkpoints every X episodes
REWARD_MODE = 1  # one of three different reward modes needs to be chosen

t_I = 25  # time for inspection of cutterhead [min]
t_C_max = 75  # maximum time to change one cutter [min]
K = 1.25  # factor controlling change time of cutters

AGENT = 'ppo'  # name of the agent

OPTIMIZATION = False  # Flag that indicates whether or not to run an OPTUNA optimization or a "main run" with fixed hyperparameters
STUDY = '2022_04_25_study'  # name of the study if OPTIMIZATION = True


###############################################################################
# computed variables and instantiations

# Instantiate modules
m = maintenance()
pltr = plotter()

# total number of cutters
n_c_tot = int(round((R-D/2) / D, 0)) + 1
print(f'agent: {AGENT}; total number of cutters: {n_c_tot}\n')

cutter_positions = np.cumsum(np.full((n_c_tot), D)) - D/2

cutter_pathlenghts = cutter_positions * 2 * np.pi  # [m]

t_maint_max = m.maintenance_cost(t_I, t_C_max, n_c_tot, K)[0]

env = CustomEnv(n_c_tot, LIFE, t_I, t_C_max, K, t_maint_max, MAX_STROKES,
                STROKE_LENGTH, cutter_pathlenghts, REWARD_MODE)

environment = Environment.create(environment=env,
                                 max_episode_timesteps=MAX_STROKES)


###############################################################################
# main loop definition for the study


def objective(trial):
    '''objective function that runs the RL environment and agent'''
    print('\n')

    # initialize new series of episodes
    ep, train_start = 0, datetime.now().strftime("%Y%m%d-%H%M%S")
    # dataframe to collect episode recordings for later analyses
    df = pd.DataFrame({'episode': [], 'min_rewards': [], 'max_rewards': [],
                       'avg_rewards': [], 'min_brokens': [], 'avg_brokens': [],
                       'max_brokes': [], 'avg_changes_per_interv': []})

    # initialze an agent for a OPTUNA hyperparameter study
    if OPTIMIZATION is True:
        agent = Agent.create(agent='ppo',
                             actions=dict(type='int', shape=(n_c_tot,),
                                          num_values=2),
                             environment=environment,
                             batch_size=trial.suggest_int('batch_size', 5, 40, step=5),
                             learning_rate=trial.suggest_float('learning rate', low=1e-4, high=1e-1, log=True),
                             subsampling_fraction=trial.suggest_float('subsampl. fraction', low=0.1, high=1),
                             likelihood_ratio_clipping=trial.suggest_float('likelihood_ratio_clipping', low=0.1, high=1),
                             l2_regularization=trial.suggest_float('l2_regularization', low=0.0, high=.6),
                             entropy_regularization=trial.suggest_float('entropy_regularization', low=0.0, high=0.6),
                             discount=trial.suggest_float('discount', low=0.0, high=1),
                             variable_noise=trial.suggest_float('variable_noise', low=0.0, high=1))
    # or initialize an agent with fixed hyperparameter values based on a study
    else:
        agent = Agent.create(agent='ppo',
                             actions=dict(type='int', shape=(n_c_tot,),
                                          num_values=2),
                             environment=environment,
                             batch_size=10,
                             learning_rate=0.011,
                             subsampling_fraction=0.19,
                             likelihood_ratio_clipping=0.29,
                             l2_regularization=0.43,
                             entropy_regularization=0.37,
                             discount=0.18,
                             variable_noise=0.019)

    summed_actions = []  # list to collect number of actions per episode
    # main loop that trains the agent throughout multiple episodes
    for ep in range(ep, ep+EPISODES+1):
        state = environment.reset()  # reset new environment
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
            action = agent.act(states=state)
            # environment gives new state signal, terminal flag and reward
            state, terminal, reward = environment.execute(actions=action)
            # agent observes "response" of environment to its action
            agent.observe(terminal=terminal, reward=reward)
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
        # show intermediate results on checkpoints
        if ep % CHECKPOINT == 0 and ep != 0:
            # Report intermediate objective value.
            interm_rew = df['avg_rewards'].iloc[-CHECKPOINT:].mean()
            interm_broken = df['avg_brokens'].iloc[-CHECKPOINT:].mean()
            interm_changed = df['avg_changes_per_interv'].iloc[-CHECKPOINT:].mean()
            print(f'{ep}: rew {interm_rew} broken {interm_broken} changed {interm_changed}')

            if OPTIMIZATION is True:
                trial.report(interm_rew, ep)
            else:
                environment.save(AGENT, train_start, ep, states, actions,
                                 rewards, df, summed_actions, agent)
    # return final reward -> only necessary for optimization
    # final reward is computed as the average of the top 200 episodes of one
    # series of episodes
    final_reward = np.mean(np.sort(df['avg_rewards'].values)[-200:])
    return final_reward

###############################################################################
# finally run the loop either as study for hyperparameter optimization or for a
# main run

if OPTIMIZATION is True:  # study
    try:  # evtl. load already existing study if one exists
        study = joblib.load(fr'results\{STUDY}.pkl')
        print('prev. study loaded')
    except FileNotFoundError:  # or create a new study
        study = optuna.create_study(direction='maximize')
        print('new study created')
    # the OPTUNA study is then run in a loop so that intermediate results are
    # saved and can be checked every 2 trials
    for _ in range(200):  # save every second study
        study.optimize(objective, n_trials=2)
        joblib.dump(study, fr"results\{STUDY}.pkl")

        df = study.trials_dataframe()
        df.to_csv(fr'results\{STUDY}.csv')
    # print results of study
    trial = study.best_trial
    print('\nhighest reward: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
else:  # main run
    print('new main run created; no study results will be saved')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1)
