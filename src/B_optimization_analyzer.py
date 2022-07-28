# -*- coding: utf-8 -*-
"""
Towards optimized TBM cutter changing policies with reinforcement learning
G.H. Erharter, T.F. Hansen
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Script that analyzes / visualizes the log of an OPTUNA study

Created on Thu Apr 14 13:28:07 2022
code contributors: Georg H. Erharter, Tom F. Hansen
"""

import joblib
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import pandas as pd
from pathlib import Path


###############################################################################
# Constants and fixed variables
name = 'PPO_2022_07_25_study'
agent = name.split('_')[0]

###############################################################################
# processing

# load data from completed OPTUNA study
df_study = pd.read_csv(Path(fr'results\{name}.csv'))
study = joblib.load(Path(fr"results\{name}.pkl"))

# print values of best trial in study
trial = study.best_trial
print('\nhighest reward: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

###############################################################################
# different visualizations of OPTUNA optimization

#####
# plot that shows the progress of the optimization over the individual trials
values_max = []
for i, value in enumerate(df_study['value']):
    if i == 0:
        values_max.append(value)
    else:
        if value > values_max[int(i-1)]:
            values_max.append(value)
        else:
            values_max.append(values_max[int(i-1)])

fig, ax = plt.subplots(figsize=(5, 5))
# only scatter complete studies -> in case there are pruned or incomplete ones
ax.scatter(df_study[df_study['state'] == 'COMPLETE']['number'],
           df_study[df_study['state'] == 'COMPLETE']['value'],
           s=30, alpha=0.7, color='grey', edgecolor='black', label='COMPLETE')
ax.scatter(df_study[df_study['state'] == 'FAIL']['number'],
           np.full(len(df_study[df_study['state'] == 'FAIL']),
                   df_study['value'].min()),
           s=30, alpha=0.7, color='red', edgecolor='black', label='FAIL')
ax.plot(df_study['number'], values_max, color='black')
ax.grid(alpha=0.5)
ax.set_xlabel('trial number')
ax.set_ylabel('reward')
ax.legend()
plt.tight_layout()
plt.savefig(Path(f'graphics/{name}_optimization_progress.svg'))
plt.close()

#####
# scatterplot of indivdual hyperparameters vs. reward
PARAMS = [p for p in df_study.columns if "params_" in p]

# replance NaN with "None"
if agent == 'SAC':
    df_study['params_SAC_action_noise'].fillna(value='None', inplace=True)

fig = plt.figure(figsize=(18, 9))

for i, param in enumerate(PARAMS):
    ax = fig.add_subplot(2, 7, i+1)
    ax.scatter(df_study[df_study['state'] == 'COMPLETE'][param],
               df_study[df_study['state'] == 'COMPLETE']['value'],
               s=20, color='grey', edgecolor='black', alpha=0.5,
               label='COMPLETE')
    ax.scatter(df_study[df_study['state'] == 'FAIL'][param],
               np.full(len(df_study[df_study['state'] == 'FAIL']),
                       df_study['value'].min()),
               s=20, color='red', edgecolor='black', alpha=0.5, label='FAIL')
    ax.grid(alpha=0.5)
    ax.set_xlabel(param.split('_')[-1])
    if i == 0:
        ax.set_ylabel('reward')

    if 'learning rate' in param:
        ax.set_xscale('log')
ax.legend()
fig.suptitle(name.split('_')[0])

plt.tight_layout()
plt.savefig(Path(f'graphics/{name}_optimization_scatter.svg'))
plt.close()

#####
# plot of the progress of individual runs
fig, ax = plt.subplots(figsize=(10, 8))

for trial in listdir('optimization'):
    if agent in trial:
        df_log = pd.read_csv(fr'optimization\{trial}\progress.csv')
        df_log['episodes'] = df_log[r'time/total_timesteps'] / df_log[r'rollout/ep_len_mean']
        df_log.dropna(axis=0, subset=[r'time/iterations'], inplace=True)
        if 'PPO' in trial:
            ax.plot(df_log['episodes'], df_log[r'rollout/ep_rew_mean'],
                    alpha=0.5, color='C0')
        elif 'A2C' in trial:
            ax.plot(df_log['episodes'], df_log[r'rollout/ep_rew_mean'],
                    alpha=0.5, color='C0')
        elif 'DDPG' in trial:
            ax.plot(df_log['episodes'], df_log[r'rollout/ep_rew_mean'],
                    alpha=0.5, color='C0')
        elif 'SAC' in trial:
            ax.plot(df_log['episodes'], df_log[r'rollout/ep_rew_mean'],
                    alpha=0.5, color='C0')
        elif 'TD3' in trial:
            ax.plot(df_log['episodes'], df_log[r'rollout/ep_rew_mean'],
                    alpha=0.5, color='C0')

custom_lines = [Line2D([0], [0], color='C0', lw=4),
                Line2D([0], [0], color='C1', lw=4),
                Line2D([0], [0], color='C2', lw=4),
                Line2D([0], [0], color='C3', lw=4),
                Line2D([0], [0], color='C4', lw=4)]

ax.legend(custom_lines, ['PPO', 'A2C', 'DDPG', 'SAC', 'TD3'])
# ax.set_ylim(top=1000, bottom=0)
ax.grid(alpha=0.5)
ax.set_xlabel('episodes')
ax.set_ylabel('reward')
plt.tight_layout()
plt.savefig(Path(f'graphics/{name}_optimization_intermediates.svg'))
plt.close()