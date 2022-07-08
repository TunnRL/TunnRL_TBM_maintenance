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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from os import listdir
import pandas as pd
from pathlib import Path


###############################################################################
# Constants and fixed variables

name = '2022_07_08_study'

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

fig, ax = plt.subplots(figsize=(3.465, 3.465))
# only scatter complete studies -> in case there are pruned or incomplete ones
ax.scatter(df_study[df_study['state'] == 'COMPLETE']['number'],
            df_study[df_study['state'] == 'COMPLETE']['value'],
            s=30, alpha=0.5, color='grey', edgecolor='black')
ax.plot(df_study['number'], values_max, color='black')
ax.grid(alpha=0.5)
ax.set_xlabel('trial number')
ax.set_ylabel('reward')
plt.tight_layout()
plt.savefig(Path(f'graphics/{name}_optimization_progress.svg'))
plt.close()

#####
# scatterplot of indivdual hyperparameters vs. reward
AGENTS = ['PPO', 'A2C', 'DDPG', 'SAC', 'TD3']
PARAMS = [p for p in df_study.columns if "params_" in p]

agent_params = []
for a in AGENTS:
    agent_params.append([p_a for p_a in PARAMS if a in p_a])
lenghts = [len(l) for l in agent_params]

from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(max(lenghts)*3, len(AGENTS)*3))
gs = GridSpec(len(AGENTS), max(lenghts), figure=fig)

for i, a in enumerate(AGENTS):
    for j, param in enumerate(agent_params[i]):
        ax = fig.add_subplot(gs[i, j])
        ax.scatter(df_study[df_study['state'] == 'COMPLETE'][param],
                   df_study[df_study['state'] == 'COMPLETE']['value'],
                   s=20, color='grey', edgecolor='black', alpha=0.5)
        ax.grid(alpha=0.5)
        ax.set_xlabel(param.split('_')[-1])
        if j == 0:
            ax.set_ylabel(f'{a}\nreward')
        if 'learning rate' in param:
            ax.set_xscale('log')

plt.tight_layout()
plt.savefig(Path(f'graphics/{name}_optimization_scatter.svg'))
plt.close()

#####
# plot of the progress of individual runs
fig, ax = plt.subplots(figsize=(10, 8))

for trial in listdir('optimization'):
    try:
        records = np.load(fr'optimization\{trial}\evaluations.npz')
        if 'PPO' in trial:
            ax.plot(np.arange(len(records['timesteps'])),
                    np.mean(records['results'], axis=1), alpha=0.5, color='C0')
        elif 'A2C' in trial:
            ax.plot(np.arange(len(records['timesteps'])),
                    np.mean(records['results'], axis=1), alpha=0.5, color='C1')
        elif 'DDPG' in trial:
            ax.plot(np.arange(len(records['timesteps'])),
                    np.mean(records['results'], axis=1), alpha=0.5, color='C2')
        elif 'SAC' in trial:
            ax.plot(np.arange(len(records['timesteps'])),
                    np.mean(records['results'], axis=1), alpha=0.5, color='C3')
        elif 'TD3' in trial:
            ax.plot(np.arange(len(records['timesteps'])),
                    np.mean(records['results'], axis=1), alpha=0.5, color='C4')
    except FileNotFoundError:
        pass

custom_lines = [Line2D([0], [0], color='C0', lw=4),
                Line2D([0], [0], color='C1', lw=4),
                Line2D([0], [0], color='C2', lw=4),
                Line2D([0], [0], color='C3', lw=4),
                Line2D([0], [0], color='C4', lw=4)]

ax.legend(custom_lines, ['PPO', 'A2C', 'DDPG', 'SAC', 'TD3'])
ax.set_ylim(top=1000, bottom=0)
ax.grid(alpha=0.5)
ax.set_xlabel('episodes')
ax.set_ylabel('reward')
plt.tight_layout()
plt.savefig(Path(f'graphics/{name}_optimization_intermediates.svg'))
plt.close()
