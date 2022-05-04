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
import numpy as np
import pandas as pd


###############################################################################
# Constants and fixed variables

name = '2022_04_21_study'

###############################################################################
# processing

# load data from completed OPTUNA study
STUDY = fr"results\{name}.pkl"
DF = fr'results\{name}.csv'
df_study = pd.read_csv(DF)
study = joblib.load(STUDY)

df_study.dropna(inplace=True)

trial = study.best_trial
print('\nhighest reward: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

###############################################################################
# different visualizations of OPTUNA optimization

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
plt.savefig(fr'graphics\{name}_optimization_progress.svg')
plt.close()

# scatterplot of indivdual hyperparameters vs. reward
PARAMS = ['params_batch_size', 'params_discount', 
          'params_entropy_regularization', 'params_l2_regularization',
          'params_learning rate', 'params_likelihood_ratio_clipping',
          'params_subsampl. fraction', 'params_variable_noise']

fig = plt.figure(figsize=(7.126, 4))  # 3*len(PARAMS), 4

for i, param in enumerate(PARAMS):
    ax = fig.add_subplot(2, int(len(PARAMS)/2), i+1)
    # only scatter complete studies -> see above
    ax.scatter(df_study[df_study['state'] == 'COMPLETE'][param],
               df_study[df_study['state'] == 'COMPLETE']['value'],
               s=20, color='grey', edgecolor='black', alpha=0.5)
    ax.grid(alpha=0.5)
    if i < int(len(PARAMS)/2):
        ax.set_title(param[7:], fontsize=10)
    else:
        ax.set_xlabel(param[7:])
    if i == 0 or i == int(len(PARAMS)/2):
        ax.set_ylabel('reward')
    else:
        ax.tick_params(axis='y', which='both', length=0, labelsize=0)
    if param == 'params_learning rate':
        ax.set_xscale('log')

plt.tight_layout()
plt.savefig(fr'graphics\{name}_optimization_scatter.svg')
plt.close()

# plot of the progress of individual runs
fig, ax = plt.subplots(figsize=(10, 8))

for trial in study.trials:
    eps = list(trial.intermediate_values.keys())
    vals = list(trial.intermediate_values.values())
    
    if len(eps) == 1:
        ax.plot(eps, vals, alpha=0.5, color='black')
    else:
        if min(np.diff(vals)) < 0:
            ax.plot(eps, vals, alpha=0.5, color='black')
        else:
            ax.plot(eps, vals, alpha=0.5, color='black')

ax.set_ylim(top=1, bottom=0)
ax.grid(alpha=0.5)
ax.set_xlabel('episodes')
ax.set_ylabel('reward')
plt.tight_layout()
plt.savefig(fr'graphics\{name}_optimization_intermediates.svg')
plt.close()
