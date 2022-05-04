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

name = '2022_04_21_study'

###############################################################################
# load data

STUDY = fr"results\{name}.pkl"
DF = fr'results\{name}.csv'

df_study = pd.read_csv(DF)

study = joblib.load(STUDY)


###############################################################################
# processing

df_study.dropna(inplace=True)

trial = study.best_trial
print('\nhighest reward: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))


###############################################################################
# visualization

# progress plot
values_max = []
for i, value in enumerate(df_study['value']):
    if i == 0:
        values_max.append(value)
    else:
        if value > values_max[int(i-1)]:
            values_max.append(value)
        else:
            values_max.append(values_max[int(i-1)])

fig, ax = plt.subplots()
ax.scatter(df_study[df_study['state'] == 'COMPLETE']['number'],
            df_study[df_study['state'] == 'COMPLETE']['value'],
            s=50, color='grey', edgecolor='black', label='complete')
ax.scatter(df_study[df_study['state'] == 'PRUNED']['number'],
            df_study[df_study['state'] == 'PRUNED']['value'],
            s=50, color='red', edgecolor='black', label='pruned')
ax.plot(df_study['number'], values_max, color='black')
ax.grid(alpha=0.5)
ax.set_xlabel('trial number')
ax.set_ylabel('reward')
# ax.legend()
plt.tight_layout()
plt.savefig(fr'graphics\{name}_optimization_progress.pdf')
plt.close()

# indivdual parameter plot
PARAMS = ['params_batch_size', 'params_discount', 
          'params_entropy_regularization', 'params_l2_regularization',
          'params_learning rate', 'params_likelihood_ratio_clipping',
          'params_subsampl. fraction', 'params_variable_noise']

fig = plt.figure(figsize=(3*len(PARAMS), 4))

for i, param in enumerate(PARAMS):
    ax = fig.add_subplot(1, len(PARAMS), i+1)
    ax.scatter(df_study[df_study['state'] == 'COMPLETE'][param],
                df_study[df_study['state'] == 'COMPLETE']['value'],
                s=50, color='grey', edgecolor='black', label='complete',
                alpha=0.5)
    ax.scatter(df_study[df_study['state'] == 'PRUNED'][param],
                df_study[df_study['state'] == 'PRUNED']['value'],
                s=50, color='red', edgecolor='black', label='pruned',
                alpha=0.5)
    ax.set_xlabel(param[7:])
    ax.grid(alpha=0.5)
    if i == 0:
        ax.set_ylabel('reward')
        # ax.legend()
    else:
        ax.tick_params(axis='y', which='both', length=0, labelsize=0)
    if param == 'params_learning rate':
        ax.set_xscale('log')

plt.tight_layout()
plt.savefig(fr'graphics\{name}_optimization_scatter.pdf')
plt.close()

# intermediate runs plot
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
plt.savefig(fr'graphics\{name}_optimization_intermediates.pdf')
plt.close()
