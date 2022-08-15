# -*- coding: utf-8 -*-
"""
Towards optimized TBM cutter changing policies with reinforcement learning
G.H. Erharter, T.F. Hansen
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Script that analyzes / visualizes the log of an OPTUNA study

Created on Thu Apr 14 13:28:07 2022
code contributors: Georg H. Erharter, Tom F. Hansen
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import optuna
import joblib


from D_training_path_analyzer import training_path

###############################################################################
# Constants and fixed variables
STUDY = 'PPO_2022_08_10_study'  # DDPG_2022_07_27_study TD3_2022_07_27_study
agent = STUDY.split('_')[0]
FILETYPE_TO_LOAD = "db"  # "pkl" "db"

###############################################################################
# processing

# load data from completed OPTUNA study
if FILETYPE_TO_LOAD == "db":
    db_path = f"results/{STUDY}.db"
    db_file = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=STUDY, storage=db_file)
elif FILETYPE_TO_LOAD == "pkl":
    study = joblib.load(f"results/{STUDY}.pkl")
else:
    raise ValueError(f"{FILETYPE_TO_LOAD} is not a valid filetype. "
                     "Valid filetypes are: db, pkl")

df_study = study.trials_dataframe()
print(df_study.tail(n=20))
# print values of best trial in study
trial = study.best_trial
print('\nHighest reward: {}'.format(trial.value))
print("Best hyperparameters:\n {}".format(trial.params))


# optuna.importance.get_param_importances(study)

params = [p for p in df_study.columns if "params_" in p]

params_ = params[2:]

fig, ax = plt.subplots(figsize=(18, 9))

mins = df_study[params_].min().values
f = df_study[params_].values-mins
maxs = np.max(f, axis=0)

cmap = matplotlib.cm.get_cmap('cividis')
norm = matplotlib.colors.Normalize(vmin=600, vmax=df_study['value'].max())

for t in range(len(df_study)):
    df_temp = df_study.sort_values(by='value').iloc[t]
    x = np.arange(len(params_))
    y = df_temp[params_].values

    y = y - mins
    y = y / maxs

    if df_temp['state'] == 'FAIL':
        ax.plot(x, y, c='red', alpha=0.5)
    elif df_temp['state'] == 'RUNNING':
        pass
    else:
        
        if df_temp['value'] < 600:
            ax.plot(x, y, c=cmap(norm(df_temp['value'])), alpha=0.2)
        else:
            ax.plot(x, y, c=cmap(norm(df_temp['value'])), alpha=1, zorder=10)

ax.scatter(x, np.zeros(x.shape), color='black')
ax.scatter(x, np.ones(x.shape), color='black')

for i in range(len(x)):
    ax.text(x=x[i], y=-0.01, s=np.round(df_study[params_].min().values[i], 4),
            horizontalalignment='center', verticalalignment='top')
    ax.text(x=x[i], y=1.01, s=np.round(df_study[params_].max().values[i], 4),
            horizontalalignment='center', verticalalignment='bottom')

ax.set_xticks(x)
ax.set_yticks([0, 1])
ax.set_xticklabels([p[7:] for p in params_])
ax.set_yticklabels([])
ax.grid()

plt.tight_layout()

# print(ghjkl)
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
# plt.savefig(f'graphics/{STUDY}_optimization_progress.svg')
# plt.close()

#####
# scatterplot of indivdual hyperparameters vs. reward

# replance NaN with "None"
df_study[f'params_action_noise'].fillna(value='None', inplace=True)

fig = plt.figure(figsize=(18, 9))

for i, param in enumerate(params):
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
    ax.set_xlabel(param[7:])
    if i == 0:
        ax.set_ylabel('reward')

    if 'learning rate' in param:
        ax.set_xscale('log')
ax.legend()
fig.suptitle(STUDY.split('_')[0])

plt.tight_layout()
# plt.savefig(f'graphics/{STUDY}_optimization_scatter.svg')
# plt.close()

#####
# plot intermediate steps of the training paths
training_path(agent, folder='optimization')#,
              # savepath=f'graphics/{STUDY}_optimization_interms.svg')
