# -*- coding: utf-8 -*-
"""
Code for the paper:

Towards smart TBM cutter changing with reinforcement learning (working title)
Georg H. Erharter, Tom F. Hansen, Thomas Marcher, Amund Bruland
JOURNAL NAME
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Code that analyzes / visualizes the log of an ongoing or finished OPTUNA
optimization study.

code contributors: Georg H. Erharter, Tom F. Hansen
"""

# import ipdb
import joblib
import optuna
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import yaml

from XX_plotting import Plotter

###############################################################################
# Constants and fixed variables

# name of study object
STUDY = 'A2C_2022_11_30_study'  # 'PPO_2022_09_27_study' 'DDPG_2022_10_03_study' 'A2C_2022_11_30_study' 'TD3_2022_09_27_study'
# folder where study database is located
FOLDER_DB = 'results'  # 'P:/2022/00/20220043/Calculations', 'results'
# folder wehere records of individual runs are located
FOLDER_INDIVIDUALS = 'optimization'  # f'P:/2022/00/20220043/Calculations/{STUDY}' 'optimization'
agent = STUDY.split('_')[0]
FILETYPE_TO_LOAD = "db"

###############################################################################
# processing

pltr = Plotter()

# load data from completed OPTUNA study
if FILETYPE_TO_LOAD == "db":
    db_path = f"{FOLDER_DB}/{STUDY}.db"
    db_file = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=STUDY, storage=db_file)
elif FILETYPE_TO_LOAD == "pkl":
    study = joblib.load(f"{FOLDER_DB}/{STUDY}.pkl")
else:
    raise ValueError(f"{FILETYPE_TO_LOAD} is not a valid filetype. "
                     "Valid filetypes are: db, pkl")

df_study: pd.DataFrame = study.trials_dataframe()
# drop nan that come from runs with default params
df_study.dropna(subset=['params_learning_rate'], inplace=True)

print(STUDY)
print(df_study.tail(n=25))
print(df_study['state'].value_counts())

# some cleaning
if "params_action_noise" in df_study.columns:
    le_noise = LabelEncoder()
    df_study["params_action_noise"] = le_noise.fit_transform(df_study["params_action_noise"])
else:
    le_noise = None
if "params_activation_fn" in df_study.columns:
    le_activation = LabelEncoder()
    df_study["params_activation_fn"] = le_activation.fit_transform(df_study["params_activation_fn"])
if "params_lr_schedule" in df_study.columns:
    le_schedule = LabelEncoder()
    df_study["params_lr_schedule"] = le_schedule.fit_transform(df_study["params_lr_schedule"])

# print values of best trial in study
trial = study.best_trial
print('\nHighest reward: {}'.format(trial.value))
print("Best hyperparameters:\n {}".format(trial.params))

print("Saving best parameters to a yaml_file")
with open(
    f"{FOLDER_DB}/{STUDY}_best_params_{study.best_value: .2f}.yaml", "w"
) as file:
    yaml.dump(study.best_params, file)

params = [p for p in df_study.columns if "params_" in p]

# replance NaN with "None"
if agent == 'SAC' or agent == 'DDPG' or agent == 'TD3':
    df_study['params_action_noise'].fillna(value='None', inplace=True)

###############################################################################
# different visualizations of OPTUNA optimization

# plot that shows the progress of the optimization over the individual trials
pltr.custom_optimization_history_plot(df_study,
                                      savepath=f'graphics/{STUDY}_optimization_progress.svg')

# scatterplot of indivdual hyperparameters vs. reward
pltr.custom_slice_plot(df_study, params, le_activation=le_activation,
                       le_noise=le_noise, le_schedule=le_schedule,
                       savepath=f'graphics/{STUDY}_optimization_scatter.svg')

# plot intermediate steps of the training paths
pltr.custom_intermediate_values_plot(agent, folder=FOLDER_INDIVIDUALS,
                                     print_thresh=940, mode='eval',
                                     savepath=f'graphics/{STUDY}_optimization_interms.svg')

pltr.custom_parallel_coordinate_plot(df_study, params, le_activation,
                                     savepath=f'graphics/{STUDY}_parallel_plot.svg')
