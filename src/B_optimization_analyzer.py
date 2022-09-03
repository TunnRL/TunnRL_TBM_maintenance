# -*- coding: utf-8 -*-
"""
Towards optimized TBM cutter changing policies with reinforcement learning
G.H. Erharter, T.F. Hansen
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Script that analyzes / visualizes the log of an OPTUNA study

Created on Thu Apr 14 13:28:07 2022
code contributors: Georg H. Erharter, Tom F. Hansen
"""

# import ipdb
import joblib
import optuna
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from XX_plotting import Plotter


###############################################################################
# Constants and fixed variables

STUDY = "A2C_2022_08_21_study"
agent = STUDY.split("_")[0]
FILETYPE_TO_LOAD = "db"

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
    raise ValueError(
        f"{FILETYPE_TO_LOAD} is not a valid filetype. " "Valid filetypes are: db, pkl"
    )

df_study: pd.DataFrame = study.trials_dataframe()

print(df_study.tail(n=25))

# some cleaning
if "params_action_noise" in df_study.columns:
    le_noise = LabelEncoder()
    df_study["params_action_noise"] = le_noise.fit_transform(
        df_study["params_action_noise"]
    )
else:
    le_noise = None
if "params_activation_fn" in df_study.columns:
    le_activation = LabelEncoder()
    df_study["params_activation_fn"] = le_activation.fit_transform(
        df_study["params_activation_fn"]
    )

# print values of best trial in study
trial = study.best_trial
print("\nHighest reward: {}".format(trial.value))
print("Best hyperparameters:\n {}".format(trial.params))


# optuna.importance.get_param_importances(study)

params = [p for p in df_study.columns if "params_" in p]

# replance NaN with "None"
if agent == "SAC" or agent == "DDPG" or agent == "TD3":
    df_study["params_action_noise"].fillna(value="None", inplace=True)

###############################################################################
# different visualizations of OPTUNA optimization

Plotter.custom_parallel_coordinate_plot(
    df_study, params, le_activation, savepath=f"graphics/{STUDY}_parallel_plot.svg"
)

# plot that shows the progress of the optimization over the individual trials
Plotter.custom_optimization_history_plot(
    df_study, savepath=f"graphics/{STUDY}_optimization_progress.svg"
)

# scatterplot of indivdual hyperparameters vs. reward
Plotter.custom_slice_plot(
    df_study,
    params,
    le_activation=le_activation,
    le_noise=le_noise,
    savepath=f"graphics/{STUDY}_optimization_scatter.svg",
)

# plot intermediate steps of the training paths
Plotter.custom_intermediate_values_plot(
    agent, folder="optimization", savepath=f"graphics/{STUDY}_optimization_interms.svg"
)
