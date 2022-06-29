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
from tensorforce.environments import Environment

from XX_maintenance_lib import maintenance, CustomEnv, plotter, optimization

###############################################################################
# Constants and fixed variables

R = 3  # cutterhead radius [m]
D = 0.11  # cutter track spacing [m]
LIFE = 400000  # theoretical durability of one cutter [m]

STROKE_LENGTH = 1.8  # length of one stroke [m]
MAX_STROKES = 1000  # number of strokes per episode

EPISODES = 10_000  # episodes to train for
CHECKPOINT = 100  # checkpoints every X episodes
REWARD_MODE = 1  # one of three different reward modes needs to be chosen

t_I = 25  # time for inspection of cutterhead [min]
t_C_max = 75  # maximum time to change one cutter [min]
K = 1.25  # factor controlling change time of cutters

AGENT = 'ppo'  # name of the agent

OPTIMIZATION = False  # Flag that indicates whether or not to run an OPTUNA optimization or a "main run" with fixed hyperparameters
STUDY = '2022_04_21_study'  # name of the study if OPTIMIZATION = True


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
                STROKE_LENGTH, cutter_pathlenghts, REWARD_MODE, R)

environment = Environment.create(environment=env,
                                 max_episode_timesteps=MAX_STROKES)

# Instantiate optimization function
optim = optimization(n_c_tot, environment, EPISODES, CHECKPOINT, AGENT,
                     OPTIMIZATION)


###############################################################################
# either run a study for hyperparameter optimization or an optimized main run

if OPTIMIZATION is True:  # study
    try:  # evtl. load already existing study if one exists
        study = joblib.load(Path(f'results/{STUDY}.pkl'))
        print('prev. study loaded')
    except FileNotFoundError:  # or create a new study
        study = optuna.create_study(direction='maximize')
        print('new study created')
    # the OPTUNA study is then run in a loop so that intermediate results are
    # saved and can be checked every 2 trials
    for _ in range(200):  # save every second study
        study.optimize(optim.objective, n_trials=2)
        joblib.dump(study, Path(f"results/{STUDY}.pkl"))

        df = study.trials_dataframe()
        df.to_csv(Path(f'results/{STUDY}.csv'))
    # print results of study
    trial = study.best_trial
    print('\nhighest reward: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
else:  # main run
    print('new main run with optimized parameters started')
    study = joblib.load(fr'results\{STUDY}.pkl')
    score = optim.objective(study.best_trial)
