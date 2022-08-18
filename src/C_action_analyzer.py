# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:15:32 2022

@author: GEr
"""

import numpy as np
from sklearn.decomposition import PCA
from stable_baselines3 import PPO

from XX_maintenance_lib import plotter, CustomEnv

R = 3  # cutterhead radius [m]
D = 0.11  # cutter track spacing [m]
LIFE = 400000  # theoretical durability of one cutter [m]

STROKE_LENGTH = 1.8  # length of one stroke [m]
MAX_STROKES = 1000  # number of strokes per episode

N_STATES = 1000  # number of states to generate and predict

t_C_max = 75  # maximum time to change one cutter [min]

# total number of cutters
n_c_tot = int(round((R-D/2) / D, 0)) + 1
print(f'total number of cutters: {n_c_tot}\n')

cutter_positions = np.cumsum(np.full((n_c_tot), D)) - D/2

cutter_pathlenghts = cutter_positions * 2 * np.pi  # [m]

env = CustomEnv(n_c_tot, LIFE, MAX_STROKES, STROKE_LENGTH, cutter_pathlenghts,
                R, t_C_max)

pltr = plotter()
env = CustomEnv(n_c_tot, LIFE, MAX_STROKES, STROKE_LENGTH,
                cutter_pathlenghts, R, t_C_max)

agent = PPO.load('checkpoints/PPO_4400000_steps')
choices = [0, 0.25, 0.5, 0.75, 1]

states = [np.zeros(n_c_tot), np.ones(n_c_tot)]
actions = []

for i in range(N_STATES):
    state = np.random.choice(choices, size=n_c_tot, replace=True)
    # check if the state is already in states
    if any((state == x).all() for x in states) is not True:
        states.append(state)
        actions.append(agent.predict(state, deterministic=True)[0])
    else:
        print('duplicate state')

pca = PCA(n_components=2)

# env.implement_action(action, np.zeros(n_c_tot))
# print(env.replaced_cutters)
# print(env.moved_cutters)

# pltr.action_visualization(actions[0], n_c_tot, binary=False)

