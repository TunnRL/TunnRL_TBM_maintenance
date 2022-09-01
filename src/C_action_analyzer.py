# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:15:32 2022

@author: GEr
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from stable_baselines3 import PPO, TD3
import plotly.express as px

from XX_plotting import Plotter
from XX_TBM_environment import CustomEnv

R = 3  # cutterhead radius [m]
D = 0.11  # cutter track spacing [m]
LIFE = 400000  # theoretical durability of one cutter [m]

STROKE_LENGTH = 1.8  # length of one stroke [m]
MAX_STROKES = 1000  # number of strokes per episode

N_STATES = 5_000  # number of states to generate and predict

BROKEN_CUTTERS_THRESH = 0.5

# total number of cutters
n_c_tot = int(round((R-D/2) / D, 0)) + 1
print(f'total number of cutters: {n_c_tot}\n')

cutter_positions = np.cumsum(np.full((n_c_tot), D)) - D/2

cutter_pathlenghts = cutter_positions * 2 * np.pi  # [m]

env = CustomEnv(n_c_tot, LIFE, MAX_STROKES, STROKE_LENGTH, cutter_pathlenghts,
                R, BROKEN_CUTTERS_THRESH)

pltr = Plotter()

choices = np.linspace(0, 1, num=5)

states = [np.zeros(n_c_tot), np.ones(n_c_tot)]
n_broken_cutters = [n_c_tot, 0]
avg_cutter_life = [0, 1]

i = 0
while len(states) < N_STATES:
    state = np.random.choice(choices, size=n_c_tot, replace=True)

    if any((state == x).all() for x in states) is not True:
        broken_cutters = len(np.where(state == 0)[0])
        if broken_cutters / n_c_tot < BROKEN_CUTTERS_THRESH:
            if i % 50 == 0:
                print(i)
            states.append(state)
            n_broken_cutters.append(broken_cutters)
            avg_cutter_life.append(np.mean(state))
            i += 1
    else:
        print('duplicate state')

print(f'{len(states)} states generated')

model = 'TD3_11bd6ce5-92d5-4b95-befb-e82d23eae32e'  # TD3_11bd6ce5-92d5-4b95-befb-e82d23eae32e, TD3_9d32568f-b436-4155-832b-a2a4d3b2c909, TD3_bc9af932-d658-4b11-b157-01be25e9afa1
agent = TD3.load(f'optimization/{model}/best_model')

actions = []
n_replaced_cutters = []
n_moved_cutters = []

for state in states:
    action = agent.predict(state, deterministic=True)[0]
    actions.append(agent.predict(state, deterministic=True)[0])
    env._implement_action(action, np.zeros(n_c_tot))
    n_replaced_cutters.append(len(env.replaced_cutters))
    n_moved_cutters.append(len(env.moved_cutters))

actions = np.array(actions)

perp = 30 if N_STATES < 3000 else N_STATES / 100
reducer = TSNE(n_components=2, perplexity=perp,
               init='random', learning_rate='auto', n_jobs=-1)

actions_reduced = reducer.fit_transform(actions)

df = pd.DataFrame({'x': actions_reduced[:, 0], 'y': actions_reduced[:, 1],
                   'broken cutters': n_broken_cutters,
                   'avg. cutter life': avg_cutter_life,
                   'replaced cutters': n_replaced_cutters,
                   'moved cutters': n_moved_cutters,
                   'state': states})

fig = px.scatter(df, x='x', y='y', color='broken cutters',
                 hover_data={'x':False, 'y':False, 'state':True,
                             'replaced cutters': True, 'moved cutters':True,
                             'avg. cutter life': True})
fig.update_layout(xaxis_title=None, yaxis_title=None)
fig.write_html("graphics/file.html")


fig, ax = plt.subplots(figsize=(9, 9))
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)

im = ax.scatter(actions_reduced[:, 0], actions_reduced[:, 1],
                c=n_moved_cutters, cmap='turbo')
fig.colorbar(im, cax=cax, orientation='vertical',
             label='number of moved cutters')
ax.set_title('TSNE mapping of actions')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.savefig(f'graphics/tsne.png', dpi=300)