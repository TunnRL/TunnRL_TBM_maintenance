# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 12:46:42 2021

@author: Schorsch
"""

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorforce.agents import Agent
from tensorforce.environments import Environment

from maintenance_lib import maintenance, CustomEnv, SimpleAgent


###############################################################################
# cosntants etc
R = 3  # cutterhead radius [m]
D = 0.11  # cutter track spacing [m]
LIFE = 400000  # theoretical durability of one cutter [m]

STROKE_LENGTH = 1.8  # length of one stroke [m]
MAX_STROKES = 500  # number of stroke / maintenance intervalls of one episode
INTERVALL = 20  # maintenance all XX strokes

EPISODES = 50_000  # episodes to train for
CHECKPOINT = 2_000  # checkpoints every X episodes
PRETRAIN_EPISODES = 0
REWARD_MODE = 2

t_I = 25  # time for inspection of cutterhead [min]
t_C_max = 75  # maximum time to change one cutter [min]
K = 1.25  # factor controlling change time of cutters

AGENT = 'ppo'  # 'ppo', 'a2c', 'load', 'simple'
FNAME = None

# total number of cutters
n_c_tot = int(round((R-D/2) / D, 0)) + 1
print(f'agent: {AGENT}; total number of cutters: {n_c_tot}\n')
# Instantiate modules
m = maintenance()


###############################################################################
# computed variables

cutter_positions = np.cumsum(np.full((n_c_tot), D)) - D/2

cutter_pathlenghts = cutter_positions * 2 * np.pi  # [m]

t_maint_max = m.maintenance_cost(t_I, t_C_max, n_c_tot, K)[0]

env = CustomEnv(n_c_tot, LIFE, t_I, t_C_max, K, t_maint_max, MAX_STROKES,
                STROKE_LENGTH, cutter_pathlenghts, INTERVALL, REWARD_MODE)

environment = Environment.create(environment=env,
                                 max_episode_timesteps=MAX_STROKES)

###############################################################################
# agent selection

# TODO check out OPTUNA

if AGENT == 'a2c':
    agent = Agent.create(agent='a2c',
                         actions=dict(type='int', shape=(n_c_tot,),
                                      num_values=2),
                         environment=environment, batch_size=100,
                         learning_rate=dict(initial_value=1e-4, type='linear',
                                            unit='episodes', num_steps=4000,
                                            final_value=1e-5),
                         exploration=dict(initial_value=1.0, type='linear',
                                          unit='episodes', num_steps=2000,
                                          final_value=0.01),
                         entropy_regularization=0.5, l2_regularization=0.8)
    ep, train_start = 0, datetime.now().strftime("%Y%m%d-%H%M%S")
elif AGENT == 'ppo':
    agent = Agent.create(agent='ppo',
                         actions=dict(type='int', shape=(n_c_tot,),
                                      num_values=2),
                         environment=environment, batch_size=20,
                         # exploration=dict(initial_value=1.0, type='linear',
                         #                  unit='episodes', num_steps=2_000,
                         #                  final_value=0.01),
                         learning_rate=1e-2,
                         subsampling_fraction=0.33,
                         likelihood_ratio_clipping=0.25,
                         l2_regularization=0.0,
                         entropy_regularization=0.01)
    ep, train_start = 0, datetime.now().strftime("%Y%m%d-%H%M%S")
elif AGENT == 'simple':
    agent = SimpleAgent()
    ep, train_start = 0, datetime.now().strftime("%Y%m%d-%H%M%S")
elif AGENT == 'load':
    agent = Agent.load(directory='checkpoints', filename=FNAME, format='hdf5',
                       environment=environment)
    AGENT, train_start, ep = FNAME.split('_')
    ep = int(ep)

###############################################################################
# pretraining on simple strategy

print('start pretraining...')

for p_ep in range(PRETRAIN_EPISODES):
    episode_states = list()
    episode_internals = list()
    episode_actions = list()
    episode_terminal = list()
    episode_reward = list()

    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False

    s_agent = SimpleAgent()

    while not terminal:
        episode_states.append(states)
        episode_internals.append(internals)
        actions, internals = agent.act(states=states, internals=internals,
                                       independent=True)
        actions = s_agent.act(states=states, internals=internals,
                              independent=True)
        episode_actions.append(actions)
        states, terminal, reward = environment.execute(actions=actions)
        episode_terminal.append(terminal)
        episode_reward.append(reward)

    agent.experience(states=episode_states, internals=episode_internals,
                     actions=episode_actions, terminal=episode_terminal,
                     reward=episode_reward)
    agent.update()
    if p_ep % 50 == 0:
        print(f'... pretraining ep: {p_ep} ...')

print('...pretraining finished\n')

###############################################################################
# main loop

if FNAME is not None:
    df = pd.read_csv(fr'checkpoints/{FNAME}.csv')
    ep += 1
else:
    df = pd.DataFrame({'episode': [], 'min_rewards': [], 'max_rewards': [],
                       'avg_rewards': [], 'min_brokens': [], 'avg_brokens': [],
                       'max_brokes': [], 'avg_changes_per_interv': []})

summed_actions = []

for ep in range(ep, ep+EPISODES):
    state = environment.reset()
    terminal = False

    actions = []
    states = [state]  # = cutter lifes / LIFE
    rewards = []
    broken_cutters = []

    while not terminal:
        broken_cutters.append(len(np.where(state == 0)[0]))

        action = agent.act(states=state)  # , independent=True)
        state, terminal, reward = environment.execute(actions=action)
        agent.observe(terminal=terminal, reward=reward)

        actions.append(action)
        states.append(state)
        rewards.append(reward)

    avg_reward = round(np.mean(rewards), 3)
    avg_broken = round(np.mean(broken_cutters))
    avg_changed = np.mean(np.sum(np.where(np.vstack(actions) > .5, 1, 0), 1))

    print(f'{AGENT}; ep: {ep}, broken cutters: {avg_broken}, reward: {avg_reward}')

    df_temp = pd.DataFrame({'episode': [ep], 'min_rewards': [min(rewards)],
                            'max_rewards': [max(rewards)],
                            'avg_rewards': [avg_reward],
                            'min_brokens': [min(broken_cutters)],
                            'avg_brokens': [avg_broken],
                            'max_brokes': [max(broken_cutters)],
                            'avg_changes_per_interv': [avg_changed]})
    df = df.append(df_temp, ignore_index=True)
    summed_actions.append(np.sum(actions, axis=0))

    # TODO avg changes / intervall throughout ep

    if ep % CHECKPOINT == 0 and ep != 0:
        name = f'{AGENT}_{train_start}_{ep}'

        states_arr = np.vstack(states[:-1])
        actions_arr = np.vstack(actions)
        actions_arr = np.where(actions_arr > .5, 1, 0)  # binarize
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 8))
        for cutter in range(states_arr.shape[1]):
            ax1.plot(np.arange(states_arr.shape[0]), states_arr[:, cutter])

        for intervall in range(actions_arr.shape[0]):
            ax2.scatter(np.full((actions_arr.shape[1]), intervall),
                        np.arange(actions_arr.shape[1]),
                        c=actions_arr[intervall, :], cmap='Greys', s=10,
                        edgecolor='black', lw=.1, vmin=0, vmax=1)

        ax1.set_xlim(left=-1, right=actions_arr.shape[0]+1)
        ax2.set_xlim(left=-1, right=actions_arr.shape[0]+1)
        plt.tight_layout()
        plt.savefig(fr'checkpoints\{name}_sample.png')
        plt.close()

        fig, ax = plt.subplots()
        ax.bar(np.arange(len(action)), np.sum(summed_actions, axis=0))
        ax.set_xlabel('cutter positions')
        ax.set_ylabel('number of changes')
        ax.set_title(name)
        plt.tight_layout()
        plt.savefig(fr'checkpoints\{name}_bars.png')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(np.vstack(summed_actions).T, aspect='auto')
        ax.set_xlabel('episodes')
        ax.set_title(name)
        plt.tight_layout()
        plt.savefig(fr'checkpoints\{name}_actions.png')
        plt.close()

        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))

        ax0.plot(df['episode'], df['avg_changes_per_interv'], color='black')
        ax0.grid(alpha=0.5)
        ax0.set_xlim(left=0, right=len(df))
        ax0.set_ylabel('n cutter changes / stroke / episode')
        ax0.set_title(name)

        ax1.plot(df['episode'], df['min_brokens'],
                 color='black', alpha=0.5, label='min')
        ax1.plot(df['episode'], df['avg_brokens'],
                 color='black', label='avg')
        ax1.grid(alpha=0.5)
        ax1.legend()
        ax1.set_xlim(left=0, right=len(df))
        ax1.set_ylabel('broken cutters / stroke / episode')

        ax2.plot(df['episode'], df['avg_rewards'], color='black')
        ax2.set_xlim(left=0, right=len(df))
        ax2.grid(alpha=0.5)
        ax2.set_ylabel('reward / stroke / ep')
        ax2.set_xlabel('episodes')
        plt.tight_layout()
        plt.savefig(fr'checkpoints\{name}_progress.png')
        plt.close()

        agent.save(directory='checkpoints', filename=name, format='hdf5')
        df.to_csv(fr'checkpoints\{name}.csv', index=False)

# fig, ax = plt.subplots()

# ax.bar(cutter_positions, cutter_pathlenghts, width=D)
# ax.set_xlabel('position from center [m]')
# ax.set_ylabel('rolling path length per cutterhead rotation [m]')
