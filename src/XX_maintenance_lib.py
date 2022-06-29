# -*- coding: utf-8 -*-
"""
Towards optimized TBM cutter changing policies with reinforcement learning
G.H. Erharter, T.F. Hansen
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Custom library that contains different classes for the reinforcement learning
based cutter changing environment

Created on Sat Oct 30 12:57:51 2021
code contributors: Georg H. Erharter, Tom F. Hansen
"""

from datetime import datetime
import matplotlib.cm as mplcm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import optuna
import pandas as pd
from pathlib import Path
from tensorforce.agents import Agent
from tensorforce.environments import Environment


class plotter:
    '''class that contains function to visualzie the progress of the
    training and / or individual samples of it'''

    def __init__(self):
        pass

    def sample_ep_plot(self, states, actions, rewards, ep, savename):
        '''plot of different recordings of one exemplary episode'''

        cmap = mplcm.get_cmap('viridis')

        states_arr = np.vstack(states[:-1])
        actions_arr = np.vstack(actions)
        actions_arr = np.where(actions_arr > .5, 1, 0)  # binarize

        fig = plt.figure(tight_layout=True, figsize=(10.236, 7.126))
        gs = gridspec.GridSpec(nrows=4, ncols=2, width_ratios=[5, 1])

        # cutter life line plot
        ax = fig.add_subplot(gs[0, 0])
        for cutter in range(states_arr.shape[1]):
            rgba = cmap(cutter / states_arr.shape[1])
            ax.plot(np.arange(states_arr.shape[0]), states_arr[:, cutter],
                    color=rgba, label=cutter)
        h, l = ax.get_legend_handles_labels()
        ax.set_xlim(left=-1, right=actions_arr.shape[0]+1)
        ax.set_ylim(top=1.05, bottom=-0.05)
        ax.set_title(f'episode {ep}', fontsize=10)
        ax.set_ylabel('cutter life')
        ax.set_xticklabels([])
        ax.grid(alpha=0.5)

        # dedicated subplot for a legend to first axis
        lax = fig.add_subplot(gs[0, 1])
        lax.legend(h, l, borderaxespad=0, ncol=3, loc='upper left',
                   fontsize=7.5)
        lax.axis('off')

        # show which cutters were selected for change after every stroke
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(actions_arr.T, aspect='auto', cmap='Greys',
                  interpolation='none')
        ax.set_xlim(left=-1, right=actions_arr.shape[0]+1)
        ax.set_ylim(bottom=-1, top=actions_arr.shape[1])
        ax.set_ylabel('changes on\ncutter positions')
        ax.set_xticklabels([])
        ax.grid(alpha=0.5)

        # barchart of frequently changed cutter positions
        ax = fig.add_subplot(gs[1, 1])
        ax.barh(np.arange(actions_arr.shape[1]), np.sum(actions_arr, axis=0),
                color='grey', edgecolor='black')
        ax.set_ylim(bottom=-1, top=actions_arr.shape[1]+1)
        ax.grid()
        ax.set_yticklabels([])
        ax.set_ylabel('cutter changes\nper episode')
        ax.yaxis.set_label_position('right')

        # bar plot that shows how many cutters were changed
        ax = fig.add_subplot(gs[2, 0])
        ax.bar(x=np.arange(actions_arr.shape[0]),
               height=np.sum(actions_arr, axis=1), color='grey')
        avg_changed = np.mean(np.sum(actions_arr, axis=1))
        ax.axhline(y=avg_changed, color='black')
        ax.text(x=950, y=avg_changed-avg_changed*0.05,
                s=f'avg. changes / stroke: {round(avg_changed, 2)}',
                color='black', va='top', ha='right', fontsize=7.5)
        ax.set_xlim(left=-1, right=actions_arr.shape[0]+1)
        ax.set_ylabel('cutter changes\nper stroke')
        ax.set_xticklabels([])
        ax.grid(alpha=0.5)

        # bar plot that shows the reward per stroke
        ax = fig.add_subplot(gs[3, 0])
        ax.bar(x=np.arange(len(rewards)), height=rewards, color='grey',
               lw=0.5)
        ax.axhline(y=np.mean(rewards), color='black')
        ax.text(x=950, y=np.mean(rewards)-0.05,
                s=f'avg. reward / stroke: {round(np.mean(rewards), 2)}',
                color='black', va='top', ha='right', fontsize=7.5)
        ax.set_ylim(bottom=0, top=1)
        ax.set_xlim(left=-1, right=actions_arr.shape[0]+1)
        ax.set_ylabel('reward / stroke')
        ax.set_xlabel('strokes')
        ax.grid(alpha=0.5)

        plt.tight_layout()
        # plt.savefig(Path(f'checkpoints/{savename}_sample.png'), dpi=600)
        plt.savefig(Path(f'checkpoints/{savename}_sample.svg'))
        plt.close()

    def environment_parameter_plot(self, savename, ep):
        '''plot that shows the generated TBM parameters of the episode'''
        x = np.arange(len(self.Jv_s))  # strokes
        # count broken cutters due to blocky conditions
        n_brokens = np.count_nonzero(self.brokens, axis=1)

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6, figsize=(12, 9))

        ax1.plot(x, self.Jv_s, color='black')
        ax1.grid(alpha=0.5)
        ax1.set_ylabel('Volumetric Joint count\n[joints / m3]')
        ax1.set_xlim(left=0, right=len(x))
        ax1.set_title(f'episode {ep}', fontsize=10)
        ax1.set_xticklabels([])

        ax2.plot(x, self.UCS_s, color='black')
        ax2.grid(alpha=0.5)
        ax2.set_ylabel('Rock UCS\n[MPa]')
        ax2.set_xlim(left=0, right=len(x))
        ax2.set_xticklabels([])

        ax3.plot(x, self.FPIblocky_s, color='black')
        ax3.hlines([50, 100, 200, 300], xmin=0, xmax=len(x), color='black',
                   alpha=0.5)
        ax3.set_ylim(bottom=0, top=400)
        ax3.set_ylabel('FPI blocky\n[kN/m/mm/rot]')
        ax3.set_xlim(left=0, right=len(x))
        ax3.set_xticklabels([])

        ax4.plot(x, self.TF_s, color='black')
        ax4.grid(alpha=0.5)
        ax4.set_ylabel('thrust force\n[kN]')
        ax4.set_xlim(left=0, right=len(x))
        ax4.set_xticklabels([])

        ax5.plot(x, self.penetration, color='black')
        ax5.grid(alpha=0.5)
        ax5.set_ylabel('penetration\n[mm/rot]')
        ax5.set_xlim(left=0, right=len(x))
        ax5.set_xticklabels([])

        ax6.plot(x, n_brokens, color='black')
        ax6.grid(alpha=0.5)
        ax6.set_ylabel('broken cutters\ndue to blocks')
        ax6.set_xlabel('strokes')
        ax6.set_xlim(left=0, right=len(x))

        plt.tight_layout()
        # plt.savefig(Path(f'checkpoints/{savename}_episode.png'), dpi=600)
        plt.savefig(Path(f'checkpoints/{savename}_episode.svg'))
        plt.close()


    def trainingprogress_plot(self, df, summed_actions, name):
        '''plot of different metrices of the whole training progress so far'''
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, ncols=1,
                                                 figsize=(7.126, 5))  # 12, 9

        ax0.imshow(np.vstack(summed_actions).T, aspect='auto', cmap='Greys_r',
                   interpolation='none')
        ax0.set_ylabel('actions on\ncutter positions')
        ax0.set_title(name, fontsize=10)
        ax0.set_xticklabels([])

        ax1.plot(df['episode'], df['avg_changes_per_interv'], color='black')
        ax1.grid(alpha=0.5)
        ax1.set_xlim(left=0, right=len(df))
        ax1.set_ylabel('avg. cutter\nchanges / stroke')
        ax1.yaxis.set_label_position('right')
        ax1.set_xticklabels([])

        ax2.plot(df['episode'], df['avg_brokens'], color='black')
        ax2.grid(alpha=0.5)
        ax2.set_xlim(left=0, right=len(df))
        ax2.set_ylabel('avg. n broken\ncutters / stroke')
        ax2.set_xticklabels([])

        ax3.plot(df['episode'], df['avg_rewards'], color='black')
        ax3.set_xlim(left=0, right=len(df))
        ax3.set_ylim(top=1, bottom=0)
        ax3.grid(alpha=0.5)
        ax3.set_ylabel('avg. reward\n/ stroke')
        ax3.yaxis.set_label_position('right')
        ax3.set_xlabel('episodes')

        plt.tight_layout()
        # plt.savefig(Path(f'checkpoints/{name}_progress.png'), dpi=600)
        plt.savefig(Path(f'checkpoints/{name}_progress.svg'))
        plt.close()


class SimpleAgent:
    '''agent that applies a simple / not smart policy:
    if cutter = broken -> change it.
    Can be used like a common tensorforce agent'''

    def act(self, states, internals=None, independent=None):
        return np.where(states <= 0, 1, 0)

    def observe(self, terminal, reward):
        pass

    def save(self, directory, filename, format):
        pass


class maintenance:
    '''class that contains functions that describe the maintenance effort of
    changing cutters on a TBM's cutterhead. Based on this the reward is
    computed'''

    def __init__(self):
        pass

    def maintenance_core_function(self, t_inspection, t_change_max,
                                  n_cutters, K):
        time_per_cutter = t_change_max - n_cutters * K  # [min]
        t_maint = t_inspection + n_cutters * time_per_cutter  # [min]
        return t_maint

    def maintenance_cost(self, t_inspection, t_change_max, n_c_to_change, K,
                         cap=True):
        # compute first derivative of function to get maximum number of cutters
        # beyond which change time flattens out
        max_c = t_change_max / (2*K)

        if n_c_to_change > max_c and cap is True:
            t_maint = self.maintenance_core_function(t_inspection,
                                                     t_change_max, max_c, K)
        else:
            t_maint = self. maintenance_core_function(t_inspection,
                                                      t_change_max,
                                                      n_c_to_change, K)

        return t_maint, max_c  # total time for maintenance [min]

    def reward(self, good_cutters, n_c_tot, t_maint, t_maint_max,
               n_c_to_change, mode, cap_zero=False, scale=False):
        '''function that computes the reward based on one of three options'''
        if mode == 1:  # complex reward featuring maintenance time function
            r = good_cutters / n_c_tot - t_maint / t_maint_max
        elif mode == 2:  # intermediate reward
            r = (good_cutters / n_c_tot) - (n_c_to_change / n_c_tot)
        elif mode == 3:  # simple reward only going for max good cutters
            r = good_cutters / n_c_tot

        if cap_zero is True:  # cap values below 0 to get reward in range 0-1
            r = 0 if r < 0 else r
        if scale is True:  # force reward into -1 - 1 range
            r = r * 2 - 1

        return r


class CustomEnv(Environment, maintenance, plotter):
    '''implementation of the custom environment that simulates the cutter wear
    and provides the agent with a state and reward signal'''
    # https://tensorforce.readthedocs.io/en/latest/basics/getting-started.html

    def __init__(self, n_c_tot, LIFE, t_I, t_C_max, K, t_maint_max,
                 MAX_STROKES, STROKE_LENGTH, cutter_pathlenghts, REWARD_MODE,
                 R):
        super().__init__()

        self.n_c_tot = n_c_tot  # total number of cutters
        self.LIFE = LIFE  # theoretical durability of one cutter [m]
        self.t_I = t_I  # time for inspection of cutterhead [min]
        self.t_C_max = t_C_max  # maximum time to change one cutter [min]
        self.K = K  # factor controlling change time of cutters
        self.t_maint_max = t_maint_max  # max. time for maintenance
        self.MAX_STROKES = MAX_STROKES  # number of strokes to simulate
        self.STROKE_LENGTH = STROKE_LENGTH  # length of one stroke [m]
        self.cutter_pathlenghts = cutter_pathlenghts  # rolling lengths [m]
        self.REWARD_MODE = REWARD_MODE  # which of 3 reward modes to use
        self.R = R  # radius of cutterhead [m]

    def states(self):
        '''return dictionary with information on how the state looks like'''
        return dict(type='float', shape=(self.n_c_tot,), min_value=0,
                    max_value=1)

    def execute(self, actions):
        '''main function that moves the environment one step further'''
        # replace cutters based on action of agent
        replace_ids = np.where(actions > 0.5)[0]
        self.state[replace_ids] = 1  # update state / cutter life

        # compute reward
        good_cutters = len(np.where(self.state > 0)[0])
        n_c_to_change = len(replace_ids)
        t_maint = self.maintenance_cost(self.t_I, self.t_C_max,
                                        n_c_to_change, self.K)[0]
        reward = self.reward(good_cutters=good_cutters, n_c_tot=self.n_c_tot,
                             t_maint=t_maint, t_maint_max=self.t_maint_max,
                             n_c_to_change=n_c_to_change,
                             mode=self.REWARD_MODE, cap_zero=True, scale=False)

        # update cutter life based on how much wear occurs
        p = self.penetration[self.epoch] / 1000  # [m/rot]
        rot_per_stroke = self.STROKE_LENGTH / p
        self.state = self.state - rot_per_stroke * (self.cutter_pathlenghts / self.LIFE)
        self.state = np.where(self.state <= 0, 0, self.state)

        # set cutter lifes to 0 based on blockyness
        self.state[np.where(self.brokens[self.epoch, :] == 1)[0]] = 0

        self.epoch += 1
        if self.epoch >= self.MAX_STROKES:
            terminal = True
        else:
            terminal = False

        return self.state, terminal, reward

    def rand_walk_with_bounds(self, n_dp):
        '''function generates a random walk within the boundaries 0 and 1'''
        bounds = .05

        x = [np.random.uniform(low=0, high=1, size=1)]

        for move in np.random.uniform(low=-bounds, high=bounds, size=n_dp):
            x_temp = x[-1] + move
            if x_temp <= 1 and x_temp >= 0:
                x.append(x_temp)
            else:
                x.append(x[-1] - move)

        return np.array(x[1:])

    def generate(self, Jv_low=0, Jv_high=22, UCS_center=80, UCS_range=30):
        '''function generates TBM recordings for one episode. Equations and
        models based on Delisio & Zhao (2014) - "A new model for TBM
        performance prediction in blocky rock conditions",
        http://dx.doi.org/10.1016/j.tust.2014.06.004'''

        Jv_s = self.rand_walk_with_bounds(self.MAX_STROKES) * (Jv_high - Jv_low) + Jv_low  # [joints / m3]
        UCS_s = UCS_center + self.rand_walk_with_bounds(self.MAX_STROKES) * UCS_range  # [MPa]

        # eq 9, Delisio & Zhao (2014) - [kN/m/mm/rot]
        FPIblocky_s = np.squeeze(np.exp(6) * Jv_s**-0.82 * UCS_s**0.17)

        brokens = np.zeros(shape=(self.MAX_STROKES, self.n_c_tot))

        for stroke in range(self.MAX_STROKES):
            # based on FPI blocky cutters have different likelyhoods to break
            if FPIblocky_s[stroke] > 200 and FPIblocky_s[stroke] <= 300:
                if np.random.randint(0, 100) == 0:
                    size = np.random.randint(1, 4)
                    selection = np.random.choice(np.arange(self.n_c_tot),
                                                 replace=False, size=size)
                    brokens[stroke, :][selection] = 1
            elif FPIblocky_s[stroke] > 100 and FPIblocky_s[stroke] <= 200:
                if np.random.randint(0, 50) == 0:
                    size = np.random.randint(1, 4)
                    selection = np.random.choice(np.arange(self.n_c_tot),
                                                 replace=False, size=size)
                    brokens[stroke, :][selection] = 1
            elif FPIblocky_s[stroke] >= 50 and FPIblocky_s[stroke] <= 100:
                if np.random.randint(0, 10) == 0:
                    size = np.random.randint(1, 4)
                    selection = np.random.choice(np.arange(self.n_c_tot),
                                                 replace=False, size=size)
                    brokens[stroke, :][selection] = 1
            elif FPIblocky_s[stroke] < 50:
                if np.random.randint(0, 100) == 0:
                    size = np.random.randint(1, 4)
                    selection = np.random.choice(np.arange(self.n_c_tot),
                                                 replace=False, size=size)
                    brokens[stroke, :][selection] = 1

        # eq 13, Delisio & Zhao (2014)
        TF_s = np.squeeze((-523 * np.log(Jv_s) + 2312) * (self.R*2))  # [kN]
        TF_s = np.where(TF_s > 20_000, 20_000, TF_s)

        # eq 7, Delisio & Zhao (2014)
        # TF_s already considers shield friction
        penetration = (TF_s / (self.R*2)) / FPIblocky_s  # [mm/rot]

        return Jv_s, UCS_s, FPIblocky_s, brokens, TF_s, penetration

    def reset(self):
        '''reset an environment to its beginning state'''
        # start with either new cutters, or all cutters broken
        self.state = np.full((self.n_c_tot), 1)  # start all cutters good
        # self.state = np.full((self.n_c_tot), 0)  # start all cutters broken

        self.epoch = 0  # reset epoch counter
        # generate new TBM data for episode
        self.Jv_s, self.UCS_s, self.FPIblocky_s, self.brokens, self.TF_s, self.penetration = self.generate()

        return self.state

    def save(self, AGENT, train_start, ep, states, actions, rewards, df,
             summed_actions, agent):
        '''save plots that show the current training progress, the current
        version of the agent and records of the current training progress'''
        name = f'{AGENT}_{train_start}_{ep}'

        self.sample_ep_plot(states, actions, rewards, ep, savename=name)

        self.environment_parameter_plot(savename=name, ep=ep)

        self.trainingprogress_plot(df, summed_actions, name)

        agent.save(directory='checkpoints', filename=name, format='hdf5')
        df.to_csv(Path(f'checkpoints/{name}.csv', index=False))


class optimization:

    def __init__(self, n_c_tot, environment, EPISODES, CHECKPOINT, AGENT, OPTIMIZATION):
        self.n_c_tot = n_c_tot
        self.environment = environment
        self.EPISODES = EPISODES
        self.CHECKPOINT = CHECKPOINT
        self.AGENT = AGENT
        self.OPTIMIZATION = OPTIMIZATION

    def objective(self, trial):
        '''objective function that runs the RL environment and agent for
        optimization'''
        print('\n')

        # initialize new series of episodes
        ep, train_start = 0, datetime.now().strftime("%Y%m%d-%H%M%S")
        # dataframe to collect episode recordings for later analyses
        df = pd.DataFrame({'episode': [], 'min_rewards': [], 'max_rewards': [],
                           'avg_rewards': [], 'min_brokens': [],
                           'avg_brokens': [], 'max_brokes': [],
                           'avg_changes_per_interv': []})

        # initialze an agent for a OPTUNA hyperparameter study
        agent = Agent.create(agent='ppo',
                             actions=dict(type='int', shape=(self.n_c_tot,),
                                          num_values=2),
                             environment=self.environment,
                             batch_size=trial.suggest_int('batch_size', 5, 40, step=5),
                             learning_rate=trial.suggest_float('learning rate', low=1e-4, high=1e-1, log=True),
                             subsampling_fraction=trial.suggest_float('subsampl. fraction', low=0.1, high=1),
                             likelihood_ratio_clipping=trial.suggest_float('likelihood_ratio_clipping', low=0.1, high=1),
                             l2_regularization=trial.suggest_float('l2_regularization', low=0.0, high=.6),
                             entropy_regularization=trial.suggest_float('entropy_regularization', low=0.0, high=0.6),
                             discount=trial.suggest_float('discount', low=0.0, high=1),
                             variable_noise=trial.suggest_float('variable_noise', low=0.0, high=1))

        summed_actions = []  # list to collect number of actions per episode
        # main loop that trains the agent throughout multiple episodes
        for ep in range(ep, ep+self.EPISODES+1):
            state = self.environment.reset()  # reset new environment
            terminal = False  # reset terminal flag

            actions = []  # collect actions per episode
            states = [state]  # collect states per episode
            rewards = []  # collect rewards per episode
            broken_cutters = []  # collect number of broken cutters per episode
            # loop that trains agent in "act-observe" style on one episode
            while not terminal:
                # collect number of broken cutters in curr. state
                broken_cutters.append(len(np.where(state == 0)[0]))
                # agent takes an action -> tells which cutters to replace
                action = agent.act(states=state)
                # environment gives new state signal, terminal flag and reward
                state, terminal, reward = self.environment.execute(actions=action)
                # agent observes "response" of environment to its action
                agent.observe(terminal=terminal, reward=reward)
                # collect actions, states and rewards for later analyses
                actions.append(action)
                states.append(state)
                rewards.append(reward)
            # compute and collect average statistics per episode for later analyses
            avg_reward = np.mean(rewards)
            avg_broken = np.mean(broken_cutters)
            avg_changed = np.mean(np.sum(np.where(np.vstack(actions) > .5, 1, 0), 1))
            df_temp = pd.DataFrame({'episode': [ep], 'min_rewards': [min(rewards)],
                                    'max_rewards': [max(rewards)],
                                    'avg_rewards': [avg_reward],
                                    'min_brokens': [min(broken_cutters)],
                                    'avg_brokens': [avg_broken],
                                    'max_brokes': [max(broken_cutters)],
                                    'avg_changes_per_interv': [avg_changed]})
            df = pd.concat([df, df_temp])
            summed_actions.append(np.sum(actions, axis=0))
            # show intermediate results on checkpoints
            if ep % self.CHECKPOINT == 0 and ep != 0:
                # Report intermediate objective value.
                interm_rew = df['avg_rewards'].iloc[-self.CHECKPOINT:].mean()
                interm_broken = df['avg_brokens'].iloc[-self.CHECKPOINT:].mean()
                interm_changed = df['avg_changes_per_interv'].iloc[-self.CHECKPOINT:].mean()
                print(f'episode {ep}: rew.: {round(interm_rew, 3)} '
                      f'broken: {round(interm_broken, 3)} '
                      f'changed: {round(interm_changed, 3)}')

                if self.OPTIMIZATION is True:
                    trial.report(interm_rew, ep)
                else:
                    self.environment.save(self.AGENT, train_start, ep, states,
                                          actions, rewards, df, summed_actions,
                                          agent)

        # return final reward -> only necessary for optimization
        # final reward is computed as the average of the top 200 episodes of
        # one series of episodes
        final_reward = np.mean(np.sort(df['avg_rewards'].values)[-200:])
        return final_reward


if __name__ == "__main__":
    # direct implementation of some functions to get plots for the paper.

    t_I = 25  # time for inspection of cutterhead [min]
    t_C_max = 75  # maximum time to change one cutter [min]
    n_c_tot = 28  # total number of cutters
    K = 1.25  # factor controlling change time of cutters

    m = maintenance()

    ##########################################################################
    # plot of cutters to change vs. maintenance time

    cutters_to_change = np.arange(0, n_c_tot*2)
    maint_times = [m.maintenance_cost(t_I, t_C_max, n_c, K)[0] for n_c in cutters_to_change]
    maint_times_uncapped = [m.maintenance_cost(t_I, t_C_max, n_c, K, cap=False)[0] for n_c in cutters_to_change]
    threshold = m.maintenance_cost(t_I, t_C_max, t_C_max, K)[1]

    fig, ax = plt.subplots(figsize=(3.465, 3))
    ax.plot(cutters_to_change, maint_times_uncapped, color='grey')
    ax.plot(cutters_to_change, maint_times, color='black', lw=4)
    ax.axvline(threshold, color='black', alpha=0.5, ls='--')
    ax.grid(alpha=0.5)
    ax.set_xlabel('cutters to change')
    ax.set_ylabel('total maintenance time [min]')
    plt.tight_layout()
    plt.savefig(Path('graphics/cutter_changing_function.svg'))

    ##########################################################################
    # 3D plot of reward functions

    # compute max possible time for maintenance
    t_maint_max = m.maintenance_cost(t_I, t_C_max, n_c_tot, K)[0]

    t_maints = []  # maintenance times
    r1_s = []  # rewards mode 1
    r2_s = []  # rewards mode 2
    r3_s = []  # rewards mode 3
    xs = []  # number of "good cutters"
    ys = []  # number of cutters that should be changed

    for good_cutters in np.arange(0, n_c_tot+1):
        for cutters_to_change in np.arange(0, n_c_tot+1):
            t_maint = m. maintenance_cost(t_I, t_C_max,
                                          cutters_to_change, K)[0]
            r1 = m.reward(good_cutters, n_c_tot, t_maint, t_maint_max,
                          cutters_to_change, mode=1, cap_zero=True,
                          scale=False)
            r2 = m.reward(good_cutters, n_c_tot, t_maint, t_maint_max,
                          cutters_to_change, mode=2, cap_zero=True,
                          scale=False)
            r3 = m.reward(good_cutters, n_c_tot, t_maint, t_maint_max,
                          cutters_to_change, mode=3, cap_zero=True,
                          scale=False)
            xs.append(good_cutters)
            ys.append(cutters_to_change)
            t_maints.append(t_maint)
            r1_s.append(r1)
            r2_s.append(r2)
            r3_s.append(r3)

    XS = np.reshape(xs, (n_c_tot+1, n_c_tot+1))
    YS = np.reshape(ys, (n_c_tot+1, n_c_tot+1))
    ZS1 = np.reshape(r1_s, (n_c_tot+1, n_c_tot+1))
    ZS2 = np.reshape(r2_s, (n_c_tot+1, n_c_tot+1))
    ZS3 = np.reshape(r3_s, (n_c_tot+1, n_c_tot+1))

    fig = plt.figure(figsize=(3.465, 3.465))

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(XS, YS, ZS1, color='grey')

    ax.view_init(elev=20., azim=150)
    ax.set_xlabel('good cutters')
    ax.set_ylabel('cutters to change')
    ax.set_zlabel('reward')
    ax.set_zlim(top=1)

    plt.savefig(Path(f'graphics/reward_function.svg'))
