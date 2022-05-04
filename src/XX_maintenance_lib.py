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

import matplotlib.cm as mplcm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tensorforce.environments import Environment


class plotter:
    '''class that contains function to visualzie the progress of the
    training and / or individual samples'''

    def __init__(self):
        pass

    def sample_ep_plot(self, states, actions, ep, savename):
        '''plot of different recordings of one exemplary episode'''

        cmap = mplcm.get_cmap('viridis')

        states_arr = np.vstack(states[:-1])
        actions_arr = np.vstack(actions)
        actions_arr = np.where(actions_arr > .5, 1, 0)  # binarize

        fig = plt.figure(tight_layout=True, figsize=(20, 8))
        gs = gridspec.GridSpec(nrows=3, ncols=2, width_ratios=[5, 1])

        # cutter life line plot
        ax = fig.add_subplot(gs[0, 0])
        for cutter in range(states_arr.shape[1]):
            rgba = cmap(cutter / states_arr.shape[1])
            ax.plot(np.arange(states_arr.shape[0]), states_arr[:, cutter],
                    color=rgba, label=cutter)
        h, l = ax.get_legend_handles_labels()
        ax.set_xlim(left=-1, right=actions_arr.shape[0]+1)
        ax.set_ylim(top=1.05, bottom=-0.05)
        ax.set_title(f'epoch {ep}')
        ax.set_ylabel('cutter life')
        ax.set_xticklabels([])
        ax.grid(alpha=0.5)

        # dedicated subplot for a legend to first axis
        lax = fig.add_subplot(gs[0, 1])
        lax.legend(h, l, borderaxespad=0, ncol=3, loc='upper left')
        lax.axis('off')

        # scatter plot that shows which actions were chosen
        ax = fig.add_subplot(gs[1, 0])
        for intervall in range(actions_arr.shape[0]):
            ax.scatter(np.full((actions_arr.shape[1]), intervall),
                       np.arange(actions_arr.shape[1]),
                       c=actions_arr[intervall, :], cmap='Greys', s=10,
                       edgecolor='black', lw=.1, vmin=0, vmax=1)

        ax.set_xlim(left=-1, right=actions_arr.shape[0]+1)
        ax.set_ylim(bottom=-1, top=actions_arr.shape[1]+1)
        ax.set_ylabel('ations on cutter positions')
        ax.set_xticklabels([])
        ax.grid(alpha=0.5)

        # barchart of frequently changed cutter positions
        ax = fig.add_subplot(gs[1, 1])
        ax.barh(np.arange(actions_arr.shape[1]), np.sum(actions_arr, axis=0),
                color='grey', edgecolor='black')
        ax.set_ylim(bottom=-1, top=actions_arr.shape[1]+1)
        ax.grid()
        ax.set_yticklabels([])
        ax.set_ylabel('n cutter changes per episode')
        ax.yaxis.set_label_position('right')

        # bar plot that shows how many cutters were changed
        ax = fig.add_subplot(gs[2, 0])
        ax.bar(x=np.arange(actions_arr.shape[0]),
               height=np.sum(actions_arr, axis=1), color='black')
        ax.set_xlim(left=-1, right=actions_arr.shape[0]+1)
        ax.set_ylabel('n cutter changes per stroke')
        ax.set_xlabel('strokes')
        ax.grid(alpha=0.5)

        plt.savefig(fr'checkpoints\{savename}_sample.png', dpi=600)
        plt.savefig(fr'checkpoints\{savename}_sample.svg')
        plt.close()


    def trainingprogress_plot(self, df, summed_actions, name, savepath):
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, ncols=1,
                                                 figsize=(12, 9))

        ax0.imshow(np.vstack(summed_actions).T, aspect='auto')
        ax0.set_ylabel('actions on\ncutter positions')
        ax0.set_title(name)
        ax0.set_xticklabels([])

        ax1.plot(df['episode'], df['avg_changes_per_interv'], color='black')
        ax1.grid(alpha=0.5)
        ax1.set_xlim(left=0, right=len(df))
        ax1.set_ylabel('avg. n cutter changes / stroke')
        ax1.set_xticklabels([])

        ax2.plot(df['episode'], df['avg_brokens'], color='black')
        ax2.grid(alpha=0.5)
        ax2.set_xlim(left=0, right=len(df))
        ax2.set_ylabel('avg. n broken cutters / stroke')
        ax2.set_xticklabels([])

        ax3.plot(df['episode'], df['avg_rewards'], color='black', alpha=0.7,
                 label='agent')

        ax3.set_xlim(left=0, right=len(df))
        ax3.set_ylim(top=1, bottom=0)
        ax3.grid(alpha=0.5)
        ax3.set_ylabel('avg. reward / stroke')
        ax3.set_xlabel('episodes')

        plt.tight_layout()
        plt.savefig(savepath, dpi=600)
        plt.close()


class SimpleAgent:

    def act(self, states, internals=None, independent=None):
        return np.where(states <= 0, 1, 0)

    def observe(self, terminal, reward):
        pass

    def save(self, directory, filename, format):
        pass


class maintenance:

    def __init__(self):
        pass

    def maintenance_core_function(self, t_inspection, t_change_max,
                                  n_cutters, K):
        time_per_cutter = t_change_max - n_cutters * K  # [min]
        t_maint = t_inspection + n_cutters * time_per_cutter  # [min]
        return t_maint

    def maintenance_cost(self, t_inspection, t_change_max, n_c_to_change, K):
        # change time per cutter decreases with increasing n_c_to_change
        # acc to Farrokh 2021 -> 0.95 estimated
        t_maint = self. maintenance_core_function(t_inspection, t_change_max,
                                                  n_c_to_change, K)

        # compute first derivative of function to get maximum number of cutters
        # beyond which change time flattens out
        max_c = t_change_max / (2*K)

        if n_c_to_change > max_c:
            t_maint = self.maintenance_core_function(t_inspection,
                                                     t_change_max, max_c, K)

        return t_maint, max_c  # total time for maintenance [min]

    def reward(self, good_cutters, n_c_tot, t_maint, t_maint_max,
               n_c_to_change, mode, cap_zero=False, scale=False):
        if mode == 1:  # complex reward featuring maintenance time function
            r = good_cutters / n_c_tot - t_maint / t_maint_max
        elif mode == 2:  # normal reward
            r = (good_cutters / n_c_tot) - (n_c_to_change / n_c_tot)
        elif mode == 3:  # easy reward only going for max good cutters
            r = good_cutters / n_c_tot

        if cap_zero is True:
            r = 0 if r < 0 else r
        if scale is True:
            r = r * 2 - 1

        return r


class CustomEnv(Environment, maintenance, plotter):
    # https://tensorforce.readthedocs.io/en/latest/basics/getting-started.html

    def __init__(self, n_c_tot, LIFE, t_I, t_C_max, K, t_maint_max,
                 MAX_STROKES, STROKE_LENGTH, cutter_pathlenghts, REWARD_MODE,
                 ):
        super().__init__()

        self.n_c_tot = n_c_tot  # total number of cutters
        self.LIFE = LIFE  # theoretical durability of one cutter [m]
        self.t_I = t_I  # time for inspection of cutterhead [min]
        self.t_C_max = t_C_max  # maximum time to change one cutter [min]
        self.K = K  # factor controlling change time of cutters
        self.t_maint_max = t_maint_max  # max. time for maintenance
        self.MAX_STROKES = MAX_STROKES  # number of strokes to simulate
        self.STROKE_LENGTH = STROKE_LENGTH  # length of one stroke [m]
        self.cutter_pathlenghts = cutter_pathlenghts
        self.REWARD_MODE = REWARD_MODE

    def states(self):
        return dict(type='float', shape=(self.n_c_tot,), min_value=0,
                    max_value=1)

    # def actions(self):
    #     #return dict(type='int', shape=(self.n_c_tot,), num_values=2)
    #     return dict(type='float', shape=(self.n_c_tot,), min_value=0,
    #                 max_value=1)

    def execute(self, actions):
        # replace cutters based on action
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
        penetration = np.random.normal(loc=0.008, scale=0.003)  # [m/rot]
        penetration = np.where(penetration < 0.0005, 0.0005, penetration)
        rot_per_stroke = self.STROKE_LENGTH / penetration

        self.state = self.state - rot_per_stroke * (self.cutter_pathlenghts / self.LIFE)
        self.state = np.where(self.state <= 0, 0, self.state)

        self.stroke_intervall += 1
        if self.stroke_intervall >= self.MAX_STROKES:
            terminal = True
        else:
            terminal = False

        return self.state, terminal, reward

    def reset(self):
        self.state = np.full((self.n_c_tot), 1)  # all cutters good at start
        self.stroke_intervall = 0  # reset maintenance intervall
        # self.state = np.full((self.n_c_tot), 0)  # all cutters broken at start
        return self.state  # reward, done, info can't be included

    def save(self, AGENT, train_start, ep, states, actions, df, summed_actions,
             agent):
        name = f'{AGENT}_{train_start}_{ep}'

        self.sample_ep_plot(states, actions, ep, savename=name)

        self.trainingprogress_plot(df, summed_actions, name,
                                   savepath=fr'checkpoints\{name}_progress.png')

        agent.save(directory='checkpoints', filename=name, format='hdf5')
        df.to_csv(fr'checkpoints\{name}.csv', index=False)


if __name__ == "__main__":

    t_I = 25  # time for inspection of cutterhead [min]
    t_C_max = 75  # maximum time to change one cutter [min]
    n_c_tot = 28  # total number of cutters
    K = 1.25  # factor controlling change time of cutters

    m = maintenance()

    ##########################################################################
    # plot of cutters to change vs. maintenance time

    cutters_to_change = np.arange(0, n_c_tot)
    maint_times = [m.maintenance_cost(t_I, t_C_max, n_c, K)[0] for n_c in cutters_to_change]
    threshold = m.maintenance_cost(t_I, t_C_max, t_C_max, K)[1]

    fig, ax = plt.subplots()
    ax.scatter(cutters_to_change, maint_times)
    ax.axvline(threshold)
    ax.set_xlabel('cutters to change')
    ax.set_ylabel('total maintenance time [min]')

    ##########################################################################
    # 3D plot of reward function

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
    # print(min(rewards), max(rewards))

    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(XS, YS, ZS3, shade=True)

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(XS, YS, ZS2)

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot_surface(XS, YS, ZS1)

    ax1.view_init(elev=20., azim=150)
    ax1.set_xlabel('good cutters')
    ax1.set_ylabel('cutters to change')
    ax1.set_zlabel('reward')
    ax1.set_title('simple reward')

    ax2.view_init(elev=20., azim=150)
    ax2.set_xlabel('good cutters')
    ax2.set_ylabel('cutters to change')
    ax2.set_zlabel('reward')
    ax2.set_title('normal reward')

    ax3.view_init(elev=20., azim=150)
    ax3.set_xlabel('good cutters')
    ax3.set_ylabel('cutters to change')
    ax3.set_zlabel('reward')
    ax3.set_title('complex reward')
    ax3.set_zlim(top=1)

    # plt.savefig(r'graphics\reward_functions.pdf')
