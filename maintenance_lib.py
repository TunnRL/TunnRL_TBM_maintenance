# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 12:57:51 2021

@author: Schorsch
"""

import matplotlib.pyplot as plt
import numpy as np
# from tensorforce.environments import Environment


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


class CustomEnv(Environment, maintenance):
    # https://tensorforce.readthedocs.io/en/latest/basics/getting-started.html

    def __init__(self, n_c_tot, LIFE, t_I, t_C_max, K, t_maint_max,
                  MAX_STROKES, STROKE_LENGTH, cutter_pathlenghts, INTERVALL,
                  REWARD_MODE):
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
        self.INTERVALL = INTERVALL
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
                              mode=self.REWARD_MODE, cap_zero=True, scale=True)

        # update cutter life based on how much wear occurs
        penetration = np.random.normal(loc=0.008, scale=0.003)  # [m/rot]
        penetration = np.where(penetration < 0.0005, 0.0005, penetration)
        rot_per_stroke = self.STROKE_LENGTH / penetration
        rot_per_interv = rot_per_stroke * self.INTERVALL

        self.state = self.state - rot_per_interv * (self.cutter_pathlenghts / self.LIFE)
        self.state = np.where(self.state <= 0, 0, self.state)

        self.stroke_intervall += 1
        if self.stroke_intervall >= self.MAX_STROKES:
            terminal = True
        else:
            terminal = False

        return self.state, terminal, reward

    def reset(self):
        # self.state = np.full((self.n_c_tot), 1)  # all cutters good at start
        self.stroke_intervall = 0  # reset maintenance intervall
        self.state = np.full((self.n_c_tot), 0)  # all cutters broken at start
        return self.state  # reward, done, info can't be included


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
    rewards = []  # rewards
    xs = []  # number of "good cutters"
    ys = []  # number of cutters that should be changed

    for good_cutters in np.arange(0, n_c_tot):
        for cutters_to_change in np.arange(0, n_c_tot):
            t_maint = m. maintenance_cost(t_I, t_C_max,
                                          cutters_to_change, K)[0]
            r = m.reward(good_cutters, n_c_tot, t_maint, t_maint_max)
            xs.append(good_cutters)
            ys.append(cutters_to_change)
            t_maints.append(t_maint)
            rewards.append(r)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(xs, ys, rewards)
    ax.set_xlabel('good cutters')
    ax.set_ylabel('cutters to change')
    ax.set_zlabel('reward')
