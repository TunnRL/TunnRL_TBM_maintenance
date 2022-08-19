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

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from numpy.typing import NDArray


class Maintenance:
    '''class that contains functions that describe the maintenance effort of
    changing cutters on a TBM's cutterhead. Based on this the reward is
    computed'''

    def __init__(self, n_c_tot: int, broken_cutters_thresh: float):

        """Setup

        Args:
            n_c_tot (int): total number of cutters
            broken_cutters_thresh (float): minimum required percentage of
                functional cutters
        """
        self.n_c_tot = n_c_tot
        self.broken_cutters_thresh = broken_cutters_thresh
        self.t_i = 1  # cost of entering the cutterhead for maintenance
        self.alpha = 0.2  # weighting factor for replacing cutters
        self.beta = 0.3  # weighting factor for moving cutters
        self.gamma = 0.25  # weighting factor for cutter distance
        self.delta = 0.25  # weighting factor for entering cutterhead

        if self.alpha + self.beta + self.gamma + self.delta != 1:
            raise ValueError('reward weighting factors do not sum up to 1!')

    def reward(self, replaced_cutters: list, moved_cutters: list,
               good_cutters: int) -> float:
        """Reward function. Drives the agent learning process.

        Handle the replacing and moving of cutters.
        """
        # if good_cutters < self.n_c_tot * 0.5:
        #     r = 0
        # elif good_cutters == self.n_c_tot and replaced_cutters + moved_cutters == 0:
        #     r = 1
        # else:
        #     ratio1 = good_cutters / self.n_c_tot
        #     ratio2 = (moved_cutters / self.n_c_tot) * 1.05
        #     ratio3 = (replaced_cutters / self.n_c_tot) * 0.9
        #     r = ratio1 - ratio2 - ratio3

        # r = max(0, r)

        # compute distance between acted on cutters -> encourage series change
        acted_on_cutters = sorted(replaced_cutters+moved_cutters)
        dist_cutters = np.sum(np.diff(acted_on_cutters))

        if good_cutters < self.n_c_tot * self.broken_cutters_thresh:
            # if more than threshhold number of cutters are broken
            r = -1
        elif len(acted_on_cutters) == 0:
            # if no cutters are acted on
            r = good_cutters / self.n_c_tot
        else:
            # weighted representation of cutters to penalize changing of outer
            # cutters more than inner cutters
            weighted_cutters = np.linspace(1, 2, num=self.n_c_tot)

            ratio1 = good_cutters / self.n_c_tot
            ratio2 = (np.sum(np.take(weighted_cutters, replaced_cutters)) / np.sum(weighted_cutters)) * self.alpha
            ratio3 = (np.sum(np.take(weighted_cutters, moved_cutters)) / np.sum(weighted_cutters)) * self.beta
            ratio4 = ((dist_cutters+1) / self.n_c_tot) * self.gamma
            change_penalty = self.t_i * self.delta
            r = ratio1 - ratio2 - ratio3 - ratio4 - change_penalty

        return r


class CustomEnv(gym.Env):
    '''Implementation of the custom environment that simulates the cutter wear
    and provides the agent with a state and reward signal.
    '''
    def __init__(self,
                 n_c_tot: int,
                 LIFE: int,
                 MAX_STROKES: int,
                 STROKE_LENGTH: float,
                 cutter_pathlenghts: float,
                 CUTTERHEAD_RADIUS: float,
                 broken_cutters_thresh: float) -> None:
        """Initializing custom environment for a TBM cutter operation.

        Args:
            n_c_tot (int): total number of cutters
            LIFE (int): theoretical durability of one cutter [m]
            MAX_STROKES (int): numer of strokes to simulate
            STROKE_LENGTH (float): length of one stroke [m]
            cutter_pathlenghts (float): rolling lengths [m]
            CUTTERHEAD_RADIUS (float): radius of cutterhead
            broken_cutters_thresh (float): minimum required percentage of
                functional cutters
        """
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(n_c_tot * n_c_tot,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_c_tot,))

        self.n_c_tot = n_c_tot
        self.LIFE = LIFE
        self.MAX_STROKES = MAX_STROKES
        self.STROKE_LENGTH = STROKE_LENGTH
        self.cutter_pathlenghts = cutter_pathlenghts
        self.R = CUTTERHEAD_RADIUS

        # instantiated state variables
        self.m = Maintenance(n_c_tot, broken_cutters_thresh)

        # state variables assigned in class methods:
        self.state: NDArray
        self.epoch: int
        self.replaced_cutters: list
        self.moved_cutters: list
        self.penetration: NDArray

    def step(self, actions: NDArray) -> tuple[NDArray, float, bool, dict]:
        '''Main function that moves the environment one step further.
            - Updates the state and reward.
            - Checks if the terminal state is reached.
        '''
        # replace cutters based on action of agent
        self.state = self._implement_action(actions, self.state)

        # compute reward
        good_cutters = len(np.where(self.state > 0)[0])
        reward = self.m.reward(self.replaced_cutters, self.moved_cutters,
                               good_cutters)

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

        return self.state, reward, terminal, {}

    def _implement_action(self, action: NDArray, state_before: NDArray) -> NDArray:
        '''Function that interprets the "raw action" and modifies the state.'''
        state_new = state_before
        self.replaced_cutters = []
        self.moved_cutters = []

        for i in range(self.n_c_tot):
            cutter = action[i * self.n_c_tot: i * self.n_c_tot + self.n_c_tot]
            if np.max(cutter) < 0.9:
                # cutter is not acted on
                pass
            elif np.argmax(cutter) == i:
                # cutter is replaced
                state_new[i] = 1
                self.replaced_cutters.append(i)
            else:
                # cutter is moved from original position to somewhere else
                state_new[np.argmax(cutter)] = state_new[i]
                state_new[i] = 1
                self.moved_cutters.append(i)

        return state_new

    def rand_walk_with_bounds(self, n_dp: int) -> NDArray:
        '''function generates a random walk within the limits 0 and 1'''
        bounds = .05

        x = [np.random.uniform(low=0, high=1, size=1)]

        for move in np.random.uniform(low=-bounds, high=bounds, size=n_dp):
            x_temp = x[-1] + move
            if x_temp <= 1 and x_temp >= 0:
                x.append(x_temp)
            else:
                x.append(x[-1] - move)

        return np.array(x[1:])

    def generate(self, Jv_low: int = 0, Jv_high: int = 22, 
                 UCS_center: int = 80, UCS_range: int = 30) -> tuple:
        '''Function generates TBM recordings for one episode. Equations and
        models based on Delisio & Zhao (2014) - "A new model for TBM
        performance prediction in blocky rock conditions",
        http://dx.doi.org/10.1016/j.tust.2014.06.004'''

        Jv_s = self.rand_walk_with_bounds(self.MAX_STROKES) * (Jv_high - Jv_low) + Jv_low  # [joints / m3]
        UCS_s = UCS_center + self.rand_walk_with_bounds(self.MAX_STROKES) * UCS_range  # [MPa]

        # eq 9, Delisio & Zhao (2014) - [kN/m/mm/rot]
        FPIblocky_s = np.squeeze(np.exp(6) * Jv_s**-0.82 * UCS_s**0.17)

        brokens = np.zeros(shape=(self.MAX_STROKES, self.n_c_tot))

        for stroke in range(self.MAX_STROKES):
            # based on FPI-blocky, cutters have different likelyhoods to break
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

    def reset(self) -> NDArray:
        '''reset an environment to its initial state'''
        self.state = np.full((self.n_c_tot), 1)  # start with new cutters
        self.epoch = 0  # reset epoch counter
        # generate new TBM data for episode
        self.Jv_s, self.UCS_s, self.FPIblocky_s, self.brokens, self.TF_s, self.penetration = self.generate()
        return self.state

    def render(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":

    # visualization of all possible reward states
    n_c_tot = 28  # total number of cutters
    broken_cutters_thresh = 0.5

    m = Maintenance(n_c_tot, broken_cutters_thresh)

    cutters = np.arange(n_c_tot)

    n_repls = []
    n_moves = []
    n_good_cutters = []
    r_s = []

    for _ in range(10_000):
        n_repl = np.random.randint(0, n_c_tot)
        replaced_cutters = np.sort(np.random.choice(cutters, n_repl, replace=False))
        n_move = np.random.randint(0, n_c_tot - n_repl)
        moved_cutters = np.sort(np.random.choice(np.delete(cutters, replaced_cutters),
                                                 n_move, replace=False))

        good_cutters = np.random.randint(n_repl+n_move, n_c_tot)
        # print(n_repl, n_move, good_cutters)

        r = m.reward(list(replaced_cutters), list(moved_cutters), good_cutters)
        n_repls.append(n_repl)
        n_moves.append(n_move)
        n_good_cutters.append(good_cutters)
        r_s.append(r)
        # print(r)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    ax1.scatter(n_repls, r_s)
    ax2.scatter(n_moves, r_s)
    ax3.scatter(n_good_cutters, r_s)
    plt.tight_layout()

    # rewards = []
    # for i in range(len(combined)):
    #     rewards.append(m.reward(replaced_cutters=replaced_cutters[i],
    #                             moved_cutters=moved_cutters[i],
    #                             good_cutters=good_cutters[i],
    #                             dist_cutters=dist_cutters[i]))
    # rewards = np.array(rewards)

    # low_r_id = np.argmin(rewards)
    # high_r_id = np.argmax(rewards)
    # len_low_r = len(np.where(rewards == rewards.min())[0])
    # len_high_r = len(np.where(rewards == rewards.max())[0])

    # print(f'there are {len_low_r} combinations with {rewards.min()} reward')
    # print(f'lowest reward of {rewards.min()} with combination:')
    # print(f'\t{replaced_cutters[low_r_id]} replaced cutters')
    # print(f'\t{moved_cutters[low_r_id]} moved cutters')
    # print(f'\t{good_cutters[low_r_id]} good cutters\n')

    # print(f'there are {len_high_r} combinations with {rewards.max()} reward')
    # print(f'highest reward of {rewards.max()} with combination:')
    # print(f'\t{replaced_cutters[high_r_id]} replaced cutters')
    # print(f'\t{moved_cutters[high_r_id]} moved cutters')
    # print(f'\t{good_cutters[high_r_id]} good cutters')

    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(projection='3d')
    # scatter_plot= ax.scatter(replaced_cutters, moved_cutters, good_cutters,
    #                          c=rewards, edgecolor='black', s=40)
    # ax.set_xlabel('n replaced cutters')
    # ax.set_ylabel('n moved cutters')
    # ax.set_zlabel('n good cutters')

    # plt.colorbar(scatter_plot, label='reward')
    # plt.tight_layout()
