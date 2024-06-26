# -*- coding: utf-8 -*-
"""
Code for the paper:

Towards smart TBM cutter changing with reinforcement learning (working title)
Georg H. Erharter, Tom F. Hansen, Thomas Marcher, Amund Bruland
JOURNAL NAME
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Custom library that contains different classes for the reinforcement learning
based cutter changing environment

code contributors: Georg H. Erharter, Tom F. Hansen
"""

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from numpy.typing import NDArray


class Maintenance:
    """class that contains functions that describe the maintenance effort of
    changing cutters on a TBM's cutterhead. Based on this the reward is
    computed"""

    def __init__(self, n_c_tot: int, broken_cutters_thresh: float,
                 check_bearing_failure: bool, bearing_failure_penalty: bool, alpha: float, beta: float,
                 gamma: float, delta: float) -> None:
        """Setup

        Args:
            n_c_tot (int): total number of cutters
            broken_cutters_thresh (float): minimum required percentage of
                functional cutters
            check_bearing_failure (bool): if the reward function should check
                for / consider cutter bearing failures or not
            alpha (float): weighting factor for replacing cutters
            beta (float): weighting factor for moving cutters
            gamma (float): weighting factor for cutter distance
            delta (float): weighting factor for entering cutterhead
        """
        self.n_c_tot = n_c_tot
        self.broken_cutters_thresh = broken_cutters_thresh
        self.check_bearing_failure = check_bearing_failure
        self.bearing_failure_penalty = bearing_failure_penalty
        self.t_i = 1  # cost of entering the cutterhead for maintenance
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        if self.alpha + self.beta + self.gamma + self.delta != 1:
            raise ValueError('reward weighting factors do not sum up to 1!')

    def reward(self, replaced_cutters: list, moved_cutters: list,
               good_cutters: int, damaged_bearing: bool) -> float:
        """Reward function. Drives the agent learning process.

        Handle the replacing and moving of cutters.

        Args:
            replaced_cutters (list): list of replaced cutters
            moved_cutters (list): list of moved cutters
            good_cutters (int): number of good cutters, ie. cutters with life
                greater than 0
            damaged_bearing (bool): if at least one cutter bearing fails due to
                blockyness damage to the cutter and no subsequent repair. Only
                effective if check_bearing_failure == True

        Returns:
            float: computed reward for a certain state.
        """
        # compute distance between acted on cutters -> encourage series change
        acted_on_cutters = sorted(replaced_cutters + moved_cutters)
        dist_cutters = np.sum(np.diff(acted_on_cutters))

        if good_cutters < self.n_c_tot * self.broken_cutters_thresh:
            # if more than threshhold number of cutters are broken
            r = -1
        elif self.check_bearing_failure is True and damaged_bearing is True:
            # if check for bearing failures is set and bearing failure occurs
            r = self.bearing_failure_penalty
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
            ratio4 = ((dist_cutters + 1) / self.n_c_tot) * self.gamma
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
                 broken_cutters_thresh: float,
                 alpha: float,
                 beta: float,
                 gamma: float,
                 delta: float,
                 check_bearing_failure: bool,
                 bearing_failur_penalty: bool) -> None:
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
            alpha (float): weighting factor for replacing cutters
            beta (float): weighting factor for moving cutters
            gamma (float): weighting factor for cutter distance
            delta (float): weighting factor for entering cutterhead
            check_bearing_failure (bool): if the reward function should check
                for / consider cutter bearing failures or not
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
        self.check_bearing_failure = check_bearing_failure
        self.bearing_failure_penalty = bearing_failur_penalty

        # instantiated state variables
        self.m = Maintenance(n_c_tot, broken_cutters_thresh, check_bearing_failure,
                             bearing_failur_penalty, alpha, beta, gamma, delta)

        # state variables assigned in class methods:
        self.state: NDArray
        self.epoch: int
        self.replaced_cutters: list
        self.moved_cutters: list
        self.good_cutters: NDArray
        self.penetration: NDArray

    def step(self, actions: NDArray) -> tuple[NDArray, float, bool, dict]:
        '''Main function that moves the environment one step further.
            - Updates the state and reward.
            - Checks if the terminal state is reached.
        '''
        # replace cutters based on action of agent
        self.state = self._implement_action(actions, self.state)

        # compute inputs to reward and reward
        self.broken_cutters = np.where(self.state == 0)[0]  # for statistics
        self.good_cutters = np.where(self.state > 0)[0]
        n_good_cutters = len(self.good_cutters)
        self.moved_cutters = sorted(self.inwards_moved_cutters + self.wrong_moved_cutters)
        # check if cutters broke due to blockyness in prev. stroke and have not
        # yet been replaced and create a bearing failure subsequently
        # if state at indizes of broken cutters due to blockyness of prev. stroke == 0
        # potential issue: damaged bearing only gets punished once
        prev_blocky_failure_ids = np.where(self.brokens[self.epoch, :] == 1)[0]
        if len(prev_blocky_failure_ids) > 0:
            if min(self.state[prev_blocky_failure_ids]) == 0:
                damaged_bearing = True
            else:
                damaged_bearing = False
        else:
            damaged_bearing = False
        reward = self.m.reward(self.replaced_cutters, self.moved_cutters,
                               n_good_cutters, damaged_bearing)

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
        self.inwards_moved_cutters = []
        self.wrong_moved_cutters = []
        # iterate through cutters starting from outside
        for i in np.arange(self.n_c_tot)[::-1]:
            cutter = action[i * self.n_c_tot: i * self.n_c_tot + self.n_c_tot]
            if np.max(cutter) < 0.9:  # TODO: explain this value
                # cutter is not acted on
                pass
            elif np.argmax(cutter) == i:
                # cutter is replaced
                state_new[i] = 1
                self.replaced_cutters.append(i)
            else:
                # cutter is moved from original position towards center
                if np.argmax(cutter) < i:
                    state_new[np.argmax(cutter)] = state_new[i]
                    state_new[i] = 1
                    self.inwards_moved_cutters.append(i)
                else:
                    self.wrong_moved_cutters.append(i)

        return state_new

    def _rand_walk_with_bounds(self, n_dp: int) -> NDArray:
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
                 UCS_center: int = 80, UCS_range: int = 30
                 ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        '''Function generates TBM recordings for one episode. Equations and
        models based on Delisio & Zhao (2014) - "A new model for TBM
        performance prediction in blocky rock conditions",
        http://dx.doi.org/10.1016/j.tust.2014.06.004'''

        Jv_s = self._rand_walk_with_bounds(self.MAX_STROKES) * (Jv_high - Jv_low) + Jv_low  # [joints / m3]
        UCS_s = UCS_center + self._rand_walk_with_bounds(self.MAX_STROKES) * UCS_range  # [MPa]

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
        TF_s = np.squeeze((-523 * np.log(Jv_s) + 2312) * (self.R * 2))  # [kN]
        TF_s = np.where(TF_s > 20_000, 20_000, TF_s)

        # eq 7, Delisio & Zhao (2014)
        # TF_s already considers shield friction
        penetration = (TF_s / (self.R * 2)) / FPIblocky_s  # [mm/rot]

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
    # create plots describing the environment for the paper
    from XX_plotting import Plotter

    # TBM EXCAVATION PARAMETERS
    ######################
    CUTTERHEAD_RADIUS = 4  # cutterhead radius [m]
    TRACK_SPACING = 0.1  # cutter track spacing [m]
    LIFE = 400000  # theoretical durability of one cutter [m]
    STROKE_LENGTH = 1.8  # length of one stroke [m]
    MAX_STROKES = 1000  # number of strokes per episode

    # REWARD FUNCTION PARAMETERS
    ######################
    BROKEN_CUTTERS_THRESH = 0.85  # minimum required % of functional cutters
    ALPHA = 0.1  # weighting factor for replacing cutters
    BETA = 0.65  # weighting factor for moving cutters
    GAMMA = 0.1  # weighting factor for cutter distance
    DELTA = 0.15  # weighting factor for entering cutterhead
    CHECK_BEARING_FAILURE = True  # if True should check cutter bearing failures
    BEARING_FAILURE_PENALTY = 0

    np.random.seed(14)

    n_c_tot = (int(round((CUTTERHEAD_RADIUS - TRACK_SPACING / 2) / TRACK_SPACING, 0)) + 1)

    cutter_positions = (
        np.cumsum(np.full((n_c_tot), TRACK_SPACING)) - TRACK_SPACING / 2)

    cutter_pathlenghts = cutter_positions * 2 * np.pi  # [m]

    env = CustomEnv(n_c_tot,
                    LIFE,
                    MAX_STROKES,
                    STROKE_LENGTH,
                    cutter_pathlenghts,
                    CUTTERHEAD_RADIUS,
                    BROKEN_CUTTERS_THRESH,
                    ALPHA, BETA, GAMMA, DELTA,
                    CHECK_BEARING_FAILURE,
                    BEARING_FAILURE_PENALTY)
    env.reset()
    plotter = Plotter()
    plotter.environment_parameter_plot(0, env, show=True,
                                       savepath='results/paper_example_episode.png')

    # visualization of possible reward states
    env.m.reward(replaced_cutters=[], moved_cutters=[], good_cutters=35,
                 damaged_bearing=True)

    x = []
    r1_s = []
    r2_s = []
    r3_s = []
    r4_s = []
    r5_s = []

    for n_good_cutters in range(int(n_c_tot*BROKEN_CUTTERS_THRESH)-3,
                                n_c_tot+1):
        x.append(n_good_cutters)
        r1_s.append(env.m.reward(replaced_cutters=[], moved_cutters=[],
                                 good_cutters=n_good_cutters,
                                 damaged_bearing=False))
        r2_s.append(env.m.reward(replaced_cutters=[], moved_cutters=[],
                                 good_cutters=n_good_cutters,
                                 damaged_bearing=True))
        r3_s.append(env.m.reward(replaced_cutters=[0, 1, 2, 3, 4],
                                 moved_cutters=[],
                                 good_cutters=n_good_cutters,
                                 damaged_bearing=False))
        r4_s.append(env.m.reward(replaced_cutters=[],
                                 moved_cutters=[0, 1, 2, 3, 4],
                                 good_cutters=n_good_cutters,
                                 damaged_bearing=False))
        r5_s.append(env.m.reward(replaced_cutters=list(range(10)),
                                 moved_cutters=list(range(20, 30)),
                                 good_cutters=n_good_cutters,
                                 damaged_bearing=False))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(x, r1_s)
    ax.text(40, 1, '1')
    ax.plot(x, r2_s)
    ax.text(40, 0.83, '2')
    ax.plot(x, r3_s)
    ax.text(40, 0.67, '3')
    ax.plot(x, r4_s)
    ax.text(40, 0.46, '4')
    ax.plot(x, r5_s)
    ax.text(40, 0.02, '5')
    ax.axvline(n_c_tot*BROKEN_CUTTERS_THRESH, ls='--', color='black')
    ax.grid(alpha=0.5)
    ax.set_xlabel('number of functional cutters')
    ax.set_ylabel('reward per stroke')
    plt.tight_layout()
    plt.savefig('results/paper_reward_examples.svg', dpi=600)

