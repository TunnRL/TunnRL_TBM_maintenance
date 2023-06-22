"""
Towards optimized TBM cutter changing policies with reinforcement learning
G.H. Erharter, T.F. Hansen
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Custom library that contains different classes for the reinforcement learning
based cutter changing environment

Created on Sat Oct 30 12:57:51 2021
code contributors: Georg H. Erharter, Tom F. Hansen
"""

from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray


# ALTERNATIVE REWARD TO ADJUST AND TRY OUT. Made in companion with GPT
class Reward2:
    def __init__(
        self,
        operating_weight: float = 1.0,
        broken_weight: float = 1.0,
        replacement_weight: float = 0.5,
        move_weight: float = 0.2,
        bearing_weight: float = 2.0,
        threshold_weight: float = 10.0,
        broken_threshold: float = 0.15,
    ) -> None:
        self.operating_weight = operating_weight
        self.broken_weight = broken_weight
        self.replacement_weight = replacement_weight
        self.move_weight = move_weight
        self.bearing_weight = bearing_weight
        self.threshold_weight = threshold_weight
        self.broken_threshold = broken_threshold

    def __call__(
        self,
        prev_operating_disks: int,
        curr_operating_disks: int,
        prev_broken_disks: int,
        curr_broken_disks: int,
        num_replacements: int,
        num_moves: int,
        bearing_failures: int,
    ) -> float:
        # Reward for maintaining or increasing the number of operating disks
        operating_reward = (
            max(curr_operating_disks - prev_operating_disks, 0) * self.operating_weight
        )

        # Penalty for increasing the number of broken disks
        broken_penalty = (
            max(curr_broken_disks - prev_broken_disks, 0) * -self.broken_weight
        )

        # Penalty for unnecessary replacements
        replacement_penalty = num_replacements * -self.replacement_weight

        # Reward for moving cutter disks towards the center
        move_reward = num_moves * self.move_weight

        # Penalty for bearing failures
        bearing_penalty = bearing_failures * -self.bearing_weight

        # Penalty for exceeding the threshold of broken cutter disks
        threshold_penalty = 0
        if curr_broken_disks > self.broken_threshold:
            threshold_penalty = -self.threshold_weight

        total_reward = (
            operating_reward
            + broken_penalty
            + replacement_penalty
            + move_reward
            + bearing_penalty
            + threshold_penalty
        )
        return total_reward


@dataclass
class Reward:
    """class that contains functions that describe the maintenance effort of
    changing cutters on a TBM's cutterhead. Based on this the reward is
    computed
    Args:
        n_c_tot (int): total number of cutters
        broken_cutters_thresh (float): minimum required percentage of functional cutters
        CHECK_BEARING_FAILURE (bool): if True should check cutter bearing failures
        T_I (int): cost of entering the cutterhead for maintenance
        ALPHA (float): weighting factor for replacing cutters
        BETA (float): weighting factor for moving cutters
        GAMMA (float): weighting factor for cutter distance
        DELTA (float): weighting factor for entering cutterhead
    """

    n_c_tot: int
    BROKEN_CUTTERS_THRESH: float
    CHECK_BEARING_FAILURE: bool
    BEARING_FAILURE_PENALTY: bool
    T_I: float = 1
    ALPHA: float = 0.2
    BETA: float = 0.3
    GAMMA: float = 0.25
    DELTA: float = 0.25

    assert (
        ALPHA + BETA + GAMMA + DELTA == 1
    ), "reward weighting factors do not sum up to 1!"

    def __call__(
        self,
        replaced_cutters: list,
        moved_cutters: list,
        n_good_cutters: int,
        damaged_bearing: bool,
    ) -> float | int:
        """Setup

        Args:
            replaced_cutters (list): list of replaced cutters
            moved_cutters (list): list of moved cutters
            n_good_cutters (int): number of good cutters, ie. cutters with life
                greater than 0
            damaged_bearing (bool): if at least one cutter bearing fails due to
                blockyness damage to the cutter and no subsequent repair. Only
                effective if check_bearing_failure == True
        """
        acted_on_cutters = sorted(replaced_cutters + moved_cutters)
        dist_cutters = np.sum(np.diff(acted_on_cutters))

        if n_good_cutters < self.n_c_tot * self.BROKEN_CUTTERS_THRESH:
            # if more than threshhold number of cutters are broken
            reward = -1
        elif self.CHECK_BEARING_FAILURE is True and damaged_bearing is True:
            # if check for bearing failures is set and bearing failure occurs
            reward = self.BEARING_FAILURE_PENALTY

        elif len(acted_on_cutters) == 0:
            # if no cutters are acted on
            reward = n_good_cutters / self.n_c_tot
        else:
            # weighted representation of cutters to penalize changing of outer
            # cutters more than inner cutters
            weighted_cutters = np.linspace(1, 2, num=self.n_c_tot)

            ratio1 = n_good_cutters / self.n_c_tot
            ratio2 = (
                np.sum(np.take(weighted_cutters, replaced_cutters))
                / np.sum(weighted_cutters)
            ) * self.ALPHA
            ratio3 = (
                np.sum(np.take(weighted_cutters, moved_cutters))
                / np.sum(weighted_cutters)
            ) * self.BETA
            ratio4 = ((dist_cutters + 1) / self.n_c_tot) * self.GAMMA
            change_penalty = self.T_I * self.DELTA
            reward = ratio1 - ratio2 - ratio3 - ratio4 - change_penalty

        return reward


# TODO: use this instead if this works similar as the original one. Is coded in a better
# way than the original
@dataclass
class Reward3:
    """
    Class that computes the reward based on maintenance effort for changing cutters on a
    TBM's cutterhead.

    Remember: the main goal of the agent is to maximize the sum of future rewards. If
    the agent only choose to do nothing (gives max reward on a step), the cutters will
    be weared down and break after some steps (here strokes of a TBM machine), or a not
    treated bearing failure will give immediate stop.

    Attributes:
        n_c_tot (int): Total number of cutters.
        BROKEN_CUTTERS_THRESH (float): Minimum required percentage of functional
        cutters.
        CHECK_BEARING_FAILURE (bool): Whether to check for cutter bearing failures.
        BEARING_FAILURE_PENALTY (bool): Penalty value for cutter bearing failure.
        T_I (float, optional): Cost of entering the cutterhead for maintenance. Defaults
        to 1.
        ALPHA (float, optional): Weighting factor for cutter replacement. Defaults to
        0.2.
        BETA (float, optional): Weighting factor for cutter movement. Defaults to 0.3.
        GAMMA (float, optional): Weighting factor for cutter distance. Defaults to 0.25.
        DELTA (float, optional): Weighting factor for entering cutterhead. Defaults to
        0.25.

    Raises:
        AssertionError: If the reward weighting factors do not sum up to 1.
    """

    n_c_tot: int
    BROKEN_CUTTERS_THRESH: float
    CHECK_BEARING_FAILURE: bool
    BEARING_FAILURE_PENALTY: bool
    T_I: float = 1
    ALPHA: float = 0.2
    BETA: float = 0.3
    GAMMA: float = 0.25
    DELTA: float = 0.25

    assert (
        ALPHA + BETA + GAMMA + DELTA == 1
    ), "Reward weighting factors do not sum up to 1!"

    def __call__(
        self,
        replaced_cutters: list,
        moved_cutters: list,
        n_good_cutters: int,
        damaged_bearing: bool,
    ) -> float | int:
        """
        Compute the reward based on maintenance effort for changing cutters.

        Args:
            replaced_cutters (list): List of replaced cutters.
            moved_cutters (list): List of moved cutters.
            n_good_cutters (int): Number of good cutters (with life greater than 0).
            damaged_bearing (bool): Whether at least one cutter bearing fails due to
            blockyness damage.

        Returns:
            float or int: Computed reward value.

        Raises:
            ValueError: If invalid input is provided.
        """

        acted_on_cutters = sorted(replaced_cutters + moved_cutters)
        dist_cutters = np.sum(np.diff(acted_on_cutters))

        # If more than the threshold number of cutters are broken we cannot operate
        if n_good_cutters < self.n_c_tot * self.BROKEN_CUTTERS_THRESH:
            reward = -1
        # If check for bearing failures is enabled and a bearing failure occurs
        elif self.CHECK_BEARING_FAILURE and damaged_bearing:
            reward = self.BEARING_FAILURE_PENALTY
        # If no cutters are acted on. This will give the highest reward. We want to
        # promote that behaviour since the agent is a bit over-active
        # TODO: consider adding a weight factor on this term to let the agent behove
        # more the way we want. Then we also can tune
        elif len(acted_on_cutters) == 0:
            reward = n_good_cutters / self.n_c_tot
        # Standard.Compute reward based on various factors related to maintenance effort
        else:
            ratio1 = n_good_cutters / self.n_c_tot
            ratio2 = self._compute_replacement_penalty(replaced_cutters)
            ratio3 = self._compute_movement_penalty(moved_cutters)
            ratio4 = self._compute_distance_penalty(dist_cutters)
            change_penalty = self.T_I * self.DELTA
            reward = ratio1 - ratio2 - ratio3 - ratio4 - change_penalty

        return reward

    def _compute_replacement_penalty(self, replaced_cutters: list) -> float:
        """The values returned by this function will fall within the range of 0 to 1.
        It represents the relative weight or penalty associated with the replacement of
        cutters."""

        # weighted representation of cutters to penalize changing of outer
        # cutters more than inner cutters. Outer cutters are more demanding.
        weighted_cutters = np.linspace(1, 2, num=self.n_c_tot)
        return (
            np.sum(np.take(weighted_cutters, replaced_cutters))
            / np.sum(weighted_cutters)
        ) * self.ALPHA

    def _compute_movement_penalty(self, moved_cutters: list) -> float:
        """the values returned by this function will range from 0 to 1. It represents
        the penalty associated with the movement of cutters"""

        weighted_cutters = np.linspace(1, 2, num=self.n_c_tot)
        return (
            np.sum(np.take(weighted_cutters, moved_cutters)) / np.sum(weighted_cutters)
        ) * self.BETA

    def _compute_distance_penalty(self, dist_cutters: float) -> float:
        """calculates a penalty based on the relative distance between acted-on cutters.
        The return values will be non-negative floating-point numbers. The exact range
        depends on the specific number of cutters, but the values should typically be
        within the range of 0 to (n_c_tot / 2). This penalty represents the additional
        cost incurred by moving cutters across the cutterhead."""

        return ((dist_cutters + 1) / self.n_c_tot) * self.GAMMA


class CustomEnv(gym.Env):
    """Implementation of the custom environment that simulates the cutter wear
    and provides the agent with a state and reward signal.
    """

    def __init__(
        self,
        n_c_tot: int,
        LIFE: int,
        MAX_STROKES: int,
        STROKE_LENGTH: float,
        cutter_pathlenghts: float,
        CUTTERHEAD_RADIUS: float,
        reward_fn: Callable,
    ) -> None:
        """Initializing custom environment for a TBM cutter operation.

        Args:
            n_c_tot (int): total number of cutters
            LIFE (int): theoretical durability of one cutter [m]
            MAX_STROKES (int): numer of strokes to simulate
            STROKE_LENGTH (float): length of one stroke [m]
            cutter_pathlenghts (float): rolling lengths [m]
            CUTTERHEAD_RADIUS (float): radius of cutterhead
        """
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_c_tot * n_c_tot,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_c_tot,))

        self.n_c_tot = n_c_tot
        self.LIFE = LIFE
        self.MAX_STROKES = MAX_STROKES
        self.STROKE_LENGTH = STROKE_LENGTH
        self.cutter_pathlenghts = cutter_pathlenghts
        self.R = CUTTERHEAD_RADIUS
        self.reward_fn = reward_fn

        # state variables assigned in class methods:
        self.state: NDArray
        self.epoch: int
        self.replaced_cutters: list
        self.moved_cutters: list
        self.good_cutters: NDArray
        self.inwards_moved_cutters: list
        self.wrong_moved_cutters: list
        self.penetration: NDArray

    def step(self, actions: NDArray) -> tuple[NDArray, float, bool, dict]:
        """Main function that moves the environment one step further.
        - Updates the state and reward.
        - Checks if the terminal state is reached.
        """
        # replace cutters based on action of agent
        self.state = self._implement_action(actions, self.state)

        # compute reward
        self.broken_cutters = np.where(self.state == 0)[0]  # for statistics
        self.good_cutters = np.where(self.state > 0)[0]
        n_good_cutters = len(self.good_cutters)
        self.moved_cutters = sorted(
            self.inwards_moved_cutters + self.wrong_moved_cutters
        )
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

        reward = self.reward_fn(
            self.replaced_cutters, self.moved_cutters, n_good_cutters, damaged_bearing
        )

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
        """Function that interprets the "raw action" and modifies the state.
        TODO: this is the function which takes longest time to run. Look for
        improvements.
        """
        state_new = state_before
        self.replaced_cutters = []
        self.inwards_moved_cutters = []
        self.wrong_moved_cutters = []
        # iterate through cutters starting from outside
        for i in np.arange(self.n_c_tot)[::-1]:
            cutter = action[i * self.n_c_tot : i * self.n_c_tot + self.n_c_tot]
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
        """function generates a random walk within the limits 0 and 1"""
        bounds = 0.05

        x = [np.random.uniform(low=0, high=1, size=1)]

        for move in np.random.uniform(low=-bounds, high=bounds, size=n_dp):
            x_temp = x[-1] + move
            if x_temp <= 1 and x_temp >= 0:
                x.append(x_temp)
            else:
                x.append(x[-1] - move)

        return np.array(x[1:])

    def generate(
        self,
        Jv_low: int = 0,
        Jv_high: int = 22,
        UCS_center: int = 80,
        UCS_range: int = 30,
    ) -> tuple:
        """Function generates TBM recordings for one episode. Equations and
        models based on Delisio & Zhao (2014) - "A new model for TBM
        performance prediction in blocky rock conditions",
        http://dx.doi.org/10.1016/j.tust.2014.06.004"""

        Jv_s = (
            self._rand_walk_with_bounds(self.MAX_STROKES) * (Jv_high - Jv_low) + Jv_low
        )  # [joints / m3]
        UCS_s = (
            UCS_center + self._rand_walk_with_bounds(self.MAX_STROKES) * UCS_range
        )  # [MPa]

        # eq 9, Delisio & Zhao (2014) - [kN/m/mm/rot]
        FPIblocky_s = np.squeeze(np.exp(6) * Jv_s**-0.82 * UCS_s**0.17)

        brokens = np.zeros(shape=(self.MAX_STROKES, self.n_c_tot))

        for stroke in range(self.MAX_STROKES):
            # based on FPI-blocky, cutters have different likelyhoods to break
            if FPIblocky_s[stroke] > 200 and FPIblocky_s[stroke] <= 300:
                if np.random.randint(0, 100) == 0:
                    size = np.random.randint(1, 4)
                    selection = np.random.choice(
                        np.arange(self.n_c_tot), replace=False, size=size
                    )
                    brokens[stroke, :][selection] = 1
            elif FPIblocky_s[stroke] > 100 and FPIblocky_s[stroke] <= 200:
                if np.random.randint(0, 50) == 0:
                    size = np.random.randint(1, 4)
                    selection = np.random.choice(
                        np.arange(self.n_c_tot), replace=False, size=size
                    )
                    brokens[stroke, :][selection] = 1
            elif FPIblocky_s[stroke] >= 50 and FPIblocky_s[stroke] <= 100:
                if np.random.randint(0, 10) == 0:
                    size = np.random.randint(1, 4)
                    selection = np.random.choice(
                        np.arange(self.n_c_tot), replace=False, size=size
                    )
                    brokens[stroke, :][selection] = 1
            elif FPIblocky_s[stroke] < 50:
                if np.random.randint(0, 100) == 0:
                    size = np.random.randint(1, 4)
                    selection = np.random.choice(
                        np.arange(self.n_c_tot), replace=False, size=size
                    )
                    brokens[stroke, :][selection] = 1

        # eq 13, Delisio & Zhao (2014)
        TF_s = np.squeeze((-523 * np.log(Jv_s) + 2312) * (self.R * 2))  # [kN]
        TF_s = np.where(TF_s > 20_000, 20_000, TF_s)

        # eq 7, Delisio & Zhao (2014)
        # TF_s already considers shield friction
        penetration = (TF_s / (self.R * 2)) / FPIblocky_s  # [mm/rot]

        return Jv_s, UCS_s, FPIblocky_s, brokens, TF_s, penetration

    def reset(self) -> NDArray:
        """reset an environment to its initial state"""
        self.state = np.full((self.n_c_tot), 1)  # start with new cutters
        self.epoch = 0  # reset epoch counter
        # generate new TBM data for episode
        (
            self.Jv_s,
            self.UCS_s,
            self.FPIblocky_s,
            self.brokens,
            self.TF_s,
            self.penetration,
        ) = self.generate()
        return self.state

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass


if __name__ == "__main__":
    # visualization of all possible reward states
    n_c_tot = 28  # total number of cutters
    broken_cutters_thresh = 0.5
    check_bearing_failure = False

    reward_fn = Reward(n_c_tot, broken_cutters_thresh, check_bearing_failure)

    cutters = np.arange(n_c_tot)

    n_repls = []
    n_moves = []
    n_good_cutters = []
    r_s = []

    for _ in range(10_000):
        n_repl = np.random.randint(0, n_c_tot)
        replaced_cutters = np.sort(np.random.choice(cutters, n_repl, replace=False))
        n_move = np.random.randint(0, n_c_tot - n_repl)
        moved_cutters = np.sort(
            np.random.choice(
                np.delete(cutters, replaced_cutters), n_move, replace=False
            )
        )

        good_cutters = np.random.randint(n_repl + n_move, n_c_tot)
        # print(n_repl, n_move, good_cutters)

        r = reward_fn(list(replaced_cutters), list(moved_cutters), good_cutters, False)
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
    # plt.tight_layout()
