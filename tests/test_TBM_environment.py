"""
Tests and demonstrations of functionality in XX_TBM_environment.py

To run functionality run at root:
    pytest
To enable coverage reporting, invoke pytest with the --cov option: "pytest --cov"

@author: Tom F. Hansen
"""

from dataclasses import dataclass

import numpy as np
import pytest

from tunnrl_tbm_maintenance.TBM_environment import CustomEnv, Reward


class TestCustomEnv:
    @pytest.fixture
    def custom_env_instance(self):
        reward_fn = Reward()
        cutter_pathlenghts = np.array(
            [
                0.31415927,
                0.9424778,
                1.57079633,
                2.19911486,
                2.82743339,
                3.45575192,
                4.08407045,
                4.71238898,
                5.34070751,
                5.96902604,
                6.59734457,
                7.2256631,
                7.85398163,
                8.48230016,
                9.1106187,
                9.73893723,
                10.36725576,
                10.99557429,
                11.62389282,
                12.25221135,
                12.88052988,
                13.50884841,
                14.13716694,
                14.76548547,
                15.393804,
                16.02212253,
                16.65044106,
                17.27875959,
                17.90707813,
                18.53539666,
                19.16371519,
                19.79203372,
                20.42035225,
                21.04867078,
                21.67698931,
                22.30530784,
                22.93362637,
                23.5619449,
                24.19026343,
                24.81858196,
                25.44690049,
            ]
        )
        env = CustomEnv(
            n_c_tot=40,
            LIFE=400000,
            MAX_STROKES=1000,
            STROKE_LENGTH=1.8,
            cutter_pathlenghts=cutter_pathlenghts,
            CUTTERHEAD_RADIUS=4,
            reward_fn=reward_fn,
        )
        return env  # default values

    def test_rand_walk_with_bounds(self, custom_env_instance):
        env = custom_env_instance
        n_dp = 100
        x = env._rand_walk_with_bounds(n_dp)

        # Check if the length of the generated array is correct
        assert len(x) == n_dp

        # Check if the values are within the bounds
        assert np.all(x >= 0) and np.all(x <= 1)

        # Check if the values are within the specified bounds
        assert np.all(x >= 0.05) and np.all(x <= 0.95)


@dataclass
class RewardTestCase:
    replaced_cutters: list
    moved_cutters: list
    n_good_cutters: int
    damaged_bearing: bool
    expected_reward: float


class TestReward:
    @pytest.fixture
    def reward_instance(self):
        return Reward()  # default values

    @pytest.mark.parametrize(
        "test_case",
        [
            RewardTestCase(
                replaced_cutters=[],
                moved_cutters=[],
                n_good_cutters=40,
                damaged_bearing=False,
                expected_reward=1.0,
            ),
            RewardTestCase(
                replaced_cutters=[],
                moved_cutters=[],
                n_good_cutters=20,
                damaged_bearing=False,
                expected_reward=-1,
            ),
            RewardTestCase(
                replaced_cutters=[],
                moved_cutters=[],
                n_good_cutters=0,
                damaged_bearing=False,
                expected_reward=-1,
            ),
            RewardTestCase(
                replaced_cutters=[1, 2],
                moved_cutters=[],
                n_good_cutters=40,
                damaged_bearing=False,
                expected_reward=pytest.approx(0.73, rel=1e-2),  # type: ignore
            ),
            RewardTestCase(
                replaced_cutters=[],
                moved_cutters=[1, 2],
                n_good_cutters=40,
                damaged_bearing=False,
                expected_reward=pytest.approx(0.727, rel=1e-2),  # type: ignore
            ),
            RewardTestCase(
                replaced_cutters=[1, 2],
                moved_cutters=[3, 4],
                n_good_cutters=40,
                damaged_bearing=False,
                expected_reward=pytest.approx(0.707, rel=1e-2),  # type: ignore
            ),
            RewardTestCase(
                replaced_cutters=[],
                moved_cutters=[],
                n_good_cutters=40,
                damaged_bearing=True,
                expected_reward=0.0,
            ),
            RewardTestCase(
                replaced_cutters=[1, 2, 3, 4, 30, 31],
                moved_cutters=[5, 6, 7, 8, 20, 21],
                n_good_cutters=36,
                damaged_bearing=False,
                expected_reward=pytest.approx(0.392, rel=1e-2),  # type: ignore
            ),
            RewardTestCase(
                replaced_cutters=[],
                moved_cutters=[],
                n_good_cutters=5,
                damaged_bearing=False,
                expected_reward=-1.0,
            ),
        ],
    )
    def test_reward(self, reward_instance, test_case):
        reward = reward_instance(
            replaced_cutters=test_case.replaced_cutters,
            moved_cutters=test_case.moved_cutters,
            n_good_cutters=test_case.n_good_cutters,
            damaged_bearing=test_case.damaged_bearing,
        )
        assert (
            reward == test_case.expected_reward
        ), f"Expected {test_case.expected_reward}, but got {reward}"

    @pytest.mark.parametrize(
        "n_c_tot, n_good_cutters, expected_reward",
        [
            (10, 10, 1.0),
            (20, 20, 1.0),
            (50, 50, 1.0),
            (100, 50, -1),
        ],
    )
    def test_n_c_tot(self, n_c_tot, n_good_cutters, expected_reward):
        reward_instance = Reward(n_c_tot=n_c_tot)
        reward = reward_instance([], [], n_good_cutters, False)
        assert (
            reward == expected_reward
        ), f"Expected reward {expected_reward} for n_c_tot {n_c_tot} and n_good_cutters {n_good_cutters}, but got {reward}"

    @pytest.mark.parametrize(
        "BROKEN_CUTTERS_THRESH, n_good_cutters, expected_reward",
        [
            (0.5, 20, 0.5),
            (0.5, 19, -1.0),
            (1.0, 40, 1.0),
            (0.85, 33, -1.0),  # Below threshold should give -1.0
        ],
    )
    def test_broken_cutters_thresh(
        self, BROKEN_CUTTERS_THRESH, n_good_cutters, expected_reward
    ):
        reward_instance = Reward(BROKEN_CUTTERS_THRESH=BROKEN_CUTTERS_THRESH)
        reward = reward_instance([], [], n_good_cutters, False)
        assert (
            reward == expected_reward
        ), f"Expected reward {expected_reward} for BROKEN_CUTTERS_THRESH {BROKEN_CUTTERS_THRESH} and n_good_cutters {n_good_cutters}, but got {reward}"

    @pytest.mark.parametrize(
        "ALPHA, BETA, GAMMA, DELTA, replaced_cutters, moved_cutters, n_good_cutters, damaged_bearing, expected_reward",
        [
            (0.25, 0.25, 0.25, 0.25, [], [], 40, False, 1.0),
            (0.1, 0.2, 0.3, 0.4, [], [], 40, False, 1.0),
            (0.4, 0.3, 0.2, 0.1, [], [], 40, False, 1.0),
            (0.5, 0.2, 0.2, 0.1, [], [], 40, False, 1.0),
            (0.3, 0.3, 0.2, 0.2, [], [], 40, False, 1.0),
            (0.2, 0.3, 0.3, 0.2, [], [], 40, False, 1.0),
            (
                0.25,
                0.25,
                0.25,
                0.25,
                [1, 2],
                [],
                40,
                False,
                pytest.approx(0.728, rel=1e-2),
            ),
            (0.1, 0.2, 0.3, 0.4, [1, 2], [], 40, False, pytest.approx(0.58, rel=1e-2)),
            (0.4, 0.3, 0.2, 0.1, [1, 2], [], 40, False, pytest.approx(0.876, rel=1e-2)),
            (0.5, 0.2, 0.2, 0.1, [1, 2], [], 40, False, pytest.approx(0.872, rel=1e-2)),
            (0.3, 0.3, 0.2, 0.2, [1, 2], [], 40, False, pytest.approx(0.779, rel=1e-2)),
            (0.2, 0.3, 0.3, 0.2, [1, 2], [], 40, False, pytest.approx(0.778, rel=1e-2)),
        ],
    )
    def test_valid_weight_factors(
        self,
        ALPHA,
        BETA,
        GAMMA,
        DELTA,
        replaced_cutters,
        moved_cutters,
        n_good_cutters,
        damaged_bearing,
        expected_reward,
    ):
        reward_instance = Reward(ALPHA=ALPHA, BETA=BETA, GAMMA=GAMMA, DELTA=DELTA)
        reward = reward_instance(
            replaced_cutters, moved_cutters, n_good_cutters, damaged_bearing
        )
        assert (
            reward == expected_reward
        ), f"Expected reward {expected_reward} for ALPHA {ALPHA}, BETA {BETA}, GAMMA {GAMMA}, DELTA {DELTA}, but got {reward}"

    def test_reward_range(self, reward_instance):
        reward = reward_instance([], [], 40, False)
        assert -1 <= reward <= 1, f"Reward {reward} is out of range [-1, 1]"
