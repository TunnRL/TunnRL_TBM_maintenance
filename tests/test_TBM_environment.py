"""
Tests and demonstrations of functionality in XX_TBM_environment.py

To run functionality run at root:
    pytest
To enable coverage reporting, invoke pytest with the --cov option: "pytest --cov"

@author: Tom F. Hansen
"""

from dataclasses import dataclass

import pytest

from tunnrl_tbm_maintenance.TBM_environment import Reward


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
                expected_reward=pytest.approx(0.9, rel=1e-2),
            ),
            RewardTestCase(
                replaced_cutters=[],
                moved_cutters=[1, 2],
                n_good_cutters=40,
                damaged_bearing=False,
                expected_reward=pytest.approx(0.85, rel=1e-2),
            ),
            RewardTestCase(
                replaced_cutters=[1, 2],
                moved_cutters=[3, 4],
                n_good_cutters=40,
                damaged_bearing=False,
                expected_reward=pytest.approx(0.75, rel=1e-2),
            ),
            RewardTestCase(
                replaced_cutters=[],
                moved_cutters=[],
                n_good_cutters=40,
                damaged_bearing=True,
                expected_reward=0.0,
            ),
            RewardTestCase(
                replaced_cutters=[],
                moved_cutters=[],
                n_good_cutters=30,
                damaged_bearing=False,
                expected_reward=0.75,
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
            (20, 10, 0.5),
            (50, 25, 0.5),
            (100, 50, 0.5),
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
            (0.5, 20, 1.0),
            (0.75, 30, 1.0),
            (0.85, 34, 1.0),
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

    def test_invalid_weight_factors(self):
        with pytest.raises(AssertionError):
            Reward(ALPHA=0.1, BETA=0.2, GAMMA=0.3, DELTA=0.7)

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
                pytest.approx(0.9, rel=1e-2),
            ),
            (0.1, 0.2, 0.3, 0.4, [1, 2], [], 40, False, pytest.approx(0.9, rel=1e-2)),
            (0.4, 0.3, 0.2, 0.1, [1, 2], [], 40, False, pytest.approx(0.9, rel=1e-2)),
            (0.5, 0.2, 0.2, 0.1, [1, 2], [], 40, False, pytest.approx(0.9, rel=1e-2)),
            (0.3, 0.3, 0.2, 0.2, [1, 2], [], 40, False, pytest.approx(0.9, rel=1e-2)),
            (0.2, 0.3, 0.3, 0.2, [1, 2], [], 40, False, pytest.approx(0.9, rel=1e-2)),
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
