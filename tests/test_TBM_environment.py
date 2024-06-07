"""
Tests and demonstrations of functionality in XX_TBM_environment.py

To run functionality run at root:
    pytest
To enable coverage reporting, invoke pytest with the --cov option: "pytest --cov"

@author: Tom F. Hansen
"""

import pytest

from tunnrl_tbm_maintenance.TBM_environment import Reward


class TestReward:
    def test_initialization(self):
        # Test if the class initializes correctly and raises assertion error if weights
        # don't sum to 1
        with pytest.raises(AssertionError):
            Reward(10, 0.5, True, -1, ALPHA=0.1, BETA=0.2, GAMMA=0.3, DELTA=0.8)
        reward = Reward(10, 0.5, True, -1)
        assert reward.n_c_tot == 10
        assert reward.BROKEN_CUTTERS_THRESH == 0.5

    @pytest.mark.parametrize(
        "replaced_cutters, moved_cutters, n_good_cutters, damaged_bearing, expected_reward",
        [
            ([], [], 4, False, -1.0),  # Below threshold
            ([], [], 6, True, -1),  # Bearing failure
            ([], [], 6, False, 0.6),  # No cutters acted on
            ([0, 1], [2, 3], 6, False, 0.6 - 0.2 - 0.3 - 0.25 - 0.25),  # Standard case
        ],
    )
    def test_call(
        self,
        replaced_cutters,
        moved_cutters,
        n_good_cutters,
        damaged_bearing,
        expected_reward,
    ):
        reward = Reward(40, 0.85, True, 0)
        assert reward(
            replaced_cutters, moved_cutters, n_good_cutters, damaged_bearing
        ) == pytest.approx(expected_reward)

    def test_compute_max_reward(self):
        reward = Reward(10, 0.5, True, 0)
        assert reward._compute_max_reward(8) == 0.8

    def test_enter_face_penalty(self):
        reward = Reward(10, 0.5, True, 0)
        assert reward._enter_face_penalty() == 0.25

    def test_compute_replacement_penalty(self):
        reward = Reward(10, 0.5, True, 0)
        replaced_cutters = [0, 1, 2]
        expected_penalty = (1 + 1.1111111111111112 + 1.2222222222222223) / 15 * 0.2
        assert reward._compute_replacement_penalty(replaced_cutters) == pytest.approx(
            expected_penalty
        )

    def test_compute_movement_penalty(self):
        reward = Reward(10, 0.5, True, 0)
        moved_cutters = [0, 1, 2]
        expected_penalty = (1 + 1.1111111111111112 + 1.2222222222222223) / 15 * 0.3
        assert reward._compute_movement_penalty(moved_cutters) == pytest.approx(
            expected_penalty
        )

    def test_compute_distance_penalty(self):
        reward = Reward(10, 0.5, True, 0)
        dist_cutters = 3
        expected_penalty = (4 / 10) * 0.25
        assert reward._compute_distance_penalty(dist_cutters) == pytest.approx(
            expected_penalty
        )
