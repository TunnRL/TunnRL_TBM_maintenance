"""
Tests and demonstrations of functionality in XX_TBM_environment.py

To run functionality run at root:
    pytest
To enable coverage reporting, invoke pytest with the --cov option: "pytest --cov"

@author: Tom F. Hansen
"""

import pytest

from src.XX_TBM_environment import Reward


@pytest.mark.parametrize("replaced_cutters", [[], [12, 25]])
@pytest.mark.parametrize("check_bearing_failure", [False, True])
def test_Reward(check_bearing_failure, replaced_cutters):
    reward_fn = Reward(
        n_c_tot=28,
        BROKEN_CUTTERS_THRESH=0.85,
        CHECK_BEARING_FAILURE=check_bearing_failure,
    )
    res = reward_fn(
        replaced_cutters=replaced_cutters,
        moved_cutters=[1, 2, 3],
        n_good_cutters=5,
        damaged_bearing=False,
    )
    assert isinstance(res, float | int), f"{res} is not a float"
    assert (
        -1000 <= res <= 1000
    ), f"Function return {res}. It should not be possible to reach values outside the range [-1000, 1000]"
