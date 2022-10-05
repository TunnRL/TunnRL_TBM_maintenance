"""
Tests of functionality in XX_experiment_factory.py

To run functionality run at root:
    pytest

To enable coverage reporting, invoke pytest with the --cov option: "pytest --cov"

@author: Tom F. Hansen
"""

import pytest
from stable_baselines3.common.base_class import BaseAlgorithm

from utils.XX_experiment_factory import load_best_model


def test_load_best_model():
    res = load_best_model("PPO", "checkpoints", "_")
    assert isinstance(res, BaseAlgorithm | None), f"{res} is not a valid SB3 algorithm"
