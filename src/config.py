"""
Schema definition and validation of the hierarchical config files.

@author: Tom F. Hansen
"""

# from multiprocessing.sharedctypes import Value

from hydra.core.config_store import ConfigStore
from pydantic import validator
from pydantic.dataclasses import dataclass


@dataclass
class Agent:
    NAME: str
    CHECKPOINT_INTERVAL: int
    MAX_NO_IMPROVEMENT_EVALS: int


@dataclass
class REWARD:
    BROKEN_CUTTERS_THRESH: float
    T_I: float
    ALPHA: float
    BETA: float
    GAMMA: float
    DELTA: float
    CHECK_BEARING_FAILURE: bool
    BEARING_FAILURE_PENALTY: int


@dataclass
class TBM:
    @validator("CUTTERHEAD_RADIUS")
    def check_radius(cls, CUTTERHEAD_RADIUS: int) -> int:
        if CUTTERHEAD_RADIUS > 15:
            raise ValueError("No TBM's have greater radius than 15 m")

    CUTTERHEAD_RADIUS: int
    TRACK_SPACING: float
    LIFE: int
    STROKE_LENGTH: float
    MAX_STROKES: int


@dataclass
class EXP:
    @validator("MODE")
    def check_mode(cls, MODE: str) -> str:
        if MODE not in ["optimization", "training", "execution"]:
            raise ValueError(f"{MODE} is not a valid mode")

    MODE: str
    CHECK_ENV: bool
    DEBUG: bool
    STUDY: str
    CHECKPOINT_INTERVAL: int
    EPISODES: int
    PLOT_PROGRESS: bool
    DETERMINISTIC: bool


@dataclass
class OPT:
    DEFAULT_TRIAL: bool
    MAX_NO_IMPROVEMENT_EVALS: int
    N_SINGLE_RUN_OPTUNA_TRIALS: int
    N_CORES_PARALLELL: int
    N_PARALLELL_PROCESSES: int
    N_EVAL_EPISODES_OPTIMIZATION: int
    N_EVAL_EPISODES_REWARD: int


@dataclass
class TRAIN:
    LOAD_PARAMS_FROM_STUDY: bool
    N_EVAL_EPISODES_TRAINING: int


@dataclass
class EXECUTE:
    EXECUTION_MODEL: str
    NUM_TEST_EPISODES: int
    VISUALIZE_EPISODES: bool


@dataclass
class Config:
    agent: Agent
    REWARD: REWARD
    TBM: TBM
    EXP: EXP
    OPT: OPT
    TRAIN: TRAIN
    EXECUTE: EXECUTE


cs = ConfigStore.instance()
# name `base_config` is used for matching it with the main.yaml's default section
cs.store(name="base_config", node=Config)
