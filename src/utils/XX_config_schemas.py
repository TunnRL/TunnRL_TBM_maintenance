"""
Schema definition and validation of the hierarchical config files.

TODO: add more checks

@author: Tom F. Hansen
"""

# from multiprocessing.sharedctypes import Value

from pathlib import Path

from pydantic import BaseModel, validator


class Agent(BaseModel):
    @validator("NAME")
    def check_name(cls, NAME: str) -> str:
        if NAME not in ["PPO", "A2C", "DDPG", "TD3", "SAC", "PPO-LSTM"]:
            raise ValueError(
                f"{NAME} is not a valid algorithm name. Remember to use uppercase"
            )

    NAME: str
    CHECKPOINT_INTERVAL: int
    MAX_NO_IMPROVEMENT_EVALS: int
    EPISODES: int
    agent_params: dict


class REWARD(BaseModel):
    BROKEN_CUTTERS_THRESH: float
    T_I: float
    ALPHA: float
    BETA: float
    GAMMA: float
    DELTA: float
    CHECK_BEARING_FAILURE: bool
    BEARING_FAILURE_PENALTY: int


class TBM(BaseModel):
    @validator("CUTTERHEAD_RADIUS")
    def check_radius(cls, CUTTERHEAD_RADIUS: int) -> int:
        if CUTTERHEAD_RADIUS > 15:
            raise ValueError("No TBM's have greater radius than 15 m")

    CUTTERHEAD_RADIUS: int
    TRACK_SPACING: float
    LIFE: int
    STROKE_LENGTH: float
    MAX_STROKES: int


class EXP(BaseModel):
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
    SEED_ALGORITHM: int | None
    SEED_ALL: int | None


class OPT(BaseModel):
    DEFAULT_TRIAL: bool
    MAX_NO_IMPROVEMENT_EVALS: int
    N_SINGLE_RUN_OPTUNA_TRIALS: int
    N_CORES_PARALLELL: int
    N_PARALLELL_PROCESSES: int
    N_EVAL_EPISODES_OPTIMIZATION: int
    N_EVAL_EPISODES_REWARD: int


class TRAIN(BaseModel):
    LOAD_PARAMS_FROM_STUDY: bool
    N_EVAL_EPISODES_TRAINING: int
    N_DUPLICATES: int


class EXECUTE(BaseModel):
    EXECUTION_MODEL: str
    NUM_TEST_EPISODES: int
    VISUALIZE_STATE_ACTION_PLOT: bool


class PLOT(BaseModel):
    FIGURE_WIDTH: float
    DATA_DIR: Path
    AGENT_NAME: str
    STUDY_NAME: str
    PLOTS_TO_MAKE: list[str]
    VISUALIZATION_MODE: str
    PRINT_TRESH: None | int
    CHOOSE_NUM_BEST_REWARDS: None | int
    MAKE_MAX_REWARD_LIST: bool


class Config(BaseModel):
    agent: Agent
    REWARD: REWARD
    TBM: TBM
    EXP: EXP
    OPT: OPT
    TRAIN: TRAIN
    EXECUTE: EXECUTE
    PLOT: PLOT
