"""
Schema definition and validation of the hierarchical config files.

Benefits of validation of config in one central pydantic scheme
- Easy to read and understand
- Easy to maintain
- Easy to extend
- Keeping all validation logic within the Pydantic schema ensures that any changes or
additions to the validation rules are handled in one place, reducing the risk of
inconsistencies.
- Functions and classes focuses on their main purpose and not on validation logic
- Early validation: By validating the parameters as soon as they are parsed from the
configuration file, you catch errors early in the process, before they can propagate to
other parts of the system.

"""

# from multiprocessing.sharedctypes import Value

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, root_validator, validator


class Agent(BaseModel):
    NAME: Literal["PPO", "A2C", "DDPG", "TD3", "SAC", "PPO-LSTM"]
    CHECKPOINT_INTERVAL: int
    MAX_NO_IMPROVEMENT_EVALS: int
    EPISODES: int
    agent_params: dict


class REWARD(BaseModel):
    CHECK_BEARING_FAILURE: bool
    BEARING_FAILURE_PENALTY: int = Field(
        ..., ge=0, description="Penalty for bearing failure"
    )
    BROKEN_CUTTERS_THRESH: float = Field(
        ...,
        ge=0.5,
        le=1,
        description="Threshold for functional cutters as a percentage",
    )
    T_I: float = Field(
        ..., gt=0, description="Cost of entering the cutterhead for maintenance"
    )
    ALPHA: float = Field(
        ..., ge=0, le=1, description="Weighting factor for cutter replacement"
    )
    BETA: float = Field(
        ..., ge=0, le=1, description="Weighting factor for cutter movement"
    )
    GAMMA: float = Field(
        ..., ge=0, le=1, description="Weighting factor for cutter distance"
    )
    DELTA: float = Field(
        ..., ge=0, le=1, description="Weighting factor for entering cutterhead"
    )

    @root_validator  # used for validation of multiple fields
    def check_weight_factors(cls, values: dict[str, Any]):
        total = (
            values.get("ALPHA", 0)
            + values.get("BETA", 0)
            + values.get("GAMMA", 0)
            + values.get("DELTA", 0)
        )
        assert (
            total == 1
        ), f"Weighting factors (ALPHA, BETA, GAMMA, DELTA) must sum up to 1, but got {total}"
        return values


class TBM(BaseModel):
    CUTTERHEAD_RADIUS: int = Field(
        ..., lt=16, description="Maximum allowed cutterhead radius"
    )
    TRACK_SPACING: float
    LIFE: int
    STROKE_LENGTH: float
    MAX_STROKES: int


class EXP(BaseModel):
    MODE: Literal["optimization", "training", "execution"]
    CHECK_ENV: bool
    DEBUG: bool
    STUDY: str
    CHECKPOINT_INTERVAL: int
    EPISODES: int
    PLOT_PROGRESS: bool
    DETERMINISTIC: bool
    SEED_ALGORITHM: int | None
    SEED_ALL: int | None

    @validator("STUDY")
    def check_agent(cls, STUDY: str):
        agent_name = STUDY.split("_")[0]
        valid_agents = [
            "PPO",
            "A2C",
            "DDPG",
            "SAC",
            "TD3",
            "PPO-LSTM",
        ]
        if agent_name not in valid_agents:
            raise ValueError(
                f"The study object with {agent_name} has not a valid agent."
            )
        return agent_name


class TRAIN(BaseModel):
    LOAD_PARAMS_FROM_STUDY: bool
    N_EVAL_EPISODES_TRAINING: int
    N_DUPLICATES: int


class OPT(BaseModel):
    DEFAULT_TRIAL: bool
    MAX_NO_IMPROVEMENT_EVALS: int = Field(..., gt=0)
    N_SINGLE_RUN_OPTUNA_TRIALS: int
    N_CORES_PARALLELL: int = Field(..., ge=-1, description="Number of cores to use")
    N_PARALLELL_PROCESSES: int = Field(
        ..., gt=0, description="Number of parallel processes"
    )
    N_EVAL_EPISODES_OPTIMIZATION: int
    N_EVAL_EPISODES_REWARD: int
    STUDYS: list[str]
    AGENTS: list[str]
    BEST_PERFORMING_ALGORITHM_PATH: Path

    @root_validator
    def check_cores_and_processes(cls, values: dict[str, Any]):
        n_cores_parallell = values.get("N_CORES_PARALLELL")
        n_parallell_processes = values.get("N_PARALLELL_PROCESSES")
        if not (n_cores_parallell >= n_parallell_processes or n_cores_parallell == -1):
            raise ValueError("Num cores must be >= num parallel processes.")
        return values


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

    @root_validator
    def check_study_exists(cls, values: dict):
        exp = values.get("EXP")
        train = values.get("TRAIN")
        if (
            train is not None
            and exp is not None
            and train.LOAD_PARAMS_FROM_STUDY
            and exp.MODE == "training"
        ):
            if not Path(f"./results/{exp.STUDY}.db").exists():
                raise ValueError("The study object does not exist")
        return values
