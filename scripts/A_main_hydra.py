"""
Main code that runs one of:
- hyperparameter optimization study with OPTUNA
- training a model with optimized parameters
- execute a trained model
"""

import warnings
from pathlib import Path

import hydra
import numpy as np
import optuna
from omegaconf import DictConfig
from rich.traceback import install
from stable_baselines3.common.env_checker import check_env

from tunnrl_tbm_maintenance.experiment_factory import Optimization
from tunnrl_tbm_maintenance.TBM_environment import CustomEnv, Reward
from tunnrl_tbm_maintenance.train_optimize_execute import (
    run_execution,
    run_optimization,
    run_training,
)
from tunnrl_tbm_maintenance.utility import parse_validate_hydra_config


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3.2")
def main(cfg: DictConfig) -> None:
    install()  # better traceback messages
    ###############################################################################
    # SETUP DIRECTORY STRUCTURE
    ###############################################################################
    opt_dir = Path.cwd() / "optimization"
    ckpt_dir = Path.cwd() / "checkpoints"
    res_dir = Path.cwd() / "results"
    graphics_dir = Path.cwd() / "graphics"
    exp_dir = Path.cwd() / "experiments"
    if not opt_dir.exists():
        opt_dir.mkdir(exist_ok=True)
    if not ckpt_dir.exists():
        ckpt_dir.mkdir(exist_ok=True)
    if not res_dir.exists():
        res_dir.mkdir(exist_ok=True)
    if not graphics_dir.exists():
        graphics_dir.mkdir(exist_ok=True)
    if not exp_dir.exists():
        exp_dir.mkdir(exist_ok=True)

    ###############################################################################
    # WARNINGS AND ERROR CHECKING INPUT VARIABLES
    ###############################################################################
    p_cfg, rich_console = parse_validate_hydra_config(cfg, print_config=True)

    if p_cfg.EXP.DEBUG:
        p_cfg.EXP.EPISODES = 20
        p_cfg.EXP.CHECKPOINT_INTERVAL = 6
        p_cfg.OPT.N_PARALLELL_PROCESSES = 1
        p_cfg.OPT.N_CORES_PARALLELL = 1

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    if p_cfg.OPT.DEFAULT_TRIAL:
        warnings.warn("Optimization runs are started with default parameter values")

    assert (
        p_cfg.OPT.N_CORES_PARALLELL >= p_cfg.OPT.N_PARALLELL_PROCESSES
        or p_cfg.OPT.N_CORES_PARALLELL == -1
    ), "Num cores must be >= num parallell processes."
    if p_cfg.OPT.N_PARALLELL_PROCESSES > 1 and (
        p_cfg.EXP.MODE == "training" or p_cfg.EXP.MODE == "execution"
    ):
        warnings.warn("No parallellization in training and execution mode")

    if p_cfg.TRAIN.LOAD_PARAMS_FROM_STUDY is True and p_cfg.EXP.MODE == "training":
        assert Path(
            f"./results/{p_cfg.EXP.STUDY}.db"
        ).exists(), "The study object does not exist"

        assert (
            p_cfg.agent.NAME == (p_cfg.EXP.STUDY).split("_")[0]
        ), "Agent name and study name must be similar"

    agent_name = p_cfg.EXP.STUDY.split("_")[0]
    assert agent_name in [
        "PPO",
        "A2C",
        "DDPG",
        "SAC",
        "TD3",
        "PPO-LSTM",
    ], f"{agent_name} is not a valid agent."

    if (
        p_cfg.EXP.MODE == "execute"
        and p_cfg.EXP.DETERMINISTIC
        and agent_name in ["PPO", "A2C"]
    ):
        warnings.warn(
            "Deterministic should be true for running predictions with PPO and A2C. \
                Ref: https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html"
        )

    ###############################################################################
    # COMPUTED/DERIVED VARIABLES AND INSTANTIATIONS
    # - Defines the reward function
    # - Defines the custom environment
    # - Defines the experiment setup (optimization)
    ###############################################################################

    n_c_tot = (
        p_cfg.TBM.CUTTERHEAD_RADIUS - p_cfg.TBM.TRACK_SPACING / 2
    ) / p_cfg.TBM.TRACK_SPACING
    n_c_tot = int(round(n_c_tot, 0)) + 1

    cutter_positions = (
        np.cumsum(np.full((n_c_tot), p_cfg.TBM.TRACK_SPACING))
        - p_cfg.TBM.TRACK_SPACING / 2
    )
    cutter_pathlenghts = cutter_positions * 2 * np.pi  # [m]

    reward_fn = Reward(n_c_tot, **p_cfg.REWARD)

    env = CustomEnv(
        n_c_tot,
        p_cfg.TBM.LIFE,
        p_cfg.TBM.MAX_STROKES,
        p_cfg.TBM.STROKE_LENGTH,
        cutter_pathlenghts,
        p_cfg.TBM.CUTTERHEAD_RADIUS,
        reward_fn,
    )
    if p_cfg.EXP.CHECK_ENV:
        check_env(env)

    callbacks_cfg = dict(
        MAX_STROKES=p_cfg.TBM.MAX_STROKES,
        CHECKPOINT_INTERVAL=p_cfg.EXP.CHECKPOINT_INTERVAL,
        PLOT_PROGRESS=p_cfg.EXP.PLOT_PROGRESS,
        DETERMINISTIC=p_cfg.EXP.DETERMINISTIC,
        MAX_NO_IMPROVEMENT_EVALS=p_cfg.OPT.MAX_NO_IMPROVEMENT_EVALS,
        N_EVAL_EPISODES_OPTIMIZATION=p_cfg.OPT.N_EVAL_EPISODES_OPTIMIZATION,
        N_EVAL_EPISODES_REWARD=p_cfg.OPT.N_EVAL_EPISODES_REWARD,
        N_EVAL_EPISODES_TRAINING=p_cfg.TRAIN.N_EVAL_EPISODES_TRAINING,
    )

    optim = Optimization(
        n_c_tot,
        env,
        agent_name,
        rich_console,
        p_cfg.EXP.STUDY,
        p_cfg.EXP.EPISODES,
        p_cfg.EXP.MODE,
        p_cfg.EXP.DEBUG,
        p_cfg.TBM.MAX_STROKES,
        p_cfg.OPT.DEFAULT_TRIAL,
        callbacks_cfg,
    )

    ###############################################################################
    # RUN ONE OF THE MODES
    ###############################################################################
    match p_cfg.EXP.MODE:
        case "optimization":
            run_optimization(
                STUDY=p_cfg.EXP.STUDY,
                N_SINGLE_RUN_OPTUNA_TRIALS=p_cfg.OPT.N_SINGLE_RUN_OPTUNA_TRIALS,
                N_PARALLELL_PROCESSES=p_cfg.OPT.N_PARALLELL_PROCESSES,
                N_CORES_PARALLELL=p_cfg.OPT.N_CORES_PARALLELL,
                optim=optim,
            )

        case "training":
            for _ in range(p_cfg.TRAIN.N_DUPLICATES):
                run_training(
                    rcon=rich_console,
                    STUDY=p_cfg.EXP.STUDY,
                    LOAD_PARAMS_FROM_STUDY=p_cfg.TRAIN.LOAD_PARAMS_FROM_STUDY,
                    best_agent_params=p_cfg.agent.agent_params,
                    agent_name=agent_name,
                    n_c_tot=n_c_tot,
                    MAX_STROKES=p_cfg.TBM.MAX_STROKES,
                    STROKE_LENGTH=p_cfg.TBM.STROKE_LENGTH,
                    env=env,
                    optim=optim,
                )

        case "execution":
            run_execution(
                rcon=rich_console,
                DETERMINISTIC=p_cfg.EXP.DETERMINISTIC,
                EXECUTION_MODEL=p_cfg.EXECUTE.EXECUTION_MODEL,
                NUM_TEST_EPISODES=p_cfg.EXECUTE.NUM_TEST_EPISODES,
                VISUALIZE_STATE_ACTION_PLOT=p_cfg.EXECUTE.VISUALIZE_STATE_ACTION_PLOT,
                env=env,
                n_c_tot=n_c_tot,
            )

        case _:
            raise ValueError(f"{p_cfg.EXP.MODE} is not an option")


if __name__ == "__main__":
    main()
