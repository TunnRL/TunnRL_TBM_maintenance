"""
Towards optimized TBM cutter changing policies with reinforcement learning
G.H. Erharter, T.F. Hansen
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Main code that either runs a hyperparameter optimization study with OPTUNA or a
"main run" of just one study with fixed hyperparameters

Created on Sat Oct 30 12:46:42 2021
code contributors: Georg H. Erharter, Tom F. Hansen

"""

import warnings
from pathlib import Path

import gym
import hydra
import numpy as np
import optuna
import torch.nn as nn  # used in evaluation of yaml file
import yaml
from joblib import Parallel, delayed
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.traceback import install
from stable_baselines3.common.env_checker import check_env

from XX_experiment_factory import Optimization, load_best_model
from XX_hyperparams import Hyperparameters
from XX_plotting import Plotter
from XX_TBM_environment import CustomEnv, Reward


install()


def run_optimization(
    STUDY: str,
    N_SINGLE_RUN_OPTUNA_TRIALS: int,
    N_PARALLELL_PROCESSES: int,
    N_CORES_PARALLELL: int,
    optim: Optimization,
) -> None:
    """Optimize the hyperparameters for SB3 algorithm using Optuna.

    Args:
        STUDY (str): names the algorithm to optimize and the saved optimizing db object
        N_SINGLE_RUN_OPTUNA_TRIALS (int): n optuna trials
    """
    print(
        f"{N_SINGLE_RUN_OPTUNA_TRIALS * N_PARALLELL_PROCESSES} optuna trials are processed in {N_PARALLELL_PROCESSES} processes.\n"
    )

    db_path = f"results/{STUDY}.db"
    db_file = f"sqlite:///{db_path}"
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY,
        storage=db_file,
        load_if_exists=True,
        sampler=sampler,
    )
    optim.optimize(N_SINGLE_RUN_OPTUNA_TRIALS)

    # Parallel(n_jobs=N_CORES_PARALLELL, verbose=10, backend="loky")(
    #     delayed(optim.optimize)(N_SINGLE_RUN_OPTUNA_TRIALS) for _ in range(N_PARALLELL_PROCESSES))


def run_training(
    STUDY: str,
    LOAD_PARAMS_FROM_STUDY: bool,
    MAX_STROKES: int,
    agent_name: str,
    n_c_tot: int,
    env: gym.Env,
    optim: Optimization,
) -> None:
    """Run a full training of an RL agent.

    Args:
        STUDY (str): study description
        LOAD_PARAMS_FROM_STUDY (bool): load best params from study
        MAX_STROKES (int): number of episodes (one tunnel excavation)
        agent_name (str): RL algorithm name
        n_c_tot (int): number of cutters on cutterhead
        hparams (Hyperparameters): object with parameter interpreting functionality
        env (gym.Env): TBM environment for the agent to act with
        optim (Optimization): experiment object containing all functionality for training
    """
    print(f"New {agent_name} training run with optimized parameters started.")
    print(f" - total number of cutters: {n_c_tot}\n")

    hparams = Hyperparameters()

    if LOAD_PARAMS_FROM_STUDY:
        print(f"loading parameters from the study object: {STUDY}")
        db_path = f"results/{STUDY}.db"
        db_file = f"sqlite:///{db_path}"
        study = optuna.load_study(study_name=STUDY, storage=db_file)
        best_trial = study.best_trial
        print(f"Highest reward from best trial: {best_trial.value}")

        best_params_dict = hparams.remake_params_dict(
            algorithm=agent_name,
            raw_params_dict=best_trial.params,
            env=env,
            n_actions=n_c_tot * n_c_tot,
        )
    else:
        print("loading parameters from yaml file...")
        with open(f"results/algorithm_parameters/{agent_name}.yaml") as file:
            best_params_dict: dict = yaml.safe_load(file)
        with open(f"results/algorithm_parameters/{agent_name}_sub.yaml") as file:
            sub_params: dict = yaml.safe_load(file)

        best_params_dict["learning_rate"] = hparams.parse_learning_rate(
            best_params_dict["learning_rate"]
        )
        best_params_dict["policy_kwargs"] = eval(best_params_dict["policy_kwargs"])
        best_params_dict.update(dict(env=env, n_steps=MAX_STROKES))

    optim.train_agent(best_params_dict, sub_params)


def run_execution(
    DETERMINISTIC: bool,
    EXECUTION_MODEL: str,
    NUM_TEST_EPISODES: int,
    env: gym.Env,
    n_c_tot: int,
) -> None:
    """Runs a number of episodes for a trained TBM RL agent

    Args:
        DETERMINISTIC (bool): running on a constant environment
        EXECUTION_MODEL (str): the trained model to load
        NUM_TEST_EPISODES (int): number of episodes to run
        env (gym.Env): TBM RL environment for the agent to act with
        n_c_tot (int): number of cutters on the cutterhead
    """
    agent_name = EXECUTION_MODEL.split("_")[0]
    agent = load_best_model(
        agent_name, main_dir="checkpoints", agent_dir=EXECUTION_MODEL
    )

    # test agent throughout multiple episodes
    for test_ep_num in range(NUM_TEST_EPISODES):
        print(f"Episode num: {test_ep_num}")
        state = env.reset()  # reset new environment
        terminal = False  # reset terminal flag

        actions: list[NDArray] = []  # collect actions per episode
        states: list[NDArray] = [state]  # collect states per episode
        rewards = []  # collect rewards per episode
        broken_cutters = []  # collect number of broken cutters per stroke
        replaced_cutters = []  # collect n of replaced cutters per stroke
        moved_cutters = []  # collect n of moved_cutters cutters per stroke

        # one episode loop
        i = 0
        while not terminal:
            print(f"Stroke (step) num: {i}")
            # collect number of broken cutters in curr. state
            broken_cutters.append(len(np.where(state == 0)[0]))
            # agent takes an action -> tells which cutters to replace
            action = agent.predict(state, deterministic=DETERMINISTIC)[0]
            # environment gives new state signal, terminal flag and reward
            state, reward, terminal, info = env.step(action)
            # collect actions, states and rewards for later analyses
            actions.append(action)
            states.append(state)
            rewards.append(reward)
            replaced_cutters.append(env.replaced_cutters)
            moved_cutters.append(env.moved_cutters)
            i += 1

        Plotter.state_action_plot(
            states,
            actions,
            n_strokes=300,
            n_c_tot=n_c_tot,
            show=False,
            savepath=f"checkpoints/_sample/{EXECUTION_MODEL}{test_ep_num}_state_action.svg",
        )
        Plotter.environment_parameter_plot(
            test_ep_num,
            env,
            show=False,
            savepath=f"checkpoints/_sample/{EXECUTION_MODEL}{test_ep_num}_episode.svg",
        )
        Plotter.sample_ep_plot(
            states,
            actions,
            rewards,
            ep=test_ep_num,
            savepath=f"checkpoints/_sample/{EXECUTION_MODEL}{test_ep_num}_sample.svg",
            replaced_cutters=replaced_cutters,
            moved_cutters=moved_cutters,
        )


@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(cfg: DictConfig) -> None:

    ###############################################################################
    # WARNINGS AND ERROR CHECKING INPUT VARIABLES
    ###############################################################################
    if cfg.EXP.DEBUG:
        cfg.EXP.EPISODES = 20
        cfg.EXP.CHECKPOINT_INTERVAL = 6
        cfg.OPT.N_PARALLELL_PROCESSES = 2
        cfg.OPT.N_CORES_PARALLELL = 2

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    if cfg.OPT.DEFAULT_TRIAL:
        warnings.warn("Optimization runs are started with default parameter values")

    assert (
        cfg.OPT.N_CORES_PARALLELL >= cfg.OPT.N_PARALLELL_PROCESSES
        or cfg.OPT.N_CORES_PARALLELL == -1
    ), "Num cores must be >= num parallell processes."
    if cfg.OPT.N_PARALLELL_PROCESSES > 1 and (
        cfg.EXP.MODE == "training" or cfg.EXP.MODE == "execution"
    ):
        warnings.warn("No parallellization in training and execution mode")

    if cfg.TRAIN.LOAD_PARAMS_FROM_STUDY is True and cfg.EXP.MODE == "training":
        assert Path(
            f"./results/{cfg.EXP.STUDY}.db"
        ).exists(), "The study object does not exist"

    if cfg.TRAIN.LOAD_PARAMS_FROM_STUDY is False and cfg.EXP.MODE == "training":
        assert Path(
            f'results/algorithm_parameters/{cfg.EXP.STUDY.split("_")[0]}.yaml'
        ).exists(), "a yaml file with pararameter does not exist."

    ###############################################################################
    # COMPUTED/DERIVED VARIABLES AND INSTANTIATIONS
    ###############################################################################

    rich_console = Console()
    rich_console.print(OmegaConf.to_yaml(cfg))

    n_c_tot = (
        cfg.TBM.CUTTERHEAD_RADIUS - cfg.TBM.TRACK_SPACING / 2
    ) / cfg.TBM.TRACK_SPACING
    n_c_tot = int(round(n_c_tot, 0)) + 1

    cutter_positions = (
        np.cumsum(np.full((n_c_tot), cfg.TBM.TRACK_SPACING)) - cfg.TBM.TRACK_SPACING / 2
    )
    cutter_pathlenghts = cutter_positions * 2 * np.pi  # [m]

    reward_fn = Reward(n_c_tot, **cfg.REWARD)

    env = CustomEnv(
        n_c_tot,
        cfg.TBM.LIFE,
        cfg.TBM.MAX_STROKES,
        cfg.TBM.STROKE_LENGTH,
        cutter_pathlenghts,
        cfg.TBM.CUTTERHEAD_RADIUS,
        reward_fn,
    )
    if cfg.EXP.CHECK_ENV:
        check_env(env)

    agent_name = cfg.EXP.STUDY.split("_")[0]
    assert agent_name in [
        "PPO",
        "A2C",
        "DDPG",
        "SAC",
        "TD3",
    ], f"{agent_name} is not a valid agent."

    callbacks_cfg = dict(
        MAX_STROKES=cfg.TBM.MAX_STROKES,
        CHECKPOINT_INTERVAL=cfg.EXP.CHECKPOINT_INTERVAL,
        PLOT_PROGRESS=cfg.EXP.PLOT_PROGRESS,
        DETERMINISTIC=cfg.EXP.DETERMINISTIC,
        MAX_NO_IMPROVEMENT_EVALS=cfg.OPT.MAX_NO_IMPROVEMENT_EVALS,
        N_EVAL_EPISODES_OPTIMIZATION=cfg.OPT.N_EVAL_EPISODES_OPTIMIZATION,
        N_EVAL_EPISODES_REWARD=cfg.OPT.N_EVAL_EPISODES_REWARD,
        N_EVAL_EPISODES_TRAINING=cfg.TRAIN.N_EVAL_EPISODES_TRAINING,
    )

    optim = Optimization(
        n_c_tot,
        env,
        agent_name,
        rich_console,
        cfg.EXP.STUDY,
        cfg.EXP.EPISODES,
        cfg.EXP.MODE,
        cfg.EXP.DEBUG,
        cfg.TBM.MAX_STROKES,
        cfg.OPT.DEFAULT_TRIAL,
        callbacks_cfg,
    )

    ###############################################################################
    # RUN ONE OF THE MODES
    ###############################################################################
    match cfg.EXP.MODE:
        case "optimization":
            run_optimization(
                cfg.EXP.STUDY,
                cfg.OPT.N_SINGLE_RUN_OPTUNA_TRIALS,
                cfg.OPT.N_PARALLELL_PROCESSES,
                cfg.OPT.N_CORES_PARALLELL,
                optim,
            )

        case "training":
            run_training(
                cfg.EXP.STUDY,
                cfg.TRAIN.LOAD_PARAMS_FROM_STUDY,
                cfg.TBM.MAX_STROKES,
                agent_name,
                n_c_tot,
                env,
                optim,
            )

        case "execution":
            run_execution(
                cfg.EXP.DETERMINISTIC,
                cfg.EXECUTE.EXECUTION_MODEL,
                cfg.EXECUTE.NUM_TEST_EPISODES,
                env,
                n_c_tot,
            )

        case _:
            raise ValueError(f"{cfg.EXP.MODE} is not an option")


if __name__ == "__main__":
    main()
