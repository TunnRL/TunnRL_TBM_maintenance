"""
Towards optimized TBM cutter changing policies with reinforcement learning
G.H. Erharter, T.F. Hansen
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Main code that either runs a hyperparameter optimization study with OPTUNA or a
"main run" of just one study with fixed hyperparameters

Created on Sat Oct 30 12:46:42 2021
code contributors: Georg H. Erharter, Tom F. Hansen

"""

import multiprocessing as mp
import warnings
from pathlib import Path

import gym
import hydra
import numpy as np
import optuna
from joblib import Parallel, delayed
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.traceback import install
from stable_baselines3.common.env_checker import check_env

from XX_config_schemas import Config
from XX_experiment_factory import ExperimentAnalysis, Optimization, load_best_model
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
    if N_PARALLELL_PROCESSES <= 1:
        optim.optimize(study, N_SINGLE_RUN_OPTUNA_TRIALS)
    else:
        print(
            f"{N_SINGLE_RUN_OPTUNA_TRIALS * N_PARALLELL_PROCESSES} optuna trials are processed in {N_PARALLELL_PROCESSES} processes.\n"
        )
        Parallel(n_jobs=N_CORES_PARALLELL, verbose=10, backend="loky")(
            delayed(optim.optimize)(study, N_SINGLE_RUN_OPTUNA_TRIALS)
            for _ in range(N_PARALLELL_PROCESSES)
        )
        # pool = mp.Pool(4)
        # [pool.apply(optim.optimize, args=(study, N_SINGLE_RUN_OPTUNA_TRIALS)) for _ in range(4)]


def run_training(
    STUDY: str,
    LOAD_PARAMS_FROM_STUDY: bool,
    best_agent_params: dict,
    MAX_STROKES: int,
    agent_name: str,
    n_c_tot: int,
    env: gym.Env,
    optim: Optimization,
) -> None:
    """Run a full training run of an RL agent.

    Args:
        STUDY (str): study description
        LOAD_PARAMS_FROM_STUDY (bool): load best params from study
        MAX_STROKES (int): number of episodes (one tunnel excavation)
        agent_name (str): RL algorithm name
        n_c_tot (int): number of cutters on cutterhead
        hparams (Hyperparameters): object with parameter interpreting functionality
        env (gym.Env): TBM environment for the agent to act with
        optim (Optimization): experiment object containing all functionality for training

    Usecase (to run 5 trials of best parameters and 5 of default parameters):
        pythonr src/A_main_hydra.py --multirun agent=ppo_best, ppo_default, TRAIN.N_DUPLICATES=5
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
        best_agent_params = best_trial.params
        print(f"Highest reward from best trial: {best_trial.value}")

    else:
        print("loading parameters from best_params or default_params in yaml file...")

    best_params_dict = hparams.remake_params_dict(
        algorithm=agent_name,
        raw_params_dict=best_agent_params,
        env=env,
        n_actions=n_c_tot * n_c_tot,
    )
    optim.train_agent(best_params_dict, reporting_parameters=best_agent_params)


def run_execution(
    DETERMINISTIC: bool,
    EXECUTION_MODEL: str,
    NUM_TEST_EPISODES: int,
    VISUALIZE_EPISODES: bool,
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
    all_actions = []
    all_states = []
    all_rewards = []
    all_broken_cutters = []
    all_replaced_cutters = []
    all_moved_cutters = []

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

        all_actions.append(actions)
        all_states.append(states[:-1])
        all_rewards.append(rewards)
        all_broken_cutters.append(broken_cutters)
        all_replaced_cutters.append([len(c) for c in replaced_cutters])
        all_moved_cutters.append([len(c) for c in moved_cutters])

        if VISUALIZE_EPISODES:
            Plotter.state_action_plot(
                states,
                actions,
                n_strokes=300,
                rewards=rewards,
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
                replaced_cutters=replaced_cutters,
                moved_cutters=moved_cutters,
                n_cutters=n_c_tot,
                show=False,
                savepath=f"checkpoints/_sample/{EXECUTION_MODEL}{test_ep_num}_sample.svg",
            )

    df_reduced = ExperimentAnalysis.dimensionality_reduction(
        all_actions,
        all_states,
        all_rewards,
        all_broken_cutters,
        all_replaced_cutters,
        all_moved_cutters,
        perplexity=200,
    )

    Plotter.action_analysis_scatter_plotly(
        df_reduced,
        savepath=f"checkpoints/_sample/{EXECUTION_MODEL}_TSNE_scatter_plotly.html",
    )

    Plotter.action_analysis_scatter(
        df_reduced, savepath=f"checkpoints/_sample/{EXECUTION_MODEL}_TSNE_scatter.svg"
    )


@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(cfg: Config) -> None:

    ###############################################################################
    # WARNINGS AND ERROR CHECKING INPUT VARIABLES
    ###############################################################################
    if cfg.EXP.DEBUG:
        cfg.EXP.EPISODES = 20
        cfg.EXP.CHECKPOINT_INTERVAL = 6
        cfg.OPT.N_PARALLELL_PROCESSES = 1
        cfg.OPT.N_CORES_PARALLELL = 1

    OmegaConf.to_object(cfg)  # runs checks of inputs by pydantic, types and validation

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

    assert (
        cfg.agent.NAME == (cfg.EXP.STUDY).split("_")[0]
    ), "Agent name and study name must be similar"

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
        "PPO-LSTM",
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
                STUDY=cfg.EXP.STUDY,
                N_SINGLE_RUN_OPTUNA_TRIALS=cfg.OPT.N_SINGLE_RUN_OPTUNA_TRIALS,
                N_PARALLELL_PROCESSES=cfg.OPT.N_PARALLELL_PROCESSES,
                N_CORES_PARALLELL=cfg.OPT.N_CORES_PARALLELL,
                optim=optim,
            )

        case "training":
            for _ in range(cfg.TRAIN.N_DUPLICATES):
                run_training(
                    STUDY=cfg.EXP.STUDY,
                    LOAD_PARAMS_FROM_STUDY=cfg.TRAIN.LOAD_PARAMS_FROM_STUDY,
                    best_agent_params=cfg.agent.agent_params,
                    MAX_STROKES=cfg.TBM.MAX_STROKES,
                    agent_name=agent_name,
                    n_c_tot=n_c_tot,
                    env=env,
                    optim=optim,
                )

        case "execution":
            run_execution(
                cfg.EXP.DETERMINISTIC,
                cfg.EXECUTE.EXECUTION_MODEL,
                cfg.EXECUTE.NUM_TEST_EPISODES,
                cfg.EXECUTE.VISUALIZE_EPISODES,
                env,
                n_c_tot,
            )

        case _:
            raise ValueError(f"{cfg.EXP.MODE} is not an option")


if __name__ == "__main__":
    main()
