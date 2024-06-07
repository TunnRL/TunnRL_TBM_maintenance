"""Functionality to train, optimize, and execute an RL model."""

import gymnasium as gym
import numpy as np
import optuna
from joblib import Parallel, delayed
from numpy.typing import NDArray
from rich.console import Console

from tunnrl_tbm_maintenance.experiment_factory import (
    ExperimentAnalysis,
    Optimization,
    load_best_model,
)
from tunnrl_tbm_maintenance.hyperparameters import Hyperparameters
from tunnrl_tbm_maintenance.plotting import Plotter


def run_execution(
    rcon: Console,
    DETERMINISTIC: bool,
    EXECUTION_MODEL: str,
    NUM_TEST_EPISODES: int,
    VISUALIZE_STATE_ACTION_PLOT: bool,
    env: gym.Env,
    n_c_tot: int,
) -> None:
    """Runs a number of episodes for a trained TBM RL agent.

    Plots several plots showing the decision making in the process.

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
        rcon.print(f"Episode num: {test_ep_num}")
        state = env.reset()  # reset new environment
        terminal = False  # reset terminal flag

        actions: list[NDArray] = []  # collect actions per episode
        states: list[NDArray] = [
            state
        ]  # convert state to NumPy array and collect states per episode
        rewards = []  # collect rewards per episode
        broken_cutters = []  # collect number of broken cutters per stroke
        replaced_cutters = []  # collect n of replaced cutters per stroke
        moved_cutters = []  # collect n of moved_cutters cutters per stroke

        # one episode loop
        i = 0
        while not terminal:
            rcon.print(f"Stroke (step) num: {i}")
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

        if VISUALIZE_STATE_ACTION_PLOT:
            rcon.print("Plotting state plots for each episode...")
            Plotter.state_action_plot(
                states,
                actions,
                n_strokes=300,
                rewards=rewards,
                n_c_tot=n_c_tot,
                show=False,
                savepath=f"checkpoints/_sample/{EXECUTION_MODEL}{test_ep_num}_state_action.svg",
            )
        with rcon.status("Plotting episode plots..."):
            Plotter.environment_parameter_plot(
                test_ep_num,
                env,
                show=False,
                savepath=f"checkpoints/_sample/{EXECUTION_MODEL}_episode_{test_ep_num}.svg",
            )
            Plotter.sample_ep_plot(
                states,
                actions,
                rewards,
                replaced_cutters=replaced_cutters,
                moved_cutters=moved_cutters,
                n_cutters=n_c_tot,
                show=False,
                savepath=f"checkpoints/_sample/{EXECUTION_MODEL}_sample_{test_ep_num}.svg",
            )

    with rcon.status("Plotting action analyzing scripts for the last episode..."):
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
            df_reduced,
            savepath=f"checkpoints/_sample/{EXECUTION_MODEL}_TSNE_scatter.svg",
        )


def run_training(
    rcon: Console,
    STUDY: str,
    LOAD_PARAMS_FROM_STUDY: bool,
    best_agent_params: dict,
    agent_name: str,
    n_c_tot: int,
    MAX_STROKES: int,
    STROKE_LENGTH: float,
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
        optim (Optimization): experiment object containing all functionality for
        training

    Usecase (to run 5 trials of best parameters and 5 of default parameters):
        pythonr src/A_main_hydra.py --multirun agent=ppo_best, ppo_default,
        TRAIN.N_DUPLICATES=5
    """
    rcon.print(f"New {agent_name} training run with optimized parameters started.")
    rcon.print(f" - total number of cutters: {n_c_tot}")
    rcon.print(
        f" - tunnel length in each episode, in meters: {MAX_STROKES*STROKE_LENGTH}"
    )
    rcon.print(f" - stroke length: {STROKE_LENGTH}")
    rcon.print(f" - number of strokes: {MAX_STROKES}\n")

    hparams = Hyperparameters()

    if LOAD_PARAMS_FROM_STUDY:
        rcon.print(f"loading parameters from the study object: {STUDY}")
        db_path = f"results/{STUDY}.db"
        db_file = f"sqlite:///{db_path}"
        study = optuna.load_study(study_name=STUDY, storage=db_file)
        best_trial = study.best_trial
        best_agent_params = best_trial.params
        rcon.print(f"Highest reward from best trial: {best_trial.value}")

    else:
        rcon.print(
            "loading parameters from best_params or default_params in yaml file..."
        )

    best_params_dict = hparams.remake_params_dict(
        algorithm=agent_name,
        raw_params_dict=best_agent_params,
        env=env,
        n_actions=n_c_tot * n_c_tot,
    )
    optim.train_agent(best_params_dict, reporting_parameters=best_agent_params)


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
        N_PARALLELL_PROCESSES (int):
        N_CORES_PARALLELL (int):
        optim (Optimization):
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
            f"{N_SINGLE_RUN_OPTUNA_TRIALS * N_PARALLELL_PROCESSES} \
                optuna trials are processed in {N_PARALLELL_PROCESSES} processes.\n"
        )
        Parallel(n_jobs=N_CORES_PARALLELL, verbose=10, backend="loky")(
            delayed(optim.optimize)(study, N_SINGLE_RUN_OPTUNA_TRIALS)
            for _ in range(N_PARALLELL_PROCESSES)
        )
