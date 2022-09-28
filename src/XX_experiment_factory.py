# -*- coding: utf-8 -*-
"""
Towards optimized TBM cutter changing policies with reinforcement learning
G.H. Erharter, T.F. Hansen
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Custom library that contains different classes for:
- Parameter optimization
- Training an agent
- Callbacks used in training process
- some utility functionality (loading trained agents etc.)

code contributors: Georg H. Erharter, Tom F. Hansen
"""

import uuid
import warnings
from dataclasses import dataclass
from pprint import pformat
from typing import Any

import gym
import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from hydra.core.hydra_config import HydraConfig
from rich.console import Console
from rich.traceback import install
from sklearn.manifold import TSNE
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.evaluation import evaluate_policy

from XX_hyperparams import Hyperparameters
from XX_plotting import Plotter


install()

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


@dataclass
class Optimization:
    """Functionality to train (optimize the reward of an agent)
    and hyperparameter tuning of agent parameters using Optuna.

    Args:
        n_c_tot (int): Number of cutters
        environment (gym.Env): TBM environment with state, actions, reward
        AGENT_NAME (str): name of RL algorithm, eg. PPO, DDPG ...
        STUDY (str): Optuna study object
        EPISODES (int): Number of RL episodes in the optimization
        MODE (str): the process mode of the optimization: training, optimization
        CHECKPOINT_INTERVAL (int): Frequency of evaluations in episodes
        DEFAULT_TRIAL (bool): If an optimization first should run with default params
        cb_cfg (dict): callback config
    """

    n_c_tot: int
    environment: gym.Env
    AGENT_NAME: str
    rich_console: Console
    STUDY: str
    EPISODES: int
    MODE: str
    DEBUG: bool
    MAX_STROKES: int
    DEFAULT_TRIAL: bool
    cb_cfg: dict

    def __post_init__(self) -> None:
        """Functions are called automatically after initializing the object."""
        self.CHECKPOINT_INTERVAL = self.cb_cfg["CHECKPOINT_INTERVAL"]
        self.n_actions = self.n_c_tot * self.n_c_tot
        # n steps, eg. 1000 steps x 100 checkpoint_interval = every 100 000 steps
        self.checkpoint_frequency = self.cb_cfg["MAX_STROKES"] * self.CHECKPOINT_INTERVAL
        self.hparams = Hyperparameters()
        self.agent_dir: str = ""

    def objective(self, trial: optuna.trial.Trial) -> float | list[float]:
        """Objective function that drives the optimization of parameter values
        for the RL-agent.

        Train an RL-agent and returns a reward evaluated on the last xxx episodes.
        """

        if self.DEFAULT_TRIAL:
            parameters = {"policy": "MlpPolicy", "env": self.environment}
            self.DEFAULT_TRIAL = False
        else:
            parameters, sub_parameters = self.hparams.suggest_hyperparameters(
                trial,
                self.AGENT_NAME,
                self.environment,
                steps_episode=self.MAX_STROKES,
                num_actions=self.n_actions,
            )

        agent = self._setup_agent(self.AGENT_NAME, parameters)

        self.agent_dir = f"{self.AGENT_NAME}_{uuid.uuid4()}"
        self.hydra_dir = get_hydra_experiment_tag(HydraConfig.get().run.dir)
        callbacks, sb3_logger = self._setup_callbacks_and_logger(parameters)

        if sb3_logger is not None:
            agent.set_logger(sb3_logger)

        agent.learn(total_timesteps=self.EPISODES * self.MAX_STROKES, callback=callbacks)
        del agent

        print("Load agent and evaluate on 10 last episodes...")
        agent = load_best_model(
            self.AGENT_NAME, main_dir="optimization", agent_dir=self.agent_dir
        )

        final_reward = evaluate_policy(
            agent,
            self.environment,
            n_eval_episodes=self.cb_cfg["N_EVAL_EPISODES_REWARD"],
            deterministic=self.cb_cfg["DETERMINISTIC"],
            warn=False,
        )[0]

        print(f"Agent in dir: {self.agent_dir} has a reward of: {final_reward}\n")

        # logging to mlflow
        experiment_info = dict(
            exp_logdir=self.agent_dir,
            exp_mode=self.MODE,
            exp_study=self.STUDY,
            exp_agent=self.AGENT_NAME,
            exp_hydra_dir=self.hydra_dir,
        )

        mlflow_log_experiment(
            experiment_info=experiment_info,
            main_dir="optimization",
            agent_dir=self.agent_dir,
            parameters=parameters,
            sub_parameters=sub_parameters,
        )

        return final_reward

    def optimize(self, study: optuna.study.Study, n_trials: int) -> None:
        """Optimize-function to be called in parallell process.

        Args:
            n_trials (int): Number of trials in each parallell process.

        Returns:
            None
        """
        # db_path = f"results/{self.STUDY}.db"
        # db_file = f"sqlite:///{db_path}"
        # study = optuna.load_study(study_name=self.STUDY, storage=db_file)

        try:
            study.optimize(
                self.objective,
                n_trials=n_trials,
                catch=(ValueError,),
            )
        except KeyboardInterrupt:
            print("Number of finished trials: ", len(study.trials))
            print("Best trial:")
            trial = study.best_trial
            print("  Value: ", trial.value)
            print("  Params: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

        finally:
            print("Saving best parameters to a yaml_file")
            with open(
                f"results/{self.STUDY}_best_params_{study.best_value: .2f}.yaml", "w"
            ) as file:
                yaml.dump(study.best_params, file)

    def train_agent(self, best_parameters: dict, reporting_parameters: dict) -> None:
        """Train agent with best parameters from an optimization study.

        Args:
            best_parameters (dict): parameters for SB3 agent
        """

        agent = self._setup_agent(self.AGENT_NAME, best_parameters)

        self.agent_dir = f"{self.AGENT_NAME}_{uuid.uuid4()}"
        self.hydra_dir = get_hydra_experiment_tag(HydraConfig.get().run.dir)
        callbacks, SB3_logger = self._setup_callbacks_and_logger(best_parameters)
        if SB3_logger is not None:
            agent.set_logger(SB3_logger)

        agent.learn(total_timesteps=self.EPISODES * self.MAX_STROKES, callback=callbacks)

        # logging to mlflow
        experiment_info = dict(
            exp_logdir=self.agent_dir,
            exp_mode=self.MODE,
            exp_study=self.STUDY,
            exp_agent=self.AGENT_NAME,
            exp_hydra_dir=self.hydra_dir,
        )
        mlflow_log_experiment(
            experiment_info=experiment_info,
            main_dir="checkpoints",
            agent_dir=self.agent_dir,
            parameters=reporting_parameters,
        )

    def _setup_agent(self, agent_name: str, parameters: dict) -> BaseAlgorithm:
        """Instantiating and returning an SB3 agent.

        Args:
            agent_name (str): algorithm name (PPO, DDPG etc.)
            parameters (dict): input parameters to algorithm

        Returns:
            BaseAlgorithm: instantiated SB3 agent
        """
        match agent_name:
            case "PPO":
                agent = PPO(**parameters)
            case "SAC":
                agent = SAC(**parameters)
            case "A2C":
                agent = A2C(**parameters)
            case "DDPG":
                agent = DDPG(**parameters)
            case "TD3":
                agent = TD3(**parameters)
            case _:
                raise NotImplementedError(f"{self.AGENT_NAME} not implemented")
        return agent

    def _setup_callbacks_and_logger(self, parameters: dict) -> tuple[list, Any]:
        """Defining callbacks and logger used in training and optimizing an RL agent.
        Different setups for different training/optimization modes.
        TODO: this is a function which takes second most time. Look for improvements.

        Args:
            parameters (dict): parameters for agent in experiment

        Returns:
            tuple[list, Any, str]: callbacklist, SB3-logger
        """
        agent_dir = self.agent_dir
        cb_list = []
        sb3_logger = None  # for debugging, ie. don't make dir etc.
        # callback values

        if self.MODE == "optimization":
            stop_train_cb = (
                StopTrainingOnNoModelImprovement(  # kicked off by EvalCallback
                    max_no_improvement_evals=self.cb_cfg["MAX_NO_IMPROVEMENT_EVALS"],
                    min_evals=1,
                    verbose=1,
                )
            )

            cb_list.append(
                EvalCallback(  # saves best model
                    self.environment,
                    best_model_save_path=f"optimization/{agent_dir}",
                    log_path=f"optimization/{agent_dir}",
                    deterministic=self.cb_cfg["DETERMINISTIC"],
                    n_eval_episodes=self.cb_cfg["N_EVAL_EPISODES_OPTIMIZATION"],
                    eval_freq=self.checkpoint_frequency,
                    callback_after_eval=stop_train_cb,
                    verbose=1,
                    warn=False,
                )
            )
            cb_list.append(
                TrainingProgressCallback(
                    check_freq=self.checkpoint_frequency,
                    save_path=f"optimization/{agent_dir}",
                    name_prefix=f"{self.AGENT_NAME}",
                    MAX_STROKES=self.MAX_STROKES,
                    MODE=self.MODE,
                    rich_console=self.rich_console,
                    PLOT_PROGRESS=self.cb_cfg["PLOT_PROGRESS"],
                )
            )
            cb_list.append(
                PrintExperimentInfoCallback(
                    self.MODE,
                    agent_dir,
                    self.hydra_dir,
                    parameters,
                    self.EPISODES,
                    self.CHECKPOINT_INTERVAL,
                    self.rich_console,
                )
            )
            sb3_logger = logger.configure(
                f"optimization/{agent_dir}", ["csv", "tensorboard"]
            )

        elif self.MODE == "training":
            cb_list.append(
                CheckpointCallback(  # save model every checkpoint interval
                    save_freq=self.checkpoint_frequency,
                    save_path=f"checkpoints/{agent_dir}",
                    name_prefix=f"{self.AGENT_NAME}",
                    verbose=1,
                )
            )
            cb_list.append(
                EvalCallback(  # saves best model
                    self.environment,
                    best_model_save_path=f"checkpoints/{agent_dir}",
                    log_path=f"checkpoints/{agent_dir}",
                    deterministic=self.cb_cfg["DETERMINISTIC"],
                    n_eval_episodes=self.cb_cfg["N_EVAL_EPISODES_TRAINING"],
                    eval_freq=self.checkpoint_frequency,
                    verbose=1,
                    warn=False,
                )
            )
            cb_list.append(
                TrainingProgressCallback(
                    check_freq=self.checkpoint_frequency,
                    save_path=f"checkpoints/{agent_dir}",
                    name_prefix=f"{self.AGENT_NAME}",
                    MAX_STROKES=self.MAX_STROKES,
                    MODE=self.MODE,
                    rich_console=self.rich_console,
                    PLOT_PROGRESS=self.cb_cfg["PLOT_PROGRESS"],
                )
            )
            cb_list.append(
                PrintExperimentInfoCallback(
                    self.MODE,
                    agent_dir,
                    self.hydra_dir,
                    parameters,
                    self.EPISODES,
                    self.CHECKPOINT_INTERVAL,
                    self.rich_console,
                )
            )
            sb3_logger = logger.configure(
                f"checkpoints/{agent_dir}", ["csv", "tensorboard"]
            )
        else:
            raise ValueError("not a valid mode")

        cb_list = CallbackList(cb_list)

        return cb_list, sb3_logger


def get_hydra_experiment_tag(hydra_path: str) -> str:
    """Returns hydra experiment tag."""
    # exp_parts = hydra_path.split("/")
    # hydra_tag = exp_parts[-2] + "_" + exp_parts[-1]
    return hydra_path  # hydra_tag


def load_best_model(agent_name: str, main_dir: str, agent_dir: str) -> BaseAlgorithm:
    """Load best model from a directory.

    Args:
        agent_name (str): name of RL-architecture (PPO, DDPG ...)
        main_dir (str): main directory for loading, eg. optimization or checkpoints
        agent_dir (str): name of agent directory in main dir

    Returns:
        Instantiated RL-object
    """
    path = f"{main_dir}/{agent_dir}/best_model"
    agent = None

    try:
        if agent_name == "PPO":
            agent = PPO.load(path)
        elif agent_name == "A2C":
            agent = A2C.load(path)
        elif agent_name == "DDPG":
            agent = DDPG.load(path)
        elif agent_name == "SAC":
            agent = SAC.load(path)
        elif agent_name == "TD3":
            agent = TD3.load(path)
        else:
            raise ValueError("not a valid agent")
    except FileNotFoundError:
        print(f"{path} is not existing")

    return agent


class ExperimentAnalysis:
    @staticmethod
    def dimensionality_reduction(
        all_actions: list,
        all_states: list,
        all_rewards: list,
        all_broken_cutters: list,
        all_replaced_cutters: list,
        all_moved_cutters: list,
        perplexity: float,
    ) -> pd.DataFrame:
        # flatten all episode lists. TODO: do this smarter with itertools or the utility python pacakage I don't remember.
        all_actions = [item for sublist in all_actions for item in sublist]
        all_states = [item for sublist in all_states for item in sublist]
        all_rewards = [item for sublist in all_rewards for item in sublist]
        all_broken_cutters = [item for sublist in all_broken_cutters for item in sublist]
        all_replaced_cutters = [
            item for sublist in all_replaced_cutters for item in sublist
        ]
        all_moved_cutters = [item for sublist in all_moved_cutters for item in sublist]

        # apply TSNE
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="random",
            learning_rate="auto",
            verbose=1,
            n_jobs=-1,
        )
        all_actions_reduced_2D = reducer.fit_transform(np.array(all_actions))
        print("TSNE reduced actions to 2D")

        # collect results in dataframe and return
        df = pd.DataFrame(
            {
                "x": all_actions_reduced_2D[:, 0],
                "y": all_actions_reduced_2D[:, 1],
                "broken cutters": all_broken_cutters,
                "replaced cutters": all_replaced_cutters,
                "moved cutters": all_moved_cutters,
                "state": [np.round(s, 1) for s in all_states],
            }
        )
        return df


def mlflow_log_experiment(
    experiment_info: dict,
    main_dir: str,
    agent_dir: str,
    parameters: dict,
    sub_parameters: dict = None,
) -> None:
    """Logs setup data and results from one experiment to mlflow

    Args:
        experiment_name (str, optional): study name. Defaults to "PPO_2022_08_27_study".
    """
    print("Logging results to mlflow")

    df_log = pd.read_csv(f"{main_dir}/{agent_dir}/progress.csv")
    episode_best_reward = df_log[r"rollout/ep_rew_mean"].argmax()
    rewards = df_log.loc[episode_best_reward, r"rollout/ep_rew_mean"]
    df_env = pd.read_csv(f"{main_dir}/{agent_dir}/progress_env.csv")
    try:
        environment_results = dict(
            env_broken_cutters=df_env.loc[episode_best_reward, "avg_broken_cutters"],
            env_moved_cutters=df_env.loc[episode_best_reward, "avg_moved_cutters"],
            env_replaced_cutters=df_env.loc[episode_best_reward, "avg_replaced_cutters"],
            env_var_cutter_locations=df_env.loc[
                episode_best_reward, "var_cutter_locations"
            ],
        )
    except KeyError:
        print("The env episode values was not possible to retrieve")
        environment_results = {}
    mlflow.set_tracking_uri("./experiments/mlruns")
    mlflow.set_experiment(experiment_name=experiment_info["exp_study"])

    with mlflow.start_run():
        mlflow.log_params(parameters)
        if sub_parameters is not None:
            mlflow.log_params(sub_parameters)
        mlflow.log_params(experiment_info)
        mlflow.log_params(environment_results)
        mlflow.log_metric(key="reward", value=rewards.max())


class TrainingProgressCallback(BaseCallback):
    """Callback to log and plot parameters of the training
    progress."""

    def __init__(
        self,
        check_freq: int,
        save_path: str,
        name_prefix: str,
        MAX_STROKES: int,
        MODE: str,
        rich_console: Console,
        PLOT_PROGRESS: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)

        self.check_freq = check_freq  # checking frequency in [steps]
        self.save_path = save_path  # folder to save the plot to
        self.name_prefix = name_prefix  # name prefix for the plot
        self.MAX_STROKES = MAX_STROKES
        self.MODE = MODE
        self.r_console = rich_console
        self.PLOT_PROGRESS = PLOT_PROGRESS

        # accounting during training
        self.avg_penetration_episode: list = []
        self.broken_cutters_episode: list = []
        self.avg_broken_cutters_episode: list = []
        self.moved_cutters_episode: list = []
        self.avg_moved_cutters_episode: list = []
        self.inwards_moved_cutters_episode: list = []
        self.avg_inwards_moved_cutters_episode: list = []
        self.wrong_moved_cutters_episode: list = []
        self.avg_wrong_moved_cutters_episode: list = []
        self.replaced_cutters_episode: list[int] = []
        self.avg_replaced_cutters_episode: list[float] = []
        self.var_replaced_cutters_episode: list[float] = []
        self.cutter_locations_replaced: list = []

    def _on_step(self) -> bool:
        self.cutter_locations_replaced += self.training_env.get_attr("replaced_cutters")[
            0
        ]
        self.replaced_cutters_episode.append(
            len(self.training_env.get_attr("replaced_cutters")[0])
        )
        self.moved_cutters_episode.append(
            len(self.training_env.get_attr("moved_cutters")[0])
        )
        self.inwards_moved_cutters_episode.append(
            len(self.training_env.get_attr("inwards_moved_cutters")[0])
        )
        self.wrong_moved_cutters_episode.append(
            len(self.training_env.get_attr("wrong_moved_cutters")[0])
        )
        self.broken_cutters_episode.append(
            len(self.training_env.get_attr("broken_cutters")[0])
        )

        if self.n_calls % self.MAX_STROKES == 0:  # for every episode
            avg_replaced_cutters = np.mean(self.replaced_cutters_episode)
            var_replaced_cutters = (
                np.var(self.cutter_locations_replaced)
                if self.cutter_locations_replaced != []
                else 0
            )

            self.var_replaced_cutters_episode.append(var_replaced_cutters)
            self.avg_replaced_cutters_episode.append(avg_replaced_cutters)
            self.replaced_cutters_episode = []
            self.cutter_locations_replaced = []

            avg_moved_cutters = np.mean(self.moved_cutters_episode)
            self.avg_moved_cutters_episode.append(avg_moved_cutters)
            self.moved_cutters_episode = []

            avg_inwards_moved_cutters = np.mean(self.inwards_moved_cutters_episode)
            self.avg_inwards_moved_cutters_episode.append(avg_inwards_moved_cutters)
            self.inwards_moved_cutters_episode = []

            avg_wrong_moved_cutters = np.mean(self.wrong_moved_cutters_episode)
            self.avg_wrong_moved_cutters_episode.append(avg_wrong_moved_cutters)
            self.wrong_moved_cutters_episode = []

            avg_broken_cutters = np.mean(self.broken_cutters_episode)
            self.avg_broken_cutters_episode.append(avg_broken_cutters)
            self.broken_cutters_episode = []

            avg_penetration = np.mean(self.training_env.get_attr("penetration")[0])
            self.avg_penetration_episode.append(avg_penetration)

            if self.MODE == "training" and self.n_calls % (self.MAX_STROKES * 10) == 0:
                self.r_console.print(
                    f"Avg. #10th episode. Replaced: {avg_replaced_cutters: .2f} | Moved: {avg_moved_cutters: .2f} | Broken: {avg_broken_cutters: .3f} | Var. replaced: {var_replaced_cutters: .2f}"
                )

        if self.n_calls % self.check_freq == 0:
            rows_new = len(self.avg_replaced_cutters_episode)

            df_env_log = pd.DataFrame(
                {
                    "episodes": np.arange(rows_new),
                    "avg_replaced_cutters": self.avg_replaced_cutters_episode,
                    "var_cutter_locations": self.var_replaced_cutters_episode,
                    "avg_moved_cutters": self.avg_moved_cutters_episode,
                    "avg_inwards_moved_cutters": self.avg_inwards_moved_cutters_episode,
                    "avg_wrong_moved_cutters": self.avg_wrong_moved_cutters_episode,
                    "avg_broken_cutters": self.avg_broken_cutters_episode,
                    "avg_penetration": self.avg_penetration_episode,
                }
            )
            df_env_log.to_csv(f"{self.save_path}/progress_env.csv")

            if self.PLOT_PROGRESS:
                df_log = pd.read_csv(f"{self.save_path}/progress.csv")
                df_log["episodes"] = df_log[r"time/total_timesteps"] / self.MAX_STROKES
                Plotter.training_progress_plot(
                    df_log,
                    df_env_log,
                    savepath=f"{self.save_path}/{self.name_prefix}_training.svg",
                    show=False,
                )
        return True


class PrintExperimentInfoCallback(BaseCallback):
    """Callback that prints info when starting a new experiment.
    - parameters for agent
    - input to environment
    - agent name
    - optimization or checkpoint directory
    - NOTE: in callback you have access to self. model, env, globals, locals
    """

    def __init__(
        self,
        mode: str,
        agent_dir: str,
        hydra_dir: str,
        parameters: dict,
        n_episodes: int,
        checkpoint_interval: int,
        rich_console: Console,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.mode = mode
        self.agent_dir = agent_dir
        self.hydra_dir = hydra_dir
        self.parameters = parameters
        self.n_episodes = n_episodes
        self.checkpoint_interval = checkpoint_interval
        self.r_console = rich_console

    def _on_training_start(self) -> None:
        r_console = self.r_console
        r_console.print(f"\n{self.mode} agent in dir: {self.agent_dir}")
        r_console.print(f"Config values in: {self.hydra_dir}")
        r_console.print(
            f"Evaluation frequency is every {self.checkpoint_interval} episode / {self.checkpoint_interval * 1000} step"
        )
        r_console.print(f"Num episodes: {self.n_episodes}")
        r_console.print(
            f"\nTraining with these parameters: \n{pformat(self.parameters)}\n"
        )

    def _on_step(self) -> bool:
        return super()._on_step()


class MlflowLoggingCallback(BaseCallback):
    """Logs parameters and reward to mlflow."""

    def __init__(
        self,
        mode: str,
        agent_name: str,
        study: str,
        parameters: dict,
        sub_parameters: dict,
        agent_dir: str,
        save_path: str,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.mode = mode
        self.agent_name = agent_name
        self.study = study
        self.parameters = parameters
        self.sub_parameters = sub_parameters
        self.agent_dir = agent_dir
        self.save_path = save_path

    def _on_step(self) -> bool:
        return super()._on_step()

    def _on_training_end(self) -> None:
        experiment_info = dict(
            logdir=self.agent_dir,
            mode=self.mode,
            study=self.study,
            agent=self.agent_name,
        )
        mlflow_log_experiment(
            experiment_info,
            "checkpoints",
            self.agent_dir,
            self.parameters,
            self.sub_parameters,
        )
