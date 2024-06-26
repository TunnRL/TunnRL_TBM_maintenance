# -*- coding: utf-8 -*-
"""
Code for the paper:

Towards smart TBM cutter changing with reinforcement learning (working title)
Georg H. Erharter, Tom F. Hansen, Thomas Marcher, Amund Bruland
JOURNAL NAME
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
from typing import Any

import gym
import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
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
from umap import UMAP

from XX_hyperparams import Hyperparameters
from XX_plotting import Plotter


warnings.filterwarnings("ignore",
                        category=optuna.exceptions.ExperimentalWarning)


class PlotTrainingProgressCallback(BaseCallback):
    '''custom callback to log and visualize parameters of the training
    progress'''

    def __init__(self, check_freq: int, save_path: str, name_prefix: str,
                 MAX_STROKES: int, MODE: str, verbose: int = 0) -> None:
        super(PlotTrainingProgressCallback, self).__init__(verbose)

        self.check_freq = check_freq  # checking frequency in [steps]
        self.save_path = save_path  # folder to save the plot to
        self.name_prefix = name_prefix  # name prefix for the plot
        self.MAX_STROKES = MAX_STROKES
        self.MODE = MODE

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
        # self.cutter_locations_replaced: list = []

        self.pltr = Plotter()

    def _on_step(self) -> bool:
        # TODO: check out a potential row-shift-error due to nan from eval round
        # returning a list of the cutter numbers replaced in each stroke
        # self.cutter_locations_replaced += self.training_env.get_attr("replaced_cutters")
        self.replaced_cutters_episode.append(len(self.training_env.get_attr("replaced_cutters")[0]))
        self.moved_cutters_episode.append(len(self.training_env.get_attr("moved_cutters")[0]))
        self.inwards_moved_cutters_episode.append(len(self.training_env.get_attr("inwards_moved_cutters")[0]))
        self.wrong_moved_cutters_episode.append(len(self.training_env.get_attr("wrong_moved_cutters")[0]))
        self.broken_cutters_episode.append(len(self.training_env.get_attr("broken_cutters")[0]))

        if self.n_calls % self.MAX_STROKES == 0:  # for every episode
            avg_replaced_cutters = np.mean(self.replaced_cutters_episode)
            # var_replaced_cutters = np.nanvar(list(chain(*self.cutter_locations_replaced)))
            # We want the variance of the cutter number to be changed to be big,
            # thereby replaced cutters all over the cutterhead

            # self.cutter_locations_replaced = list(chain(*self.cutter_locations_replaced))
            # self.var_replaced_cutters_episode.append(var_replaced_cutters)
            self.avg_replaced_cutters_episode.append(avg_replaced_cutters)
            self.replaced_cutters_episode = []
            # self.cutter_locations_replaced = []

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
                print(f"Avg. #episode. Replaced: {avg_replaced_cutters} | Moved: {avg_moved_cutters} | Broken: {avg_broken_cutters} | Var. replaced: {var_replaced_cutters}")

        if self.n_calls % self.check_freq == 0:
            df_log = pd.read_csv(f'{self.save_path}/progress.csv')
            rows_new = len(self.avg_replaced_cutters_episode)

            df_env_log = pd.DataFrame({'episodes': np.arange(rows_new),
                                       'avg_replaced_cutters': self.avg_replaced_cutters_episode,
                                       # 'var_cutter_locations': self.var_replaced_cutters_episode,
                                       'avg_moved_cutters': self.avg_moved_cutters_episode,
                                       'avg_inwards_moved_cutters': self.avg_inwards_moved_cutters_episode,
                                       'avg_wrong_moved_cutters': self.avg_wrong_moved_cutters_episode,
                                       'avg_broken_cutters': self.avg_broken_cutters_episode,
                                       'avg_penetration': self.avg_penetration_episode})
            df_env_log.to_csv(f'{self.save_path}/progress_env.csv')

            df_log['episodes'] = df_log[r'time/total_timesteps'] / self.MAX_STROKES

            self.pltr.training_progress_plot(df_log, df_env_log,
                                             savepath=f'{self.save_path}/{self.name_prefix}_training.svg',
                                             show=False)
        return True


class PrintExperimentInfoCallback(BaseCallback):
    """Callback that prints info when starting a new experiment.
        - parameters for agent
        - input to environment
        - agent name
        - optimization or checkpoint directory
        - NOTE: in callback you have access to self. model, env, globals,
            locals
    """

    def __init__(self,
                 mode: str,
                 agent_dir: str,
                 parameters: dict,
                 n_episodes: int,
                 checkpoint_interval: int,
                 verbose: int = 0,
                 ) -> None:
        super().__init__(verbose=verbose)
        self.mode = mode
        self.agent_dir = agent_dir
        self.parameters = parameters
        self.n_episodes = n_episodes
        self.checkpoint_interval = checkpoint_interval

    def _on_training_start(self) -> None:
        print(f'\n{self.mode} agent in dir: {self.agent_dir} | Num episodes: {self.n_episodes}')
        print(f"Evaluation frequency is every {self.checkpoint_interval} episode / {self.checkpoint_interval * 1000} step")
        print(f"\nTraining with these parameters: \n{self.parameters}\n")

    def _on_step(self) -> bool:
        return super()._on_step()


class Optimization:
    """Functionality to train (optimize the reward of an agent)
    and hyperparameter tuning of agent parameters using Optuna."""

    def __init__(self, n_c_tot: int, environment: gym.Env, STUDY: str, EPISODES: int,
                 CHECKPOINT_INTERVAL: int, LOG_DATAFORMATS: list[str], LOG_MLFLOW: bool, 
                 MODE: str, MAX_STROKES: int, AGENT_NAME: str, DEFAULT_TRIAL: bool,
                 MAX_NO_IMPROVEMENT: int) -> None:
        """Initialize the Optimization object.

        Args:
            n_c_tot (int): Number of cutters
            environment (gym.Env): TBM environment with state, actions, reward
            STUDY (str): Optuna study object
            EPISODES (int): Number of RL episodes in the optimization
            CHECKPOINT_INTERVAL (int): Frequency of evaluations in episodes
            LOG_DATAFORMATS (list[str]): dataformats for logging, eg. ["csv", "tensorboard"]
            LOG_MLFLOW (bool): wether to log experiments to mlflow database
            MODE (str): the process mode of the optimization: training, optimization
            MAX_STROKES (int): Number of TBM strokes (steps) in each episode
            AGENT_NAME (str): name of RL algorithm, eg. PPO, DDPG ...
            DEFAULT_TRIAL (bool): If an optimization first should run with default params
            VERBOSE_LEVEL (int): Directing how much information that is logged and presented
            MAX_NO_IMPROVEMENT (int): How many episodes without improvement of reward
        """

        self.n_c_tot = n_c_tot
        self.environment = environment
        self.STUDY = STUDY
        self.EPISODES = EPISODES
        self.CHECKPOINT_INTERVAL = CHECKPOINT_INTERVAL
        self.LOG_DATAFORMATS = LOG_DATAFORMATS
        self.LOG_MLFLOW = LOG_MLFLOW
        self.MAX_STROKES = MAX_STROKES
        self.AGENT_NAME = AGENT_NAME
        self.DEFAULT_TRIAL = DEFAULT_TRIAL
        self.MAX_NO_IMPROVEMENT = MAX_NO_IMPROVEMENT
        self.MODE = MODE

        self.n_actions = n_c_tot * n_c_tot
        # n steps, eg. 1000 steps x 100 checkpoint_interval = every 100 000 steps
        self.checkpoint_frequency = self.MAX_STROKES * self.CHECKPOINT_INTERVAL
        self.hparams = Hyperparameters()
        self.agent_dir: str = ""

    def objective(self, trial: optuna.trial.Trial) -> float | list[float]:
        """Objective function that drives the optimization of parameter values
        for the RL-agent."""

        if self.DEFAULT_TRIAL:
            parameters = {"policy": "MlpPolicy", "env": self.environment}
            self.DEFAULT_TRIAL = False
        else:
            parameters = self.hparams.suggest_hyperparameters(
                trial, self.AGENT_NAME, self.environment,
                steps_episode=self.MAX_STROKES, num_actions=self.n_actions)

        agent = self._setup_agent(self.AGENT_NAME, parameters)

        self.agent_dir = f'{self.AGENT_NAME}_{uuid.uuid4()}'
        callbacks, new_logger = self._setup_callbacks_and_logger(parameters)

        if new_logger is not None:
            agent.set_logger(new_logger)

        agent.learn(total_timesteps=self.EPISODES * self.MAX_STROKES,
                    callback=callbacks)
        del agent

        print('Load agent and evaluate on 10 last episodes...')
        agent = load_best_model(
            self.AGENT_NAME, main_dir="optimization", agent_dir=self.agent_dir)

        mean_ep_reward = evaluate_policy(agent, self.environment,
                                         n_eval_episodes=10,
                                         deterministic=False,
                                         warn=False)[0]
        final_reward = mean_ep_reward  # objective's reward

        # logging and reporting
        print(f"Agent in dir: {self.agent_dir} has a reward of: {mean_ep_reward}\n")

        if self.LOG_MLFLOW is True:
            self._mlflow_log_experiment(
                main_dir="optimization",
                parameters=parameters)

        return final_reward

    def optimize(self, n_trials: int) -> None:
        """Optimize-function to be called in parallell process.

        Args:
            n_trials (int): Number of trials in each parallell process.

        Returns:
            None
        """
        db_path = f"results/{self.STUDY}.db"
        db_file = f"sqlite:///{db_path}"
        study = optuna.load_study(study_name=self.STUDY, storage=db_file)

        try:
            study.optimize(
                self.objective,
                n_trials=n_trials,
                catch=(ValueError,)
            )
        except KeyboardInterrupt:
            pass

        finally:  # this will always be run
            print("Saving best parameters to a yaml_file")
            yaml.dump(study.best_params, f"results/{self.STUDY}_best_params_{study.best_value: .2f}.yaml")

    def train_agent(self, best_parameters: dict) -> None:
        """Train agent with best parameters from an optimization study."""

        agent = self._setup_agent(self.AGENT_NAME, best_parameters)

        self.agent_dir = f'{self.AGENT_NAME}_{uuid.uuid4()}'
        callbacks, new_logger = self._setup_callbacks_and_logger(best_parameters)

        if new_logger is not None:
            agent.set_logger(new_logger)

        agent.learn(total_timesteps=self.EPISODES * self.MAX_STROKES,
                    callback=callbacks)

        if self.LOG_MLFLOW is True:
            self._mlflow_log_experiment(
                main_dir="checkpoints",
                parameters=best_parameters)

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
            stop_train_cb = StopTrainingOnNoModelImprovement(  # kicked off by EvalCallback
                max_no_improvement_evals=self.MAX_NO_IMPROVEMENT,
                min_evals=1,
                verbose=1)

            cb_list.append(
                EvalCallback(  # saves best model
                    self.environment,
                    best_model_save_path=f'optimization/{agent_dir}',
                    log_path=f'optimization/{agent_dir}',
                    deterministic=False,
                    n_eval_episodes=3,
                    eval_freq=self.checkpoint_frequency,
                    callback_after_eval=stop_train_cb,
                    verbose=1, warn=False)
            )
            cb_list.append(
                PlotTrainingProgressCallback(
                    check_freq=self.checkpoint_frequency,
                    save_path=f'optimization/{agent_dir}',
                    name_prefix=f'{self.AGENT_NAME}',
                    MAX_STROKES=self.MAX_STROKES,
                    MODE=self.MODE)
            )
            cb_list.append(
                PrintExperimentInfoCallback(
                    self.MODE, agent_dir, parameters, self.EPISODES, self.CHECKPOINT_INTERVAL)
            )
            sb3_logger = logger.configure(
                folder=f'optimization/{agent_dir}', format_strings=self.LOG_DATAFORMATS)

        elif self.MODE == "training":
            cb_list.append(
                CheckpointCallback(  # save model every checkpoint interval
                    save_freq=self.checkpoint_frequency,
                    save_path=f'checkpoints/{agent_dir}',
                    name_prefix=f'{self.AGENT_NAME}',
                    verbose=1)
            )
            cb_list.append(
                EvalCallback(  # saves best model
                    self.environment,
                    best_model_save_path=f'checkpoints/{agent_dir}',
                    log_path=f'checkpoints/{agent_dir}',
                    deterministic=False,
                    n_eval_episodes=10,
                    eval_freq=self.checkpoint_frequency,
                    verbose=1, warn=False)
            )
            cb_list.append(
                PlotTrainingProgressCallback(
                    check_freq=self.checkpoint_frequency,
                    save_path=f'checkpoints/{agent_dir}',
                    name_prefix=f'{self.AGENT_NAME}',
                    MAX_STROKES=self.MAX_STROKES,
                    MODE=self.MODE)
            )
            cb_list.append(
                PrintExperimentInfoCallback(
                    self.MODE, agent_dir, parameters, self.EPISODES, self.CHECKPOINT_INTERVAL)
            )
            sb3_logger = logger.configure(folder=f'checkpoints/{agent_dir}', format_strings=self.LOG_DATAFORMATS)
        else:
            raise ValueError("not a valid mode")

        cb_list = CallbackList(cb_list)
        return cb_list, sb3_logger

    def _mlflow_log_experiment(self, main_dir: str, parameters: dict) -> None:
        """Logs setup data and results from one experiment to mlflow

        Logging to mlflow to directly compare different runs
        and to map projectdir to parameters

        Use:
            cd into experiment dir and run command: mlflow ui
        """
        print("Logging results to mlflow")

        experiment_info = dict(
            exp_logdir=self.agent_dir,
            exp_mode=self.MODE,
            exp_study=self.STUDY,
            exp_agent=self.AGENT_NAME,
        )

        df_log = pd.read_csv(f"{main_dir}/{self.agent_dir}/progress.csv")
        episode_best_reward = df_log[r"rollout/ep_rew_mean"].argmax()
        rewards = df_log.loc[episode_best_reward, r"rollout/ep_rew_mean"]
        df_env = pd.read_csv(f"{main_dir}/{self.agent_dir}/progress_env.csv")
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
            mlflow.log_params(experiment_info)
            mlflow.log_params(environment_results)
            mlflow.log_metric(key="reward", value=rewards.max())


class ExperimentAnalysis:

    def dimensionality_reduction(self, all_actions: list, all_states: list,
                                 all_rewards: list, all_broken_cutters: list,
                                 all_replaced_cutters: list,
                                 all_moved_cutters: list,
                                 perplexity: float,
                                 reducer: str) -> pd.DataFrame:
        '''function that applies different dimensionality reduction algorithms
        (unsupervised ML) to help analyzing the high dimensional actions space
        and the overall learned policy'''
        # flatten all episode lists
        all_actions = [item for sublist in all_actions for item in sublist]
        all_states = [item for sublist in all_states for item in sublist]
        all_rewards = [item for sublist in all_rewards for item in sublist]
        all_broken_cutters = [item for sublist in all_broken_cutters for item in sublist]
        all_replaced_cutters = [item for sublist in all_replaced_cutters for item in sublist]
        all_moved_cutters = [item for sublist in all_moved_cutters for item in sublist]
        print(np.array(all_states).shape)
        avg_cutter_life = np.mean(np.array(all_states), axis=1)

        # apply TSNE
        if reducer == 'TSNE':
            reducer = TSNE(n_components=2, perplexity=perplexity, init='random',
                           learning_rate='auto', verbose=1, n_jobs=-1,
                           random_state=42)
            print('reducer to apply on actions: TSNE')
        elif reducer == 'UMAP':
            reducer = UMAP(n_components=2, n_neighbors=600, min_dist=0.4,
                           n_jobs=-1, random_state=42)
            print('reducer to apply on actions: UMAP')
        else:
            raise ValueError("not a valid reducer")
        all_actions_reduced_2D = reducer.fit_transform(np.array(all_actions))

        # collect results in dataframe and return
        df = pd.DataFrame({'x': all_actions_reduced_2D[:, 0],
                           'y': all_actions_reduced_2D[:, 1],
                           'broken cutters': all_broken_cutters,
                           'replaced cutters': all_replaced_cutters,
                           'moved cutters': all_moved_cutters,
                           'rewards': all_rewards,
                           'state': [np.round(s, 1) for s in all_states],
                           'avg. cutter life': avg_cutter_life})
        print('dimensionality reduction finished')
        return df


def load_best_model(agent_name: str, main_dir: str,
                    agent_dir: str) -> BaseAlgorithm:
    """Load best model from a zip-file on the format: best_model.zip

    Args:
        agent_name (str): name of RL-architecture (PPO, DDPG ...)
        main_dir (str): main directory for loading, eg. optimization or checkpoints
        agent_dir (str): name of agent directory in main dir

    Returns:
        Instantiated RL-object
    """
    path = f'{main_dir}/{agent_dir}/best_model'
    print(path)

    if agent_name == 'PPO':
        agent = PPO.load(path)
    elif agent_name == 'A2C':
        agent = A2C.load(path)
    elif agent_name == 'DDPG':
        agent = DDPG.load(path)
    elif agent_name == 'SAC':
        agent = SAC.load(path)
    elif agent_name == 'TD3':
        agent = TD3.load(path)

    return agent



