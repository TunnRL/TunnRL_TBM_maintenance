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

Created on Sat Oct 30 12:57:51 2021
code contributors: Georg H. Erharter, Tom F. Hansen
"""

import uuid
from pprint import pformat
from typing import Any
import warnings

import gym
import optuna
import pandas as pd
import yaml
from optuna.integration.mlflow import MLflowCallback
from rich.console import Console
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


warnings.filterwarnings("ignore",
                        category=optuna.exceptions.ExperimentalWarning)


class PlotTrainingProgressCallback(BaseCallback):
    '''custom callback to log and visualize parameters of the training
    progress'''

    def __init__(self, check_freq: int, save_path: str, name_prefix: str,
                 MAX_STROKES: int, verbose: int = 0) -> None:
        super(PlotTrainingProgressCallback, self).__init__(verbose)

        self.check_freq = check_freq  # checking frequency in [steps]
        self.save_path = save_path  # folder to save the plot to
        self.name_prefix = name_prefix  # name prefix for the plot
        self.MAX_STROKES = MAX_STROKES

        self.pltr = Plotter()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            df_log = pd.read_csv(f'{self.save_path}/progress.csv')
            df_log['episodes'] = df_log[r'time/total_timesteps'] / self.MAX_STROKES

            self.pltr.training_progress_plot(df_log,
                                             savepath=f'{self.save_path}/{self.name_prefix}_training.svg',
                                             show=False)

        return True


class PrintExperimentInfoCallback(BaseCallback):
    """Callback that prints info when starting a new experiment.
        - parameters for agent
        - input to environment
        - agent name
        - optimization or checkpoint directory
        - NOTE: in callback you have access to self. model, env, globals, locals
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
        self.verbose = verbose
        self.n_episodes = n_episodes
        self.checkpoint_interval = checkpoint_interval

    def _on_training_start(self) -> None:
        print(f'\n{self.mode} agent in dir: {self.agent_dir} | Num episodes: {self.n_episodes}')
        print(f"Evaluation frequency is every {self.checkpoint_interval} episode / {self.checkpoint_interval * 1000} step")
        if self.verbose > 0:
            console = Console()
            console.print(f"\nTraining with these parameters: \n{pformat(self.parameters)}\n")
            
    def _on_step(self) -> bool:
        if self.n_calls < 1:
            # can call self.locals["actions"] to get actions
            # TODO: try to print the value, not only the scheduler object
            print("Learning rate: ", self.model.learning_rate)
            return True


class Optimization:
    """Functionality to train (optimize the reward of an agent)
    and hyperparameter tuning of agent parameters using Optuna."""


    def __init__(self, n_c_tot: int, environment: gym.Env, STUDY: str, EPISODES: int,
                 CHECKPOINT_INTERVAL: int, MODE: str, MAX_STROKES: int,
                 AGENT_NAME: str, DEFAULT_TRIAL: bool, VERBOSE_LEVEL: int,

                 MAX_NO_IMPROVEMENT: int) -> None:

        self.n_c_tot = n_c_tot
        self.environment = environment
        self.STUDY = STUDY
        self.EPISODES = EPISODES
        self.CHECKPOINT_INTERVAL = CHECKPOINT_INTERVAL
        self.MAX_STROKES = MAX_STROKES
        self.AGENT_NAME = AGENT_NAME
        self.DEFAULT_TRIAL = DEFAULT_TRIAL
        self.VERBOSE_LEVEL = VERBOSE_LEVEL
        self.MAX_NO_IMPROVEMENT = MAX_NO_IMPROVEMENT
        self.MODE = MODE

        self.n_actions = n_c_tot * n_c_tot
        # n steps, eg. 1000 steps x 100 checkpoint_interval = every 100 000 steps
        self.checkpoint_frequency = self.MAX_STROKES * self.CHECKPOINT_INTERVAL
        self.hparams = Hyperparameters()
        self.agent_dir: str = ""
       
    def objective(self, trial: optuna.trial.Trial) -> float | list[float]:
        '''Objective function that drives the optimization of parameter values
        for the RL-agent.'''

        if self.DEFAULT_TRIAL:
            parameters = {"policy": "MlpPolicy", "env": self.environment}
            self.DEFAULT_TRIAL = False
        else:
            parameters = self.hparams.suggest_hyperparameters(
                trial, self.AGENT_NAME, self.environment,
                steps_episode=self.MAX_STROKES, num_actions=self.n_actions)

        agent = self._setup_agent(self.AGENT_NAME, parameters)

        self.agent_dir = f'{self.AGENT_NAME}_{uuid.uuid4()}'
        callbacks, new_logger = self._setup_callbacks_and_logger(
            self.VERBOSE_LEVEL, parameters)
        
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
        print(f"Agent in dir: {self.agent_dir} has a reward of: {mean_ep_reward}\n")
        return final_reward

    def optimize(self, n_trials: int) -> None:
        """Optimize-function to be called in parallell process.
        
        Saves parameter-values and corresponding reward to mlflow.
        Start mlflow GUI by calling from optimization-dir:
        >>>mlflow ui

        Args:
            n_trials (int): Number of trials in each parallell process.
        """
        cb_mlflow = MLflowCallback(
            tracking_uri=f"./optimization/{self.AGENT_NAME}/mlruns", metric_name="reward")
        db_path = f"results/{self.STUDY}.db"
        db_file = f"sqlite:///{db_path}"
        study = optuna.load_study(study_name=self.STUDY, storage=db_file)

        try:
            study.optimize(
                self.objective,
                n_trials=n_trials,
                catch=(ValueError,),
                callbacks=[cb_mlflow]
            )
            
        except KeyboardInterrupt:
            print('Number of finished trials: ', len(study.trials))
            print('Best trial:')
            trial = study.best_trial
            print('  Value: ', trial.value)
            print('  Params: ')
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
        
        finally:
            print("Saving best parameters to a yaml_file")
            with open(f"results/{self.STUDY}_best_params_{study.best_value: .2f}.yaml", "w") as file:
                yaml.dump(study.best_params, file)

    def train_agent(self, best_parameters: dict) -> None:
        """Train agent with best parameters from an optimization study."""

        agent = self._setup_agent(self.AGENT_NAME, best_parameters)
        
        self.agent_dir = f'{self.AGENT_NAME}_{uuid.uuid4()}'
        callbacks, new_logger = self._setup_callbacks_and_logger(
            self.VERBOSE_LEVEL, best_parameters
        )
        
        if new_logger is not None:
            agent.set_logger(new_logger)

        agent.learn(total_timesteps=self.EPISODES * self.MAX_STROKES,
                    callback=callbacks)
        
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
        
    def _setup_callbacks_and_logger(self,
                                    verbose_level: int,
                                    parameters: dict) -> tuple[list, Any]:
        """Defining callbacks and logger used in training and optimizing an RL agent.

        Different setups for different training/optimization modes.

        Args:
            mode (str): training or optimization
            verbosity_level (int): setting the information level for logging
            parameters (dict): parameters for agent in experiment

        Returns:
            tuple[list, Any, str]: callbacklist, SB3-logger
        """
        agent_dir = self.agent_dir
        main_dir = ("optimization" if self.MODE == "optimization" else "checkpoints")
        max_no_improvement_evals = (self.MAX_NO_IMPROVEMENT if self.MODE == "optimizing" else 5)
        n_eval_episodes = (3 if self.MODE == "optimization" else 10)
        
        cb_list = []
        sb3_logger = None  # for debugging, ie. don't make dir etc.
        # callback values
        
        # callbacks for all modes
        stop_train_cb = StopTrainingOnNoModelImprovement(  # kicked off by EvalCallback
            max_no_improvement_evals=max_no_improvement_evals,
            min_evals=1,
            verbose=1)
            
        cb_list.append(
            EvalCallback(  # saves best model
                self.environment,
                best_model_save_path=f'{self.MODE}/{agent_dir}',
                log_path=f'{self.MODE}/{agent_dir}',
                deterministic=False,
                n_eval_episodes=n_eval_episodes,
                eval_freq=self.checkpoint_frequency,
                callback_after_eval=stop_train_cb,
                verbose=1, warn=False)
        )
        cb_list.append(
            PrintExperimentInfoCallback(
                self.MODE, agent_dir, parameters, self.EPISODES, self.CHECKPOINT_INTERVAL, verbose_level)
        )
        
        # verbosity vs logger and callbacks
        match verbose_level:
            case 0:
                sb3_logger = logger.configure(f'{main_dir}/{agent_dir}', ["csv"])
                
            case 1 | -1:
                sb3_logger = logger.configure(f'{main_dir}/{agent_dir}', ["csv", "tensorboard"])
                cb_list.append(
                    PlotTrainingProgressCallback(
                        check_freq=self.checkpoint_frequency,
                        save_path=f'{main_dir}/{agent_dir}',
                        name_prefix=f'{self.AGENT_NAME}',
                        MAX_STROKES=self.MAX_STROKES)
                )
                if self.MODE == "training":
                    cb_list.append(
                        CheckpointCallback(  # save model every checkpoint interval
                            save_freq=self.checkpoint_frequency,
                            save_path=f'checkpoints/{agent_dir}',
                            name_prefix=f'{self.AGENT_NAME}',
                            verbose=1)
                    )               
            case -2:
                cb_list = []
                cb_list.append(
                    PrintExperimentInfoCallback(
                        self.MODE, agent_dir, parameters, self.EPISODES, self.CHECKPOINT_INTERVAL, verbose_level)
                )
                print("debugging: no progresslogging, no evaluation, no checkpoints")
            case _:
                raise ValueError("not a valid verbosity_level")
        
        cb_list = CallbackList(cb_list)
        return cb_list, sb3_logger


def load_best_model(agent_name: str, main_dir: str,
                    agent_dir: str) -> BaseAlgorithm:
    """Load best model from a directory.

    Args:
        agent_name (str): name of RL-architecture (PPO, DDPG ...)
    """
    if agent_name == "DDP":
        agent_name = "DDPG"

    path = f'{main_dir}/{agent_dir}/best_model.zip'

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
