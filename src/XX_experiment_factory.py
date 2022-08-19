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

import gym
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from numpy.typing import NDArray
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


class PlotTrainingProgressCallback(BaseCallback):
    '''custom callback to log and visualize parameters of the training
    progress'''

    def __init__(self, check_freq: int, save_path: str, name_prefix: str, 
                 MAX_STROKES: int, AGENT_NAME: str, verbose: int = 0) -> None:
        super(PlotTrainingProgressCallback, self).__init__(verbose)

        self.check_freq = check_freq  # checking frequency in [steps]
        self.save_path = save_path  # folder to save the plot to
        self.name_prefix = name_prefix  # name prefix for the plot
        self.MAX_STROKES = MAX_STROKES
        self.AGENT_NAME = AGENT_NAME

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            df_log = pd.read_csv(f'{self.save_path}/progress.csv')
            df_log['episodes'] = df_log[r'time/total_timesteps'] / self.MAX_STROKES

            # works for all models
            ep = df_log['episodes'].iloc[-1]
            reward = df_log[r'rollout/ep_rew_mean'].iloc[-1]

            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
            ax1.plot(df_log['episodes'], df_log[r'rollout/ep_rew_mean'],
                     label=r'rollout/ep_rew_mean')
            try:
                ax1.scatter(df_log['episodes'], df_log['eval/mean_reward'],
                            label=r'eval/mean_reward')
            except KeyError:
                pass
            ax1.legend()
            ax1.grid(alpha=0.5)
            ax1.set_ylabel('reward')

            # model specific visualization of loss
            if self.AGENT_NAME == 'TD3' or self.AGENT_NAME == 'DDPG':
                ax2.plot(df_log['episodes'], df_log[r'train/critic_loss'],
                         label=r'train/critic_loss')
                ax2.plot(df_log['episodes'], df_log[r'train/actor_loss'],
                         label=r'train/actor_loss')
            elif self.AGENT_NAME == 'PPO':
                ax2.plot(df_log['episodes'], df_log[r'train/value_loss'],
                         label=r'train/value_loss')
                ax2.plot(df_log['episodes'], df_log[r'train/loss'],
                         label=r'train/loss')
                ax2.plot(df_log['episodes'], df_log[r'train/policy_gradient_loss'],
                         label=r'train/policy_gradient_loss')
                ax2.plot(df_log['episodes'], df_log[r'train/entropy_loss'],
                         label=r'train/entropy_loss')
            ax2.legend()
            ax2.grid(alpha=0.5)
            ax2.set_xlabel('episodes')
            ax2.set_ylabel('loss')

            plt.tight_layout()
            plt.savefig(f'{self.save_path}/{self.name_prefix}_training.svg')
            plt.close()

        return True


class PrintExperimentDirCallback:
    def __init__(self, agent_dir: str):
        self.agent_dir = agent_dir
    
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(f"Experiment info is saved in: {self.agent_dir}")


class Optimization:
    """Functionality to train (optimize the reward of an agent) 
    and hyperparameter tuning of agent parameters using Optuna."""

    def __init__(self, n_c_tot: int, environment: gym.Env, STUDY: str, EPISODES: int,
                 CHECKPOINT_INTERVAL: int, MODE: str, MAX_STROKES: int,
                 AGENT_NAME: str, DEFAULT_TRIAL: bool) -> None:

        self.n_c_tot = n_c_tot
        self.environment = environment
        self.STUDY = STUDY
        self.EPISODES = EPISODES
        self.CHECKPOINT_INTERVAL = CHECKPOINT_INTERVAL
        self.MODE = MODE
        self.MAX_STROKES = MAX_STROKES
        self.AGENT_NAME = AGENT_NAME
        self.DEFAULT_TRIAL = DEFAULT_TRIAL

        self.n_actions = n_c_tot * n_c_tot
        self.freq = self.MAX_STROKES * self.CHECKPOINT_INTERVAL  # checkpoint frequency
        self.parallell_process_counter: int = 0
        self.agent_dir: str
        self.hparams = Hyperparameters()

    def objective(self, trial: optuna.trial.Trial) -> float | list[float]:
        '''Objective function that drives the optimization of parameter values for the 
        RL-agent.'''
        
        self.parallell_process_counter += 1  # TODO: this is not working properly

        if self.DEFAULT_TRIAL:
            parameters = {"policy": "MlpPolicy", "env": self.environment}
            self.DEFAULT_TRIAL = False
        else:
            # hparams = Hyperparameters()
            parameters = self.hparams.suggest_hyperparameters(
                trial, self.AGENT_NAME, self.environment,
                steps_episode=self.MAX_STROKES, num_actions=self.n_actions)

        match self.AGENT_NAME:
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

        agent_dir = f'{self.AGENT_NAME}_{uuid.uuid4()}'
        self.agent_dir = agent_dir
        new_logger = logger.configure(f'optimization/{agent_dir}', ["csv"])

        print(f'\nOptimizing agent in dir: {agent_dir}. Agent: {self.AGENT_NAME} | Num episodes: {self.EPISODES}')
        print("\nTraining with these parameters: \n", parameters)
        # train agent with early stopping and save best agents only
        stop_train_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=1,
                                                         min_evals=1,
                                                         verbose=1)
        eval_cb = EvalCallback(self.environment,
                               best_model_save_path=f'optimization/{agent_dir}',
                               log_path=f'optimization/{agent_dir}',
                               deterministic=False,
                               n_eval_episodes=3,
                               eval_freq=self.freq,
                               callback_after_eval=stop_train_cb,
                               verbose=0, warn=False)
        custom_callback = PlotTrainingProgressCallback(check_freq=self.freq,
                                         save_path=f'optimization/{agent_dir}',
                                         name_prefix=f'{self.AGENT_NAME}',
                                         MAX_STROKES=self.MAX_STROKES,
                                         AGENT_NAME=self.AGENT_NAME)
        callback = CallbackList([eval_cb, custom_callback])

        agent.set_logger(new_logger)
        agent.learn(total_timesteps=self.EPISODES * self.MAX_STROKES,
                    callback=callback)
        del agent

        print('Load agent and evaluate on 10 last episodes...')
        agent = load_best_model(
            self.AGENT_NAME, main_dir="optimization", agent_dir=agent_dir)

        mean_ep_reward = evaluate_policy(agent, self.environment,
                                         n_eval_episodes=10,
                                         deterministic=False,
                                         warn=False)[0]
        final_reward = mean_ep_reward  # objective's reward
        print(f"Agent in dir: {agent_dir} has a reward of: {mean_ep_reward}\n")
        return final_reward
    
    def optimize(self, n_trials: int) -> None:
        """Optimize-function to be called in parallell process.

        Args:
            n_trials (int): Number of trials in each parallell process.
        """
        cb_print_agent_dir = PrintExperimentDirCallback("test")
        db_path = f"results/{self.STUDY}.db"
        db_file = f"sqlite:///{db_path}"
        study = optuna.load_study(study_name=self.STUDY, storage=db_file)
        
        try:
            study.optimize(self.objective,
                        n_trials=n_trials,
                        catch=(ValueError,),
                        callbacks=[cb_print_agent_dir]
                        )
            
        except KeyboardInterrupt: #TODO: check how to interrupt
            print('Number of finished trials: ', len(study.trials))
            print('Best trial:')
            trial = study.best_trial
            print('  Value: ', trial.value)
            print('  Params: ')
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

    def train_agent(self, agent_name: str, best_parameters: dict) -> None:
        """Train agent with best parameters from an optimization study."""
        
        agent_dir = f'{self.AGENT_NAME}_{uuid.uuid4()}'
        best_parameters.update(dict(tensorboard_log=f"optimization/{agent_dir}"))
        print(f"Checkpoint dir: {agent_dir}")

        match agent_name:
            case "PPO":
                agent = PPO(**best_parameters)
            case "SAC":
                agent = SAC(**best_parameters)
            case "A2C":
                agent = A2C(**best_parameters)
            case "DDPG":
                agent = DDPG(**best_parameters)
            case "TD3":
                agent = TD3(**best_parameters)
            case _:
                raise NotImplementedError()

        new_logger = logger.configure(f'checkpoints/{agent_dir}', ["csv"])
        # mode that trains an agent based on previous OPTUNA study
        checkpoint_callback = CheckpointCallback(save_freq=self.freq,
                                                 save_path=f'checkpoints/{agent_dir}',
                                                 name_prefix=f'{self.AGENT_NAME}',
                                                 verbose=1)
        custom_callback = PlotTrainingProgressCallback(check_freq=self.freq,
                                         save_path=f'checkpoints/{agent_dir}',
                                         name_prefix=f'{self.AGENT_NAME}',
                                         MAX_STROKES=self.MAX_STROKES,
                                         AGENT_NAME=self.AGENT_NAME)
        eval_cb = EvalCallback(self.environment,
                               best_model_save_path=f'checkpoints/{agent_dir}',
                               log_path='checkpoints',
                               deterministic=False,
                               n_eval_episodes=10,
                               eval_freq=self.freq,
                               verbose=1, warn=False)

        # Create the callback list
        callback = CallbackList([checkpoint_callback, eval_cb,
                                 custom_callback])
        # TODO implement callback that logs also environmental training
        # TODO parameters (broken cutters, n changes per ep etc.)
        agent.set_logger(new_logger)
        agent.learn(total_timesteps=self.EPISODES * self.MAX_STROKES,
                    callback=callback)
    

def load_best_model(agent_name: str, main_dir: str, agent_dir: str) -> BaseAlgorithm:
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