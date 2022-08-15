# -*- coding: utf-8 -*-
"""
Towards optimized TBM cutter changing policies with reinforcement learning
G.H. Erharter, T.F. Hansen
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Custom library that contains different classes for the reinforcement learning
based cutter changing environment

Created on Sat Oct 30 12:57:51 2021
code contributors: Georg H. Erharter, Tom F. Hansen
"""

from datetime import datetime
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from gym import spaces
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
from XX_utility import load_best_model


class Maintenance:
    '''class that contains functions that describe the maintenance effort of
    changing cutters on a TBM's cutterhead. Based on this the reward is
    computed'''

    def __init__(self, t_C_max: int, n_c_tot: int) -> None:
        """Setup

        Args:
            t_C_max (int): maximum time to change one cutter  [min]
            n_c_tot (int): total number of cutters
        """
        self.t_C_max = t_C_max
        self.n_c_tot = n_c_tot

    def reward(self, replaced_cutters: int, moved_cutters: int, good_cutters: int) -> float:
        """Reward function. Drives the agent learning process.

        Handles replacing and moving cutters.

        TODO: move hardcoded values into a common place for definition or config.
        """
        if good_cutters < self.n_c_tot * 0.5:
            r = 0
        elif good_cutters == self.n_c_tot and replaced_cutters + moved_cutters == 0:
            r = 1
        else:
            ratio1 = good_cutters / self.n_c_tot
            ratio2 = (moved_cutters / self.n_c_tot) * 1.05
            ratio3 = (replaced_cutters / self.n_c_tot) * 0.9
            r = ratio1 - ratio2 - ratio3

        r = max(0, r)

        return r


class CustomEnv(gym.Env):
    '''Implementation of the custom environment that simulates the cutter wear
    and provides the agent with a state and reward signal.
    '''
    def __init__(self,
                 n_c_tot: int,
                 LIFE: int, 
                 MAX_STROKES: int, 
                 STROKE_LENGTH: float,
                 cutter_pathlenghts: float, 
                 CUTTERHEAD_RADIUS: float, 
                 T_C_MAX: int) -> None:
        """Initializing custom environment for a TBM cutter operation.

        Args:
            n_c_tot (int): total number of cutters
            LIFE (int): theoretical durability of one cutter [m]
            MAX_STROKES (int): numer of strokes to simulate
            STROKE_LENGTH (float): length of one stroke [m]
            cutter_pathlenghts (float): rolling lengths [m]
            CUTTERHEAD_RADIUS (float): radius of cutterhead
            T_C_MAX (int): max time for changing one cutter [min]
        """
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_c_tot * n_c_tot,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_c_tot,))

        self.n_c_tot = n_c_tot
        self.LIFE = LIFE
        self.MAX_STROKES = MAX_STROKES
        self.STROKE_LENGTH = STROKE_LENGTH
        self.cutter_pathlenghts = cutter_pathlenghts
        self.R = CUTTERHEAD_RADIUS

        self.m = Maintenance(T_C_MAX, n_c_tot)
        # state variables assigned in class methods:
        self.state: NDArray
        self.epoch: int
        self.replaced_cutters: int
        self.moved_cutters: int
        self.penetration: NDArray

    def step(self, actions: NDArray) -> tuple[NDArray, float, bool, dict]:
        '''Main function that moves the environment one step further.
            - Updates the state and reward.
            - Checks if the terminal state is reached.
        '''
        # replace cutters based on action of agent
        self.state = self.implement_action(actions, self.state)

        # compute reward
        good_cutters = len(np.where(self.state > 0)[0])
        # n_c_to_change = self.replaced_cutters + self.moved_cutters
        reward = self.m.reward(self.replaced_cutters, self.moved_cutters,
                               good_cutters)

        # update cutter life based on how much wear occurs
        p = self.penetration[self.epoch] / 1000  # [m/rot]
        rot_per_stroke = self.STROKE_LENGTH / p
        self.state = self.state - rot_per_stroke * (self.cutter_pathlenghts / self.LIFE)
        self.state = np.where(self.state <= 0, 0, self.state)

        # set cutter lifes to 0 based on blockyness
        self.state[np.where(self.brokens[self.epoch, :] == 1)[0]] = 0

        self.epoch += 1
        if self.epoch >= self.MAX_STROKES:
            terminal = True
        else:
            terminal = False

        return self.state, reward, terminal, {}

    def implement_action(self, action: NDArray, state_before: NDArray) -> NDArray:
        '''Function that interprets the "raw action" and modifies the state.'''
        state_new = state_before
        self.replaced_cutters = 0
        self.moved_cutters = 0

        for i in range(self.n_c_tot):
            cutter = action[i * self.n_c_tot: i * self.n_c_tot + self.n_c_tot]
            if np.max(cutter) < 0.9:
                # cutter is not acted on
                pass
            elif np.argmax(cutter) == i:
                # cutter is replaced
                state_new[i] = 1
                self.replaced_cutters += 1
            else:
                # cutter is moved from original position to somewhere else
                state_new[np.argmax(cutter)] = state_new[i]
                state_new[i] = 1
                self.moved_cutters += 1

        return state_new

    def rand_walk_with_bounds(self, n_dp: int) -> NDArray:
        '''function generates a random walk within the limits 0 and 1'''
        bounds = .05

        x = [np.random.uniform(low=0, high=1, size=1)]

        for move in np.random.uniform(low=-bounds, high=bounds, size=n_dp):
            x_temp = x[-1] + move
            if x_temp <= 1 and x_temp >= 0:
                x.append(x_temp)
            else:
                x.append(x[-1] - move)

        return np.array(x[1:])

    def generate(self, Jv_low: NDArray = 0, Jv_high: NDArray = 22, 
                 UCS_center: NDArray = 80, UCS_range: NDArray = 30) -> tuple:
        '''Function generates TBM recordings for one episode. Equations and
        models based on Delisio & Zhao (2014) - "A new model for TBM
        performance prediction in blocky rock conditions",
        http://dx.doi.org/10.1016/j.tust.2014.06.004'''

        Jv_s = self.rand_walk_with_bounds(self.MAX_STROKES) * (Jv_high - Jv_low) + Jv_low  # [joints / m3]
        UCS_s = UCS_center + self.rand_walk_with_bounds(self.MAX_STROKES) * UCS_range  # [MPa]

        # eq 9, Delisio & Zhao (2014) - [kN/m/mm/rot]
        FPIblocky_s = np.squeeze(np.exp(6) * Jv_s**-0.82 * UCS_s**0.17)

        brokens = np.zeros(shape=(self.MAX_STROKES, self.n_c_tot))

        for stroke in range(self.MAX_STROKES):
            # based on FPI blocky cutters have different likelyhoods to break
            if FPIblocky_s[stroke] > 200 and FPIblocky_s[stroke] <= 300:
                if np.random.randint(0, 100) == 0:
                    size = np.random.randint(1, 4)
                    selection = np.random.choice(np.arange(self.n_c_tot),
                                                 replace=False, size=size)
                    brokens[stroke, :][selection] = 1
            elif FPIblocky_s[stroke] > 100 and FPIblocky_s[stroke] <= 200:
                if np.random.randint(0, 50) == 0:
                    size = np.random.randint(1, 4)
                    selection = np.random.choice(np.arange(self.n_c_tot),
                                                 replace=False, size=size)
                    brokens[stroke, :][selection] = 1
            elif FPIblocky_s[stroke] >= 50 and FPIblocky_s[stroke] <= 100:
                if np.random.randint(0, 10) == 0:
                    size = np.random.randint(1, 4)
                    selection = np.random.choice(np.arange(self.n_c_tot),
                                                 replace=False, size=size)
                    brokens[stroke, :][selection] = 1
            elif FPIblocky_s[stroke] < 50:
                if np.random.randint(0, 100) == 0:
                    size = np.random.randint(1, 4)
                    selection = np.random.choice(np.arange(self.n_c_tot),
                                                 replace=False, size=size)
                    brokens[stroke, :][selection] = 1

        # eq 13, Delisio & Zhao (2014)
        TF_s = np.squeeze((-523 * np.log(Jv_s) + 2312) * (self.R*2))  # [kN]
        TF_s = np.where(TF_s > 20_000, 20_000, TF_s)

        # eq 7, Delisio & Zhao (2014)
        # TF_s already considers shield friction
        penetration = (TF_s / (self.R*2)) / FPIblocky_s  # [mm/rot]

        return Jv_s, UCS_s, FPIblocky_s, brokens, TF_s, penetration

    def reset(self) -> NDArray:
        '''reset an environment to its initial state'''
        self.state = np.full((self.n_c_tot), 1)  # start with new cutters
        self.epoch = 0  # reset epoch counter
        # generate new TBM data for episode
        self.Jv_s, self.UCS_s, self.FPIblocky_s, self.brokens, self.TF_s, self.penetration = self.generate()
        return self.state

    def render(self):
        pass

    def close(self):
        pass


class CustomCallback(BaseCallback):
    '''custom callback to log and visualize parameters of the training
    progress'''

    def __init__(self, check_freq: int, save_path: str, name_prefix: str, 
                 MAX_STROKES: int, AGENT_NAME: str, verbose: int = 0) -> None:
        super(CustomCallback, self).__init__(verbose)

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


class Optimization:
    """Functionality to train (optimize the reward of an agent) 
    and hyperparameter tuning of agent parameters using Optuna."""

    def __init__(self, n_c_tot: int, environment: gym.Env, EPISODES: int,
                 CHECKPOINT_INTERVAL: int, MODE: str, MAX_STROKES: int,
                 AGENT_NAME: str, DEFAULT_TRIAL: bool) -> None:

        self.n_c_tot = n_c_tot
        self.environment = environment
        self.EPISODES = EPISODES
        self.CHECKPOINT_INTERVAL = CHECKPOINT_INTERVAL
        self.MODE = MODE
        self.MAX_STROKES = MAX_STROKES
        self.AGENT_NAME = AGENT_NAME
        self.DEFAULT_TRIAL = DEFAULT_TRIAL

        self.n_actions = n_c_tot * n_c_tot
        self.freq = self.MAX_STROKES * self.CHECKPOINT_INTERVAL  # checkpoint frequency
        self.parallell_process_counter: int = 0

    def objective(self, trial: optuna.trial.Trial) -> float | list[float]:
        '''Objective function that drives the optimization of parameter values for the 
        RL-agent.'''
        
        self.parallell_process_counter += 1  # TODO: this is not working properly

        if self.DEFAULT_TRIAL:
            parameters = {"policy": "MlpPolicy", "env": self.environment}
            self.DEFAULT_TRIAL = False
        else:
            hparams = Hyperparameters()
            parameters = hparams.suggest_hyperparameters(
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

        agent_dir = self.AGENT_NAME + datetime.now().strftime("%Y%m%d-%H%M%S")
        new_logger = logger.configure(f'optimization/{agent_dir}', ["csv"])

        print(f'Optimizing parallell process {self.parallell_process_counter}. Agent: {self.AGENT_NAME} | Num episodes: {self.EPISODES}')
        print("\nTraining with these parameters: ", parameters)
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
        custom_callback = CustomCallback(check_freq=self.freq,
                                         save_path=f'optimization/{agent_dir}',
                                         name_prefix=f'{self.AGENT_NAME}',
                                         MAX_STROKES=self.MAX_STROKES,
                                         AGENT_NAME=self.AGENT_NAME)
        callback = CallbackList([eval_cb, custom_callback])

        agent.set_logger(new_logger)
        agent.learn(total_timesteps=self.EPISODES * self.MAX_STROKES,
                    callback=callback)
        del agent

        print('load agent and evaluate on 10 last episodes')
        agent = load_best_model(
            self.AGENT_NAME, main_dir="optimization", agent_dir=agent_dir)

        mean_ep_reward = evaluate_policy(agent, self.environment,
                                         n_eval_episodes=10,
                                         deterministic=False,
                                         warn=False)[0]
        final_reward = mean_ep_reward  # objective's reward
        return final_reward

    def train_agent(self, agent_name: str, best_parameters: dict) -> None:
        """Train agent with best parameters from an optimization study."""
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

        agent_dir = self.AGENT_NAME + datetime.now().strftime("%Y%m%d-%H%M%S")
        new_logger = logger.configure(Path(f'checkpoints/{agent_dir}'),
                                      ["csv"])
        # mode that trains an agent based on previous OPTUNA study
        checkpoint_callback = CheckpointCallback(save_freq=self.freq,
                                                 save_path=Path(f'checkpoints/{agent_dir}'),
                                                 name_prefix=f'{self.AGENT_NAME}',
                                                 verbose=1)
        custom_callback = CustomCallback(check_freq=self.freq,
                                         save_path=Path(f'checkpoints/{agent_dir}'),
                                         name_prefix=f'{self.AGENT_NAME}',
                                         MAX_STROKES=self.MAX_STROKES,
                                         AGENT_NAME=self.AGENT_NAME)
        eval_cb = EvalCallback(self.environment,
                               best_model_save_path=Path(f'checkpoints/{agent_dir}'),
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


if __name__ == "__main__":

    # visualization of all possible reward states
    t_C_max = 75  # maximum time to change one cutter [min]
    n_c_tot = 28  # total number of cutters

    replaced_cutters = np.arange(n_c_tot+1)
    moved_cutters = np.arange(n_c_tot+1)
    good_cutters = np.arange(n_c_tot+1)

    # get all combinations
    combined = np.array(np.meshgrid(replaced_cutters, moved_cutters, good_cutters)).T.reshape(-1,3)
    # delete combinations where replaced_cutters + moved_cutters > n_c_tot
    del_ids = np.where(combined[:, 0] + combined[:, 1] > n_c_tot)[0]
    combined = np.delete(combined, del_ids, axis=0)

    replaced_cutters = combined[:, 0]
    moved_cutters = combined[:, 1]
    good_cutters = combined[:, 2]

    # compute rewards for all states
    m = Maintenance(t_C_max, n_c_tot)
    rewards = []
    for i in range(len(combined)):
        rewards.append(m.reward(replaced_cutters=replaced_cutters[i],
                                moved_cutters=moved_cutters[i],
                                good_cutters=good_cutters[i]))
    rewards = np.array(rewards)

    low_r_id = np.argmin(rewards)
    high_r_id = np.argmax(rewards)
    len_low_r = len(np.where(rewards == rewards.min())[0])
    len_high_r = len(np.where(rewards == rewards.max())[0])

    print(f'there are {len_low_r} combinations with {rewards.min()} reward')
    print(f'lowest reward of {rewards.min()} with combination:')
    print(f'\t{replaced_cutters[low_r_id]} replaced cutters')
    print(f'\t{moved_cutters[low_r_id]} moved cutters')
    print(f'\t{good_cutters[low_r_id]} good cutters\n')

    print(f'there are {len_high_r} combinations with {rewards.max()} reward')
    print(f'highest reward of {rewards.max()} with combination:')
    print(f'\t{replaced_cutters[high_r_id]} replaced cutters')
    print(f'\t{moved_cutters[high_r_id]} moved cutters')
    print(f'\t{good_cutters[high_r_id]} good cutters')

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection='3d')
    scatter_plot= ax.scatter(replaced_cutters, moved_cutters, good_cutters,
                             c=rewards, edgecolor='black', s=40)
    ax.set_xlabel('n replaced cutters')
    ax.set_ylabel('n moved cutters')
    ax.set_zlabel('n good cutters')

    plt.colorbar(scatter_plot, label='reward')
    plt.tight_layout()
