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
import matplotlib.cm as mplcm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from gym import spaces
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

from XX_hyperparams import DefaultParameters, Hyperparameters


class plotter:
    '''class that contains functions to visualzie the progress of the
    training and / or individual samples of it'''

    def __init__(self):
        pass

    def sample_ep_plot(self, states, actions, rewards, ep, savepath,
                       replaced_cutters, moved_cutters):
        '''plot of different recordings of one exemplary episode'''

        cmap = mplcm.get_cmap('viridis')

        states_arr = np.vstack(states[:-1])
        actions_arr = np.vstack(actions)
        actions_arr = np.where(actions_arr > 0, 1, 0)  # binarize

        fig = plt.figure(tight_layout=True, figsize=(10.236, 7.126))
        gs = gridspec.GridSpec(nrows=4, ncols=2, width_ratios=[5, 1])

        # cutter life line plot
        ax = fig.add_subplot(gs[0, 0])
        for cutter in range(states_arr.shape[1]):
            rgba = cmap(cutter / states_arr.shape[1])
            ax.plot(np.arange(states_arr.shape[0]), states_arr[:, cutter],
                    color=rgba, label=cutter)
        h_legend, l_legend = ax.get_legend_handles_labels()
        ax.set_xlim(left=-1, right=actions_arr.shape[0]+1)
        ax.set_ylim(top=1.05, bottom=-0.05)
        ax.set_title(f'episode {ep}', fontsize=10)
        ax.set_ylabel('cutter life')
        ax.set_xticklabels([])
        ax.grid(alpha=0.5)

        # dedicated subplot for a legend to first axis
        lax = fig.add_subplot(gs[0, 1])
        lax.legend(h_legend, l_legend, borderaxespad=0, ncol=3,
                   loc='upper left', fontsize=7.5)
        lax.axis('off')

        # bar plot that shows how many cutters were moved
        ax = fig.add_subplot(gs[1, 0])
        ax.bar(x=np.arange(actions_arr.shape[0]),
               height=moved_cutters, color='grey')
        avg_changed = np.mean(moved_cutters)
        ax.axhline(y=avg_changed, color='black')
        ax.text(x=950, y=avg_changed-avg_changed*0.05,
                s=f'avg. moved cutters / stroke: {round(avg_changed, 2)}',
                color='black', va='top', ha='right', fontsize=7.5)
        ax.set_xlim(left=-1, right=actions_arr.shape[0]+1)
        ax.set_ylabel('cutter moves\nper stroke')
        ax.set_xticklabels([])
        ax.grid(alpha=0.5)

        # bar plot that shows how many cutters were replaced
        ax = fig.add_subplot(gs[2, 0])
        ax.bar(x=np.arange(actions_arr.shape[0]),
               height=replaced_cutters, color='grey')
        avg_changed = np.mean(replaced_cutters)
        ax.axhline(y=avg_changed, color='black')
        ax.text(x=950, y=avg_changed-avg_changed*0.05,
                s=f'avg. replacements / stroke: {round(avg_changed, 2)}',
                color='black', va='top', ha='right', fontsize=7.5)
        ax.set_xlim(left=-1, right=actions_arr.shape[0]+1)
        ax.set_ylabel('cutter replacements\nper stroke')
        ax.set_xticklabels([])
        ax.grid(alpha=0.5)

        # plot that shows the reward per stroke
        ax = fig.add_subplot(gs[3, 0])
        ax.scatter(x=np.arange(len(rewards)), y=rewards, color='grey', s=1)
        ax.axhline(y=np.mean(rewards), color='black')
        ax.text(x=950, y=np.mean(rewards)-0.05,
                s=f'avg. reward / stroke: {round(np.mean(rewards), 2)}',
                color='black', va='top', ha='right', fontsize=7.5)
        ax.set_ylim(bottom=0, top=1)
        ax.set_xlim(left=-1, right=actions_arr.shape[0]+1)
        ax.set_ylabel('reward / stroke')
        ax.set_xlabel('strokes')
        ax.grid(alpha=0.5)

        plt.tight_layout()
        plt.savefig(Path(savepath))
        plt.close()

    def state_action_plot(self, states, actions, n_strokes, savepath):
        '''plot that shows combinations of states and actions for the first
        n_strokes of an episode'''
        fig = plt.figure(figsize=(20, 6))

        ax = fig.add_subplot(211)
        ax.imshow(np.vstack(states[:n_strokes]).T, aspect='auto',
                  interpolation='none', vmin=0, vmax=1)
        ax.set_yticks(np.arange(-.5, self.n_c_tot), minor=True)
        ax.set_xticks(np.arange(-.5, n_strokes), minor=True)

        ax.set_xticks(np.arange(n_strokes), minor=False)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='minor', color='white')
        ax.tick_params(axis='x', which='major', length=10, color='lightgrey')

        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.set_xlim(left=-1, right=n_strokes)
        ax.set_ylim(bottom=-0.5, top=self.n_c_tot-0.5)
        ax.set_yticks
        ax.set_ylabel('cutter states on\ncutter positions')

        ax = fig.add_subplot(212)

        for stroke in range(n_strokes):
            for i in range(self.n_c_tot):
                # select cutter from action vector
                cutter = actions[stroke][i*self.n_c_tot: i*self.n_c_tot+self.n_c_tot]
                if np.max(cutter) < 0.9:
                    # cutter is not acted on
                    pass
                elif np.argmax(cutter) == i:
                    # cutter is replaced
                    ax.scatter(stroke, i, edgecolor='black', color='black',
                               zorder=50)
                else:
                    # cutter is moved from original position to somewhere else
                    # original position of old cutter that is replaced
                    ax.scatter(stroke, i, edgecolor=f'C{i}', color='black',
                               zorder=20)
                    # new position where old cutter is moved to
                    ax.scatter(stroke, np.argmax(cutter), edgecolor=f'C{i}',
                               color=f'C{i}', zorder=20)
                    # arrow / line that connects old and new positions
                    ax.arrow(x=stroke, y=i,
                             dx=0, dy=-(i-np.argmax(cutter)), color=f'C{i}',
                             zorder=10)
        ax.set_xticks(np.arange(n_strokes), minor=True)
        ax.set_yticks(np.arange(self.n_c_tot), minor=True)
        ax.set_xlim(left=-1, right=n_strokes)
        ax.grid(zorder=0, which='both', color='grey')
        ax.set_xlabel('strokes')
        ax.set_ylabel('actions on\ncutter positions')

        plt.tight_layout(h_pad=0)
        plt.savefig(Path(savepath))
        plt.close()

    def environment_parameter_plot(self, savepath, ep):
        '''plot that shows the generated TBM parameters of the episode'''
        x = np.arange(len(self.Jv_s))  # strokes
        # count broken cutters due to blocky conditions
        n_brokens = np.count_nonzero(self.brokens, axis=1)

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6,
                                                           figsize=(12, 9))

        ax1.plot(x, self.Jv_s, color='black')
        ax1.grid(alpha=0.5)
        ax1.set_ylabel('Volumetric Joint count\n[joints / m3]')
        ax1.set_xlim(left=0, right=len(x))
        ax1.set_title(f'episode {ep}', fontsize=10)
        ax1.set_xticklabels([])

        ax2.plot(x, self.UCS_s, color='black')
        ax2.grid(alpha=0.5)
        ax2.set_ylabel('Rock UCS\n[MPa]')
        ax2.set_xlim(left=0, right=len(x))
        ax2.set_xticklabels([])

        ax3.plot(x, self.FPIblocky_s, color='black')
        ax3.hlines([50, 100, 200, 300], xmin=0, xmax=len(x), color='black',
                   alpha=0.5)
        ax3.set_ylim(bottom=0, top=400)
        ax3.set_ylabel('FPI blocky\n[kN/m/mm/rot]')
        ax3.set_xlim(left=0, right=len(x))
        ax3.set_xticklabels([])

        ax4.plot(x, self.TF_s, color='black')
        ax4.grid(alpha=0.5)
        ax4.set_ylabel('thrust force\n[kN]')
        ax4.set_xlim(left=0, right=len(x))
        ax4.set_xticklabels([])

        ax5.plot(x, self.penetration, color='black')
        ax5.grid(alpha=0.5)
        ax5.set_ylabel('penetration\n[mm/rot]')
        ax5.set_xlim(left=0, right=len(x))
        ax5.set_xticklabels([])

        ax6.plot(x, n_brokens, color='black')
        ax6.grid(alpha=0.5)
        ax6.set_ylabel('broken cutters\ndue to blocks')
        ax6.set_xlabel('strokes')
        ax6.set_xlim(left=0, right=len(x))

        plt.tight_layout()
        plt.savefig(Path(savepath))
        plt.close()

    def trainingprogress_plot(self, df, summed_actions, name):
        '''plot of different metrices of the whole training progress so far'''
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, ncols=1,
                                                 figsize=(7.126, 5))  # 12, 9

        ax0.imshow(np.vstack(summed_actions).T, aspect='auto', cmap='Greys_r',
                   interpolation='none')
        ax0.set_ylabel('actions on\ncutter positions')
        ax0.set_title(name, fontsize=10)
        ax0.set_xticklabels([])

        ax1.plot(df['episode'], df['avg_changes_per_interv'], color='black')
        ax1.grid(alpha=0.5)
        ax1.set_xlim(left=0, right=len(df))
        ax1.set_ylabel('avg. cutter\nchanges / stroke')
        ax1.yaxis.set_label_position('right')
        ax1.set_xticklabels([])

        ax2.plot(df['episode'], df['avg_brokens'], color='black')
        ax2.grid(alpha=0.5)
        ax2.set_xlim(left=0, right=len(df))
        ax2.set_ylabel('avg. n broken\ncutters / stroke')
        ax2.set_xticklabels([])

        ax3.plot(df['episode'], df['avg_rewards'], color='black')
        ax3.set_xlim(left=0, right=len(df))
        ax3.set_ylim(top=1, bottom=0)
        ax3.grid(alpha=0.5)
        ax3.set_ylabel('avg. reward\n/ stroke')
        ax3.yaxis.set_label_position('right')
        ax3.set_xlabel('episodes')

        plt.tight_layout()
        plt.savefig(Path(f'checkpoints/{name}_progress.svg'))
        plt.close()

    def action_visualization(self, action, n_c_tot, savepath=None,
                             binary=False):
        '''plot that visualizes a single action'''
        if binary is True:
            action = np.where(action > 0.9, 1, -1)

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax1.imshow(np.reshape(action, (n_c_tot, n_c_tot)),
                        vmin=-1, vmax=1)
        ax1.set_xticks(np.arange(-.5, n_c_tot), minor=True)
        ax1.set_yticks(np.arange(-.5, n_c_tot), minor=True)
        ax1.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel('cutters to move to')
        ax1.set_ylabel('cutters to acton on')
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
            plt.close()


class maintenance:
    '''class that contains functions that describe the maintenance effort of
    changing cutters on a TBM's cutterhead. Based on this the reward is
    computed'''

    def __init__(self, t_C_max, n_c_tot):
        self.t_C_max = t_C_max  # maximum time to change one cutter [min]
        self.n_c_tot = n_c_tot  # total number of cutters

    def reward(self, replaced_cutters, moved_cutters, good_cutters):
        '''reward function that also takes moving of cutters into account'''
        if good_cutters < self.n_c_tot * 0.5:
            r = 0
        elif good_cutters == self.n_c_tot and replaced_cutters + moved_cutters == 0:
            r = 1
        else:
            ratio1 = good_cutters / self.n_c_tot
            ratio2 = (moved_cutters / self.n_c_tot) * 1.05
            ratio3 = (replaced_cutters / self.n_c_tot) * 0.9
            r = ratio1 - ratio2 - ratio3
        r = 0 if r < 0 else r
        return r


class CustomEnv(gym.Env, plotter):
    '''implementation of the custom environment that simulates the cutter wear
    and provides the agent with a state and reward signal'''

    def __init__(self, n_c_tot, LIFE, MAX_STROKES, STROKE_LENGTH,
                 cutter_pathlenghts, R, t_C_max):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(n_c_tot*n_c_tot,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_c_tot,))

        self.n_c_tot = n_c_tot  # total number of cutters
        self.LIFE = LIFE  # theoretical durability of one cutter [m]
        self.MAX_STROKES = MAX_STROKES  # number of strokes to simulate
        self.STROKE_LENGTH = STROKE_LENGTH  # length of one stroke [m]
        self.cutter_pathlenghts = cutter_pathlenghts  # rolling lengths [m]
        self.R = R  # radius of cutterhead [m]

        self.m = maintenance(t_C_max, n_c_tot)

    def step(self, actions):
        '''main function that moves the environment one step further'''
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

    def implement_action(self, action, state_before):
        '''function that interprets the "raw action" and modifies the state'''
        state_new = state_before
        self.replaced_cutters = 0
        self.moved_cutters = 0

        for i in range(self.n_c_tot):
            cutter = action[i*self.n_c_tot: i*self.n_c_tot+self.n_c_tot]
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

    def rand_walk_with_bounds(self, n_dp):
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

    def generate(self, Jv_low=0, Jv_high=22, UCS_center=80, UCS_range=30):
        '''function generates TBM recordings for one episode. Equations and
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

    def reset(self):
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

    def __init__(self, check_freq, save_path, name_prefix, MAX_STROKES, AGENT_NAME,
                 verbose=0):
        super(CustomCallback, self).__init__(verbose)

        self.check_freq = check_freq  # checking frequency in [steps]
        self.save_path = save_path  # folder to save the plot to
        self.name_prefix = name_prefix  # name prefix for the plot
        self.MAX_STROKES = MAX_STROKES
        self.AGENT_NAME = AGENT_NAME

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:

            df_log = pd.read_csv(Path(f'{self.save_path}/progress.csv'))
            df_log['episodes'] = df_log[r'time/total_timesteps'] / self.MAX_STROKES
            # df_log.dropna(axis=0, subset=[r'time/time_elapsed'], inplace=True)

            # works for all models
            ep = df_log['episodes'].iloc[-1]
            reward = df_log[r'rollout/ep_rew_mean'].iloc[-1]
            # print(f'episode: {ep}, reward: {reward}\n')

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

            # model psecific visualization of loss
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
    """Functionality to train (optimize an agent) and hyperparameter tuning with optuna."""

    def __init__(self, 
                 n_c_tot: int, 
                 environment: gym.Env, 
                 EPISODES: int, 
                 CHECKPOINT_INTERVAL: int,
                 MODE: str, 
                 MAX_STROKES: int, 
                 AGENT_NAME: str) -> None:
        
        self.n_c_tot = n_c_tot
        self.environment = environment
        self.EPISODES = EPISODES
        self.CHECKPOINT_INTERVAL = CHECKPOINT_INTERVAL
        self.MODE = MODE
        self.MAX_STROKES = MAX_STROKES
        self.AGENT_NAME = AGENT_NAME

        self.n_actions = n_c_tot * n_c_tot
        self.freq = self.MAX_STROKES * self.CHECKPOINT_INTERVAL  # checkpoint frequency

    def objective(self, trial: optuna.trial.Trial) -> float | list[float]:
        '''objective function that runs the RL environment and agent either for
        an optimization or an optimized agent'''
        print('\n')
        
        hparams = Hyperparameters()
        parameter_suggestions = hparams.suggest_hyperparameters(
            trial, self.AGENT_NAME, self.environment, steps_episode=self.MAX_STROKES,
            num_actions=self.n_actions)

        match self.AGENT_NAME:
            case "PPO":
                agent = PPO(**parameter_suggestions)
            case "SAC":
                agent = SAC(**parameter_suggestions)
            case "A2C":
                agent = A2C(**parameter_suggestions)
            case "DDPG":
                agent = DDPG(**parameter_suggestions)
            case "TD3":
                agent = TD3(**parameter_suggestions)
            case _:
                raise NotImplementedError(f"{self.AGENT_NAME} is not implemented")

        agent_dir = self.AGENT_NAME + datetime.now().strftime("%Y%m%d-%H%M%S")
        new_logger = logger.configure(f'optimization/{agent_dir}', ["csv"])

        print(f'agent: {self.AGENT_NAME}')
        # train agent with early stopping and save best agents only
        stop_train_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3,
                                                            min_evals=2,
                                                            verbose=1)
        eval_cb = EvalCallback(self.environment,
                                best_model_save_path=f'optimization/{agent_dir}',
                                log_path=f'optimization/{agent_dir}',
                                deterministic=False,
                                n_eval_episodes=3,
                                eval_freq=self.freq,
                                callback_after_eval=stop_train_cb,
                                verbose=1, warn=False)
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
        agent = self.load_best_model(self.AGENT_NAME, agent_dir)

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
        new_logger = logger.configure(Path(f'checkpoints/{agent_dir}'), ["csv"])
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

    def load_best_model(self, agent_name: str, agent_dir: str) -> BaseAlgorithm:
        """Load best model so far in optuna study.

        Args:
            agent_name (str): name of RL-architecture (PPO, DDPG ...)
        """
        agents = dict(PPO=PPO(), A2C=A2C(), DDPG=DDPG(), SAC=SAC(), TD3=TD3())
        trained_agent = agents[agent_name].load(f'optimization/{agent_dir}/best_model.zip')
        return trained_agent
        

    def enqueue_defaults(self, study: optuna.study.Study, agent_name: str, n_trials: int):
        '''Insert manually a study with default parameters in n_trials experiments.'''
        defaults = DefaultParameters()
        for i in range(n_trials):
            study.enqueue_trial(defaults.get_agent_default_params(agent_name))
            
        print(f'{n_trials} studies with {agent_name} default parameters inserted')

        return study


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
    m = maintenance(t_C_max, n_c_tot)
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
