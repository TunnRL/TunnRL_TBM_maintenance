# -*- coding: utf-8 -*-
"""
Code for the paper:

Towards smart TBM cutter changing with reinforcement learning (working title)
Georg H. Erharter, Tom F. Hansen, Thomas Marcher, Amund Bruland
JOURNAL NAME
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Custom library for plotting functions that are used throughout the code
framework.

code contributors: Georg H. Erharter, Tom F. Hansen
"""

from collections import Counter
from itertools import chain
import matplotlib
import matplotlib.cm as mplcm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from os import listdir
from pandas.errors import EmptyDataError
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder


class Plotter:
    '''class that contains functions to visualzie the progress of the
    training and / or individual samples of it'''

    def sample_ep_plot(self, states: list, actions: list, rewards: list,
                       replaced_cutters: list, moved_cutters: list,
                       n_cutters: int, savepath: str = None,
                       show: bool = True) -> None:
        '''plot of different recordings of one exemplary episode'''

        n_replaced_cutters = [len(cutters) for cutters in replaced_cutters]
        n_moved_cutters = [len(cutters) for cutters in moved_cutters]
        strokes = np.arange(len(moved_cutters))

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
            ax.plot(strokes, states_arr[:, cutter],
                    color=rgba, label=cutter)
        h_legend, l_legend = ax.get_legend_handles_labels()
        ax.set_xlim(left=-1, right=actions_arr.shape[0]+1)
        ax.set_ylim(top=1.05, bottom=-0.05)
        ax.set_title(f'sample episode', fontsize=10)
        ax.set_ylabel('cutter life')
        ax.set_xticklabels([])
        ax.grid(alpha=0.5)

        # dedicated subplot for a legend to first axis
        lax = fig.add_subplot(gs[0, 1])
        id_middle = int(len(l_legend)/2)
        lax.legend(list(np.take(h_legend, [0, id_middle, -1])),
                   list(np.take(l_legend, [0, id_middle, -1])),
                   borderaxespad=0, ncol=1, loc='upper left', fontsize=7.5,
                   title='cutter positions')
        lax.axis('off')

        # bar plot that shows how many cutters were moved
        ax = fig.add_subplot(gs[1, 0])
        ax.bar(x=strokes, height=n_moved_cutters, color='grey')
        avg_changed = np.mean(n_moved_cutters)
        ax.axhline(y=avg_changed, color='black')
        ax.text(x=950, y=avg_changed-avg_changed*0.05,
                s=f'avg. moved cutters / stroke: {round(avg_changed, 2)}',
                color='black', va='top', ha='right', fontsize=7.5)
        ax.set_xlim(left=-1, right=actions_arr.shape[0]+1)
        ax.set_ylabel('cutter moves\nper stroke')
        ax.set_xticklabels([])
        ax.grid(alpha=0.5)

        # bar plot that shows n replacements over episode
        moved_count = Counter(chain(*map(set, moved_cutters)))
        ax = fig.add_subplot(gs[1, 1])
        ax.bar(x=list(moved_count.keys()),
               height=list(moved_count.values()),
               color='grey', edgecolor='black')
        ax.set_xlim(left=0, right=n_cutters)
        ax.grid(alpha=0.5)
        ax.set_xlabel('n cutter moves per\nposition')

        # bar plot that shows how many cutters were replaced
        ax = fig.add_subplot(gs[2, 0])
        ax.bar(x=strokes, height=n_replaced_cutters, color='grey')
        avg_changed = np.mean(n_replaced_cutters)
        ax.axhline(y=avg_changed, color='black')
        ax.text(x=950, y=avg_changed-avg_changed*0.05,
                s=f'avg. replacements / stroke: {round(avg_changed, 2)}',
                color='black', va='top', ha='right', fontsize=7.5)
        ax.set_xlim(left=-1, right=actions_arr.shape[0]+1)
        ax.set_ylabel('cutter replacements\nper stroke')
        ax.set_xticklabels([])
        ax.grid(alpha=0.5)

        # bar plot that shows n replacements over episode
        replaced_count = Counter(chain(*map(set, replaced_cutters)))
        ax = fig.add_subplot(gs[2, 1])
        ax.bar(x=list(replaced_count.keys()),
               height=list(replaced_count.values()),
               color='grey', edgecolor='black')
        ax.set_xlim(left=0, right=n_cutters)
        ax.grid(alpha=0.5)
        ax.set_xlabel('n cutter replacements per\nposition')

        # plot that shows the reward per stroke
        ax = fig.add_subplot(gs[3, 0])
        ax.scatter(x=strokes, y=rewards, color='grey', s=1)
        ax.axhline(y=np.mean(rewards), color='black')
        ax.text(x=950, y=np.mean(rewards)-0.05,
                s=f'avg. reward / stroke: {round(np.mean(rewards), 2)}',
                color='black', va='top', ha='right', fontsize=7.5)
        ax.set_ylim(bottom=-1.05, top=1.05)
        ax.set_xlim(left=-1, right=actions_arr.shape[0]+1)
        ax.set_ylabel('reward / stroke')
        ax.set_xlabel('strokes')
        ax.grid(alpha=0.5)

        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
        if show is False:
            plt.close()

    def state_action_plot(self, states: list, actions: list, n_strokes: int,
                          n_c_tot: int, rewards: list, savepath: str = None,
                          show: bool = True) -> None:
        '''plot that shows combinations of states and actions for the first
        n_strokes of an episode'''
        fig = plt.figure(figsize=(20, 9))

        ax = fig.add_subplot(311)
        ax.imshow(np.vstack(states[:n_strokes]).T, aspect='auto',
                  interpolation='none', vmin=0, vmax=1, cmap='cividis')
        ax.set_yticks(np.arange(-.5, n_c_tot), minor=True)
        ax.set_xticks(np.arange(-.5, n_strokes), minor=True)
        ax.set_xticks(np.arange(n_strokes), minor=False)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='minor', color='white')
        ax.tick_params(axis='x', which='major', length=10, color='lightgrey')
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.set_xlim(left=-1, right=n_strokes)
        ax.set_ylim(bottom=-0.5, top=n_c_tot - 0.5)
        ax.set_ylabel('cutter states on\ncutter positions')

        ax = fig.add_subplot(312)
        for stroke in range(n_strokes):

            for i in range(n_c_tot):
                # select cutter from action vector
                cutter = actions[stroke][i*n_c_tot: i*n_c_tot+n_c_tot]
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
        ax.set_yticks(np.arange(n_c_tot), minor=True)
        ax.set_xticks(np.arange(n_strokes), minor=False)
        ax.tick_params(axis='x', which='minor', color='white')
        ax.tick_params(axis='x', which='major', length=10, color='lightgrey')
        ax.set_xticklabels([])
        ax.set_xlim(left=-1, right=n_strokes)
        ax.grid(zorder=0, which='both', color='grey')
        ax.set_ylabel('actions on\ncutter positions')

        ax = fig.add_subplot(313)
        ax.scatter(x=np.arange(n_strokes), y=rewards[:n_strokes],
                   color='grey')
        ax.set_xticks(np.arange(n_strokes), minor=True)
        ax.set_xlim(left=-1, right=n_strokes)
        ax.set_ylim(bottom=-1.05, top=1.05)
        ax.grid(zorder=0, which='both', color='grey')
        ax.set_ylabel('reward per stroke')
        ax.set_xlabel('strokes')

        plt.tight_layout(h_pad=0)
        if savepath is not None:
            plt.savefig(savepath)
        if show is False:
            plt.close()

    def environment_parameter_plot(self, ep, env, savepath: str = None,
                                   show: bool = True) -> None:
        '''plot that shows the generated TBM parameters of the episode'''
        x = np.arange(len(env.Jv_s))  # strokes
        # count broken cutters due to blocky conditions
        n_brokens = np.count_nonzero(env.brokens, axis=1)

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6,
                                                           figsize=(9, 7))

        ax1.plot(x, env.Jv_s, color='black')
        ax1.grid(alpha=0.5)
        ax1.set_ylabel('Volumetric\nJoint count\n[joints / m3]')
        ax1.set_xlim(left=0, right=len(x))
        # ax1.set_title(f'episode {ep}', fontsize=10)
        ax1.set_xticklabels([])

        ax2.plot(x, env.UCS_s, color='black')
        ax2.grid(alpha=0.5)
        ax2.set_ylabel('Rock UCS\n[MPa]')
        ax2.set_xlim(left=0, right=len(x))
        ax2.set_xticklabels([])

        ax3.plot(x, env.FPIblocky_s, color='black')
        ax3.hlines([50, 100, 200, 300], xmin=0, xmax=len(x), color='black',
                   alpha=0.5)
        ax3.fill_between(x, env.FPIblocky_s, 0, color='lightyellow')
        ax3.fill_between(x, np.where(env.FPIblocky_s <= 300, env.FPIblocky_s, 300), 0, color='yellow')
        ax3.fill_between(x, np.where(env.FPIblocky_s <= 200, env.FPIblocky_s, 200), 0, color='goldenrod')
        ax3.fill_between(x, np.where(env.FPIblocky_s <= 100, env.FPIblocky_s, 100), 0, color='darkred')
        ax3.fill_between(x, np.where(env.FPIblocky_s <= 50, env.FPIblocky_s, 50), 0, color='yellow')
        ax3.text(x=x.max()-10, y=330, s='massive', ha='right')
        ax3.text(x=x.max()-10, y=230, s='blocky', ha='right')
        ax3.text(x=x.max()-10, y=130, s='blocky/very blocky', ha='right')
        ax3.text(x=x.max()-10, y=60, s='very blocky', ha='right')
        ax3.text(x=x.max()-10, y=10, s='blocky/disturbed', ha='right')
        ax3.set_ylim(bottom=0, top=400)
        ax3.set_ylabel('FPI blocky\n[kN/m/mm/rot]')
        ax3.set_xlim(left=0, right=len(x))
        ax3.set_xticklabels([])

        ax4.plot(x, env.TF_s, color='black')
        ax4.grid(alpha=0.5)
        ax4.set_ylabel('thrust force\n[kN]')
        ax4.set_xlim(left=0, right=len(x))
        ax4.set_xticklabels([])

        ax5.plot(x, env.penetration, color='black')
        ax5.grid(alpha=0.5)
        ax5.set_ylabel('penetration\n[mm/rot]')
        ax5.set_xlim(left=0, right=len(x))
        ax5.set_xticklabels([])

        ax6.bar(x, n_brokens, color='grey', edgecolor='black', width=3)
        ax6.grid(alpha=0.5)
        ax6.set_ylabel('broken cutters\ndue to blocks')
        ax6.set_xlabel('strokes')
        ax6.set_xlim(left=0, right=len(x))

        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath, dpi=600)
        if show is False:
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

    def custom_parallel_coordinate_plot(self, df_study: pd.DataFrame,
                                        params: list,
                                        le_activation: LabelEncoder,
                                        savepath: str = None,
                                        show: bool = True) -> None:
        '''custom implementation of the plot_parallel_coordinate() function of
        optuna:
        https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_parallel_coordinate.html#optuna.visualization.plot_parallel_coordinate
        '''
        # TODO consider le_noise

        # df_study['params_lr_schedule'] = np.where(df_study['params_lr_schedule']=='constant', 0, 1)

        fig, ax = plt.subplots(figsize=(18, 9))

        mins = df_study[params].min().values
        f = df_study[params].values-mins
        maxs = np.max(f, axis=0)

        cmap = matplotlib.cm.get_cmap('cividis')
        norm = matplotlib.colors.Normalize(vmin=df_study['value'].min(),
                                           vmax=df_study['value'].max())

        for t in range(len(df_study)):
            df_temp = df_study.sort_values(by='value').iloc[t]
            x = np.arange(len(params))
            y = df_temp[params].values

            y = y - mins
            y = y / maxs

            if df_temp['state'] == 'FAIL':
                ax.plot(x, y, c='red', alpha=0.5)
            elif df_temp['state'] == 'RUNNING':
                pass
            else:

                if df_temp['value'] < 600:
                    ax.plot(x, y, c=cmap(norm(df_temp['value'])), alpha=0.2)
                else:
                    ax.plot(x, y, c=cmap(norm(df_temp['value'])), alpha=1,
                            zorder=10)

        ax.scatter(x, np.zeros(x.shape), color='black')
        ax.scatter(x, np.ones(x.shape), color='black')

        for i in range(len(x)):
            if params[i] == 'params_activation_fn':
                ax.text(x=x[i], y=-0.01, s=le_activation.classes_[0],
                        horizontalalignment='center', verticalalignment='top')
                ax.text(x=x[i], y=0.5, s=le_activation.classes_[1],
                        horizontalalignment='right', verticalalignment='top')
                ax.text(x=x[i], y=1.01, s=le_activation.classes_[2],
                        horizontalalignment='center',
                        verticalalignment='bottom')
            elif params[i] == 'params_lr_schedule':
                ax.text(x=x[i], y=-0.01, s='constant',
                        horizontalalignment='center', verticalalignment='top')
                ax.text(x=x[i], y=1.01, s='linear decrease',
                        horizontalalignment='center',
                        verticalalignment='bottom')
            elif params[i] == 'params_use_sde':
                ax.text(x=x[i], y=-0.01, s='False',
                        horizontalalignment='center', verticalalignment='top')
                ax.text(x=x[i], y=1.01, s='True',
                        horizontalalignment='center',
                        verticalalignment='bottom')
            elif params[i] == 'params_use_sde_at_warmup':
                ax.text(x=x[i], y=-0.01, s='False',
                        horizontalalignment='center', verticalalignment='top')
                ax.text(x=x[i], y=1.01, s='True',
                        horizontalalignment='center',
                        verticalalignment='bottom')
            else:
                ax.text(x=x[i], y=-0.01,
                        s=np.round(df_study[params].min().values[i], 4),
                        horizontalalignment='center', verticalalignment='top')
                ax.text(x=x[i], y=1.01,
                        s=np.round(df_study[params].max().values[i], 4),
                        horizontalalignment='center',
                        verticalalignment='bottom')

        ax.set_xticks(x)
        ax.set_yticks([0, 1])
        ax.set_xticklabels([p[7:] for p in params], rotation=45, ha='right')
        ax.set_yticklabels([])
        ax.grid()

        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
        if show is False:
            plt.close()

    def custom_optimization_history_plot(self, df_study: pd.DataFrame,
                                         savepath: str = None,
                                         show: bool = True) -> None:
        '''custom implementation of the plot_optimization_history() function of
        optuna:
        https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_optimization_history.html#optuna.visualization.plot_optimization_history
        '''

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.scatter(df_study[df_study['state'] == 'COMPLETE']['number'],
                   df_study[df_study['state'] == 'COMPLETE']['value'],
                   s=30, alpha=0.7, color='grey', edgecolor='black',
                   label='COMPLETE', zorder=10)
        ax.scatter(df_study[df_study['state'] == 'FAIL']['number'],
                   np.full(len(df_study[df_study['state'] == 'FAIL']),
                           df_study['value'].min()),
                   s=30, alpha=0.7, color='red', edgecolor='black',
                   label='FAIL', zorder=10)
        ax.scatter(df_study[df_study['state'] == 'RUNNING']['number'],
                   np.full(len(df_study[df_study['state'] == 'RUNNING']),
                           df_study['value'].min()),
                   s=30, alpha=0.7, color='white', edgecolor='black',
                   label='RUNNING', zorder=10)
        ax.plot(df_study['number'], df_study['value'].ffill().cummax(),
                color='black')
        ax.grid(alpha=0.5)
        ax.set_xlabel('trial number')
        ax.set_ylabel('reward')
        ax.legend()

        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
        if show is False:
            plt.close()

    def custom_slice_plot(self, df_study: pd.DataFrame, params: list,
                          le_noise: LabelEncoder = None,
                          le_activation: LabelEncoder = None,
                          le_schedule: LabelEncoder = None,
                          savepath: str = None,
                          show: bool = True) -> None:
        '''custom implementation of the plot_slice() function of optuna:
        https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_slice.html#optuna.visualization.plot_slice
        '''
        fig = plt.figure(figsize=(20, 12))

        for i, param in enumerate(params):
            ax = fig.add_subplot(3, 6, i+1)
            ax.scatter(df_study[df_study['state'] == 'COMPLETE'][param],
                       df_study[df_study['state'] == 'COMPLETE']['value'],
                       s=20, color='grey', edgecolor='black', alpha=0.5,
                       label='COMPLETE')
            ax.scatter(df_study[df_study['state'] == 'FAIL'][param],
                       np.full(len(df_study[df_study['state'] == 'FAIL']),
                               df_study['value'].min()),
                       s=20, color='red', edgecolor='black', alpha=0.5,
                       label='FAIL')
            ax.grid(alpha=0.5)
            ax.set_xlabel(param[7:])
            if i == 0:
                ax.set_ylabel('reward')

            if 'learning_rate' in param:
                ax.set_xscale('log')
            if "noise" in param:
                plt.xticks(range(len(le_noise.classes_)),
                           le_noise.classes_)
            if "activation" in param:
                plt.xticks(range(len(le_activation.classes_)),
                           le_activation.classes_)
            if "lr_schedule" in param:
                plt.xticks(range(len(le_schedule.classes_)),
                           le_schedule.classes_)

        ax.legend()

        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
        if show is False:
            plt.close()

    def custom_intermediate_values_plot(self, agent: str, folder: str,
                                        mode: str = 'rollout',  # 'eval'
                                        print_thresh: int = None,
                                        y_high: int = 1000,
                                        y_low: int = -1000,
                                        savepath: str = None,
                                        show: bool = True) -> None:
        '''custom implementation of the plot_intermediate_values() function of
        optuna:
        https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_intermediate_values.html#optuna.visualization.plot_intermediate_values        https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_slice.html#optuna.visualization.plot_slice
        '''

        # only get trials of one agent type
        trials = [t for t in listdir(folder) if agent in t]
        num_trials = len(trials)

        fig, ax = plt.subplots(figsize=(10, 8))

        for trial in trials:
            try:
                df_log = pd.read_csv(f'{folder}/{trial}/progress.csv')
                n_strokes = df_log[r'rollout/ep_len_mean'].median()
                df_log['episodes'] = df_log[r'time/total_timesteps'] / n_strokes

                if mode == 'rollout':
                    ax.plot(df_log['episodes'], df_log[r'rollout/ep_rew_mean'],
                            alpha=0.3, color='black')
                elif mode == 'eval':
                    try:
                        df_log.dropna(axis=0,
                                      subset=['eval/mean_reward'],
                                      inplace=True)
                        if print_thresh is not None:
                            if df_log[r'eval/mean_reward'].max() > print_thresh:
                                print(trial, df_log[r'eval/mean_reward'].max())
                        ax.plot(df_log['episodes'],
                                df_log[r'eval/mean_reward'],
                                alpha=0.3, color='black')
                    except KeyError:
                        pass
            except EmptyDataError:
                pass

        ax.set_title(f"{agent} ({num_trials} trials)")

        ax.grid(alpha=0.5)
        ax.set_xlabel('episodes')
        ax.set_ylabel('reward')
        ax.set_ylim(top=y_high, bottom=y_low)

        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
        if show is False:
            plt.close()

    def training_progress_plot(self, df_log: pd.DataFrame,
                               df_env_log: pd.DataFrame,
                               savepath: str = None,
                               show: bool = True) -> None:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))
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

        # plotting environment statistics
        for logged_var in ["avg_replaced_cutters", "avg_moved_cutters",
                           'avg_inwards_moved_cutters',
                           'avg_wrong_moved_cutters',
                           "avg_broken_cutters"]:  # , "var_cutter_locations"]:
            ax2.plot(df_env_log["episodes"], df_env_log[logged_var],
                     label=logged_var)

        ax2.legend()
        ax2.grid(alpha=0.5)
        ax2.set_ylabel("count")

        # model specific visualization of loss
        for column in df_log.columns:
            if 'train' in column and 'loss' in column:
                ax3.plot(df_log['episodes'], df_log[column], label=column)

        ax3.legend()
        ax3.grid(alpha=0.5)
        ax3.set_xlabel('episodes')
        ax3.set_ylabel('loss')

        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
        if show is False:
            plt.close()

    def action_analysis_scatter_plotly(self, df: pd.DataFrame,
                                       savepath: str = None) -> None:
        '''plot that shows actions of a certain number of episodes that were
        projected into a 2D space to analyze the policy of the agent;
        plotly version'''
        fig = px.scatter(df, x='x', y='y', color='broken cutters',
                         hover_data={'x': False, 'y': False, 'state': True,
                                     'replaced cutters': True,
                                     'moved cutters': True})
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        fig.write_html(savepath)

    def action_analysis_scatter(self, df: pd.DataFrame, savepath: str = None,
                                show: bool = True) -> None:
        '''plot that shows actions of a certain number of episodes that were
        projected into a 2D space to analyze the policy of the agent;
        matplotlib version'''

        # scale x and y to 0-1 range for plotting reasons
        df['x'] = df['x']-df['x'].min()
        df['x'] = df['x']/df['x'].max()
        df['y'] = df['y']-df['y'].min()
        df['y'] = df['y']/df['y'].max()

        fig, ax = plt.subplots(figsize=(9, 9))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        im = ax.scatter(df['x'], df['y'], s=80, edgecolor=(0,0,0,.5),
                        c=df['avg. cutter life'], cmap='viridis')
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label(label='avg. cutter life', size=15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # annotation of some points
        np.random.seed(7)
        counter = 0
        ids = []
        p_xy_s = [[0, 0]]
        while counter < 8:
            i = np.random.choice(np.arange(len(df)), size=1)[0]
            p_x, p_y = df['x'].iloc[i], df['y'].iloc[i]
            # check if annotated point is not too close to edge of plot
            if p_x > 0.0 and p_x < 0.80 and p_y > 0.0 and p_y < 1.0:
                # check if annotated point is not too close to other annoted p.
                if np.min(np.linalg.norm(np.array(p_xy_s) - np.array([p_x, p_y]), axis=1)) > 0.2:
                    p_xy_s.append([p_x, p_y])
                    counter += 1
                    text = 'avg. c. life: {}\nc.broken: {}\nc.replaced: {}\n'\
                           'c.moved: {}\nreward: {}'.format(round(df['avg. cutter life'].iloc[i], 2),
                                                            df['broken cutters'].iloc[i],
                                                            df['replaced cutters'].iloc[i],
                                                            df['moved cutters'].iloc[i],
                                                            round(df['rewards'].iloc[i], 2))
                    # some logic to control annotation placement
                    if p_x <= 0.5 and p_y <= 0.5:
                        a_x, a_y, ha, va = -20, -20, 'right', 'top'
                    elif p_x > 0.5 and p_y <= 0.5:
                        a_x, a_y, ha, va = 20, -20, 'left', 'top'
                    elif p_x > 0.5 and p_y > 0.5:
                        a_x, a_y, ha, va = 20, 20, 'left', 'bottom'
                    else:
                        a_x, a_y, ha, va = -20, 20, 'right', 'bottom'
                    ax.annotate(text=text, xy=(p_x, p_y), zorder=10,
                                xytext=(a_x, a_y), textcoords='offset points',
                                arrowprops=dict(facecolor='black', shrink=0,
                                                width=1),
                                horizontalalignment=ha, verticalalignment=va,
                                bbox=dict(facecolor='white', edgecolor='black'))

        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
        if show is False:
            plt.close()
