"""
Towards optimized TBM cutter changing policies with reinforcement learning
G.H. Erharter, T.F. Hansen
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Custom library for plotting.

code contributors: Georg H. Erharter, Tom F. Hansen
"""

from pathlib import Path

import matplotlib.cm as mplcm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray


class Plotter:
    '''class that contains functions to visualzie the progress of the
    training and / or individual samples of it'''

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
