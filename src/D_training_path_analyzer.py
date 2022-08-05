# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:32:03 2022

@author: GEr
"""

import matplotlib.pyplot as plt
from os import listdir
import pandas as pd
from pathlib import Path


agent = 'PPO'  # 'PPO' 'A2C' 'DDPG'
folder = 'optimization'  # 'checkpoints' 'optimization'


def training_path(agent, folder, savepath=None):
    maxs = []

    # plot of the progress of individual runs
    fig, ax = plt.subplots(figsize=(10, 8))

    for trial in listdir(folder):
        if agent in trial:
            df_log = pd.read_csv(Path(fr'{folder}/{trial}/progress.csv'))
            df_log['episodes'] = df_log[r'time/total_timesteps'] / df_log[r'rollout/ep_len_mean']
            df_log.dropna(axis=0, subset=[r'time/time_elapsed'], inplace=True)
            ax.plot(df_log['episodes'], df_log[r'rollout/ep_rew_mean'],
                    alpha=0.5, color='C0')
            maxs.append(df_log[r'rollout/ep_rew_mean'].max())

    ax.set_title(agent)

    # ax.set_ylim(top=1000, bottom=0)
    ax.grid(alpha=0.5)
    ax.set_xlabel('episodes')
    ax.set_ylabel('reward')
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()

    print(max(maxs))


if __name__ == "__main__":
    training_path(agent, folder,
                  savepath=Path(r'graphics/PPO_trainings_default.svg'))
