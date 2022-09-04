# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:32:03 2022

@author: GEr
"""

from XX_plotting import Plotter


AGENT = 'TD3'  # 'PPO' 'A2C' 'DDPG' 'TD3' 'SAC'
FOLDER = 'optimization'  # 'checkpoints' 'optimization'
SAVEPATH = None  # 'graphics/PPO_trainings_default.svg'
VIS_MODE = 'eval'  # 'rollout' 'eval'
PRINT_THRESH = 900  # reward threshold to print trial name in VIS_MODE 'eval'

if __name__ == "__main__":

    pltr = Plotter()
    pltr.custom_intermediate_values_plot(AGENT, folder=FOLDER, mode=VIS_MODE,
                                         print_thresh=PRINT_THRESH)
