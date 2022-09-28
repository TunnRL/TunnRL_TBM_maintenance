# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:32:03 2022

@author: GEr
"""

from XX_plotting import Plotter


AGENT = 'PPO'  # 'PPO' 'A2C' 'DDPG' 'TD3' 'SAC'
FOLDER = 'optimization'  # 'checkpoints' 'optimization'
SAVEPATH = 'graphics/PPO_optimization_exp1.svg'
VIS_MODE = 'rollout'  # 'rollout' 'eval'
PRINT_THRESH = 900  # reward threshold to print trial name in VIS_MODE 'eval'
Y_LOW = 100
Y_HIGH = 600

if __name__ == "__main__":

    pltr = Plotter()
    pltr.custom_intermediate_values_plot(
        AGENT, folder=FOLDER, mode=VIS_MODE, print_thresh=PRINT_THRESH,
        y_low=Y_LOW, y_high=Y_HIGH, 
        savepath=SAVEPATH, show=False)
