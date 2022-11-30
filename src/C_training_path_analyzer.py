# -*- coding: utf-8 -*-
"""
Code for the paper:

Towards smart TBM cutter changing with reinforcement learning (working title)
Georg H. Erharter, Tom F. Hansen, Thomas Marcher, Amund Bruland
JOURNAL NAME
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Code that plots only intermediate values of multiple training runs from either
an ongoing OPTUNA study or multiple single training runs.

code contributors: Georg H. Erharter, Tom F. Hansen
"""

from XX_plotting import Plotter


AGENT = 'DDPG'  # 'PPO' 'A2C' 'DDPG' 'TD3' 'SAC'
FOLDER = 'optimization'  # 'checkpoints' 'optimization'
SAVEPATH = 'graphics/DDPG_2022_10_03_optimization_runs.svg'  # 'graphics/DDPG_2022_10_03_optimization_runs.svg' graphics/PPO_2022_09_27_optimization_runs.svg
VIS_MODE = 'eval'  # 'rollout' 'eval'
PRINT_THRESH = 900  # reward threshold to print trial name in VIS_MODE 'eval'
Y_LOW = -1000  # 100 -1000
Y_HIGH = 1000  # 600 200

if __name__ == "__main__":

    pltr = Plotter()
    pltr.custom_intermediate_values_plot(
        AGENT, folder=FOLDER, mode=VIS_MODE, print_thresh=PRINT_THRESH,
        y_low=Y_LOW, y_high=Y_HIGH, 
        savepath=SAVEPATH, show=False)
