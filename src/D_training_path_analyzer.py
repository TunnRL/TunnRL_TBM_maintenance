# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:32:03 2022

@author: GEr
"""

from XX_plotting import Plotter


agent = 'PPO'  # 'PPO' 'A2C' 'DDPG'
folder = 'optimization'  # 'checkpoints' 'optimization'
savepath = None  # 'graphics/PPO_trainings_default.svg'


if __name__ == "__main__":

    pltr = Plotter()
    pltr.custom_intermediate_values_plot(agent, folder='optimization')
