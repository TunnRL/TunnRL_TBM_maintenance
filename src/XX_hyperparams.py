# -*- coding: utf-8 -*-
"""
Collection of default hyperparameters for agents
Created on Sun Jul 24 16:19:57 2022

code contributors: Georg H. Erharter, Tom F. Hansen
"""


class parameters:

    def __init__(self):
        self.PPO_defaults = {'PPO_gae lambda': 0.95,
                             'PPO_learning rate': 0.0003,
                             'PPO_clip range': 0.2,
                             'PPO_normalize_advantage': True,
                             'PPO_discount': 0.99,
                             'PPO_ent_coef': 0.0,
                             'PPO_vf coef': 0.5,
                             'PPO_max grad norm': 0.5,
                             'PPO_use sde': False}

        self.SAC_defaults = {'SAC_action_noise': None,
                             'SAC_learning rate': 0.0003,
                             'SAC_learning starts': 100,
                             'SAC_batch_size': 256,
                             'SAC_discount': 0.99,
                             'SAC_tau': 0.005,
                             'SAC_train_freq': 1,
                             'SAC_gradient_steps': 1,
                             'SAC_ent_coef': 0.1,
                             'SAC_target_update_interval': 1,
                             'SAC_use sde': False,
                             'SAC_use_sde_at_warmup': False}
