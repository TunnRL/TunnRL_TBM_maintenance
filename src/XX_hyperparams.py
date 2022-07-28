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

        self.A2C_defaults = {'A2C_learning rate': 0.0007,
                             'A2C_n steps': 5,
                             'A2C_discount': 0.99,
                             'A2C_gae lambda': 1.0,
                             'A2C_ent_coef': 0.0,
                             'A2C_vf coef': 0.5,
                             'A2C_max grad norm': 0.5,
                             'A2C_rms_prop_eps': 1e-05}

        self.DDPG_defaults = {'DDPG_learning rate': 0.001,
                              'DDPG_batch_size': 100,
                              'DDPG_learning starts': 100,
                              'DDPG_tau': 0.005,
                              'DDPG_discount': 0.99,
                              # 'DDPG_train_freq': (1, 'episode'),
                              'DDPG_gradient_steps': -1}

        self.TD3_defaults = {'TD3_learning rate': 0.001,
                             'TD3_learning starts': 100,
                             'TD3_batch_size': 100,
                             'TD3_tau': 0.005,
                             'TD3_discount': 0.99,
                             # 'TD3_train_freq': (1, 'episode'),
                             'TD3_gradient_steps': -1,
                             'TD3_action_noise': None,
                             'TD3_policy_delay': 2,
                             'TD3_target_policy_noise': 0.2,
                             'TD3_target_noise_clip': 0.5}

