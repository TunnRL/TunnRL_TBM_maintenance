# -*- coding: utf-8 -*-
"""
Collection of default hyperparameters for agents
Created on Sun Jul 24 16:19:57 2022

code contributors: Georg H. Erharter, Tom F. Hansen
"""

from typing import Callable
from numpy.typing import NDArray
import torch.nn as nn
import optuna
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


class DefaultParameters:
    """Functionality to return a dictionary of default parameter values for a certain
    RL-architecture."""

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
        
        self.agent_dict = dict(
            PPO=self.PPO_defaults, DDPG=self.DDPG_defaults, A2C=self.A2C_defaults, TD3=self.TD3_defaults
            )
        
    def get_agent_default_params(self, agent_name: str) -> dict:
        """Return default parameters for a certain agent architecture."""
        return self.agent_dict[agent_name]


class Hyperparameters:
    """Class that bundle functionality to return a dictionary of suggested 
    hyperparameters in Optuna trials."""

    def suggest_hyperparameters(self, 
                                trial: optuna.trial.Trial, 
                                algorithm: str, 
                                environment, 
                                steps_episode: int,
                                num_actions: int) -> dict:
        """Hyperparameter suggestions for optuna optimization of a chosen RL-architecture.
        Each lookup of algorithm returns a dictionary of parameters for that algorithm."""
        
        # suggesting different network architectures for on_policy and off_policy networks
        net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "default", "shared"])
        network_archicture_on_policy = self.on_policy_networks(net_arch)
        
        net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big", "default"])
        network_archicture_off_policy = self.off_policy_networks(net_arch)
                
        # suggesting different activation functions
        activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]
        
        # computing action noise
        action_noise = trial.suggest_categorical(
                    'action_noise',[None, 'NormalActionNoise', "OrnsteinUhlenbeckActionNoise"])
        action_noise = self.yield_action_noise(action_noise, num_actions)
                
        match algorithm:
            case "PPO":
                # adjusting the learning rate scheduler
                learning_rate = trial.suggest_float('PPO_learning rate', low=1e-4, high=1e-3, log=True),
                learning_scheduler = trial.suggest_categorical("learning_scheduler", ["constant","linear_decrease"])
                if learning_scheduler == "linear_decrease":
                    learning_rate = self.linear_schedule(learning_rate)
                
                params = dict(
                    policy='MlpPolicy',
                    env=environment,
                    n_steps=steps_episode,
                    batch_size=50,
                    n_epochs=10,
                    gamma=trial.suggest_float('PPO_discount', low=0.6, high=1),
                    gae_lambda=trial.suggest_float('PPO_gae lambda', low=0.75, high=1),
                    clip_range=trial.suggest_float('PPO_clip range', low=0.1, high=0.45),
                    learning_rate=learning_rate,
                    normalize_advantage=True,
                    ent_coef=trial.suggest_float('PPO_ent_coef', low=0.0, high=0.3),
                    vf_coef=trial.suggest_float('PPO_vf coef', low=0.4, high=0.9),
                    max_grad_norm=trial.suggest_float('PPO_max grad norm', low=0.3, high=0.7),
                    use_sde=False,
                    verbose=0,
                    policy_kwargs=dict(net_arch=network_archicture_on_policy,
                                       activation_fn=activation_fn)
                )
            case "A2C":
                params = dict(
                    policy='MlpPolicy', 
                    env=environment,
                    learning_rate=trial.suggest_float('A2C_learning rate', low=1e-5, high=1e-1, log=True),
                    n_steps=trial.suggest_int('A2C_n steps', low=1, high=20, step=1),
                    gamma=trial.suggest_float('A2C_discount', low=0.0, high=1),
                    gae_lambda=trial.suggest_float('A2C_gae lambda', low=0.0, high=1),
                    ent_coef=trial.suggest_float('A2C_ent_coef', low=0.0, high=1),
                    vf_coef=trial.suggest_float('A2C_vf coef', low=0.0, high=1),
                    max_grad_norm=trial.suggest_float('A2C_max grad norm', low=0, high=1),
                    rms_prop_eps=trial.suggest_float('A2C_rms_prop_eps', low=1e-6, high=1e-3, log=True),
                    verbose=0,
                    policy_kwargs=dict(net_arch=network_archicture_on_policy)
                )
            case "DDPG":
                params = dict(
                    policy='MlpPolicy',
                    env=environment,
                    learning_rate=trial.suggest_float('DDPG_learning rate', low=1e-5, high=1e-2, log=True),
                    batch_size=trial.suggest_int('DDPG_batch_size', low=50, high=300, step=50),
                    learning_starts=trial.suggest_int('DDPG_learning starts', low=50, high=1000, step=50),
                    tau=trial.suggest_float('DDPG_tau', low=1e-4, high=1e-1, log=True),
                    gamma=trial.suggest_float('DDPG_discount', low=0.0, high=1),
                    action_noise=action_noise,
                    # train_freq=trial.suggest_int('DDPG_train_freq', low=1, high=5, step=1),
                    gradient_steps=trial.suggest_int('DDPG_gradient_steps', low=1, high=10, step=1),
                    verbose=0,
                    policy_kwargs=dict(net_arch=network_archicture_off_policy)
                )
            case "SAC":
                params = dict(
                    policy='MlpPolicy', 
                    env=environment,
                    learning_rate=trial.suggest_float('SAC_learning rate', low=1e-5, high=1e-2, log=True),
                    learning_starts=trial.suggest_int('SAC_learning starts', low=50, high=1000, step=50),
                    batch_size=trial.suggest_int('SAC_batch_size', low=50, high=300, step=50),
                    gamma=trial.suggest_float('SAC_discount', low=0.0, high=1),
                    tau=trial.suggest_float('SAC_tau', low=1e-4, high=1, log=True),
                    train_freq=trial.suggest_int('SAC_train_freq', low=1, high=10, step=1),
                    gradient_steps=trial.suggest_int('SAC_gradient_steps', low=1, high=10, step=1),
                    action_noise=action_noise,
                    ent_coef=trial.suggest_float('SAC_ent_coef', low=0.0, high=1),
                    target_update_interval=trial.suggest_int('SAC_target_update_interval', low=1, high=10),
                    use_sde=trial.suggest_categorical('SAC_use sde', [True, False]),
                    use_sde_at_warmup=trial.suggest_categorical('SAC_use_sde_at_warmup', [True, False]),
                    verbose=0
                )
            case "TD3":
                params = dict(
                    policy='MlpPolicy', 
                    env=environment,
                    learning_rate=trial.suggest_float('TD3_learning rate', low=1e-4, high=1e-1, log=True),
                    learning_starts=trial.suggest_int('TD3_learning starts', low=50, high=1000, step=50),
                    batch_size=trial.suggest_int('TD3_batch_size', low=50, high=300, step=50),
                    tau=trial.suggest_float('TD3_tau', low=1e-4, high=1e-1, log=True),
                    gamma=trial.suggest_float('TD3_discount', low=0.0, high=1),
                    # train_freq=trial.suggest_int('TD3_train_freq', low=1, high=10, step=1),
                    gradient_steps=trial.suggest_int('TD3_gradient_steps', low=1, high=10, step=1),
                    action_noise=action_noise,
                    policy_delay=trial.suggest_int('TD3_policy_delay', low=1, high=10, step=1),
                    target_policy_noise=trial.suggest_float('TD3_target_policy_noise', low=0.05, high=1),
                    target_noise_clip=trial.suggest_float('TD3_target_noise_clip', low=0.0, high=1),
                    verbose=0
                )
            case _:
                raise ValueError(f"{algorithm} is not implemented. Implemented algorithms are: PPO, DDPG, TD3, A2C, SAC")
        return params
    
    def yield_action_noise(self, action_noise: str, n_actions: int) -> NDArray:
        """Computes noise to used in action decisions.

        Args:
            action_noise (str): type of noise
            n_actions (int): number of actions in RL-setup

        Returns:
            NDArray: numpy array of noise
        """
        match action_noise:
            case "NormalActionNoise":
                noise = NormalActionNoise(
                    mean=np.zeros(n_actions),
                    sigma=0.1 * np.ones(n_actions))
            case "OrnsteinUhlenbeckActionNoise":
                noise = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(n_actions),
                    sigma=0.1 * np.ones(n_actions)
                )
            case _:
                noise = None
        return noise

    def linear_schedule(self, initial_value: float | str) -> Callable[[float], float]:
        """
        Linear learning rate scheduler.
        :param initial_value: (float or str)
        :return: (function)
        """
        if isinstance(initial_value, str):
            initial_value = float(initial_value)

            def func(progress_remaining: float) -> float:
                """
                Progress will decrease from 1 (beginning) to 0
                :param progress_remaining: (float)
                :return: (float)
                """
                return progress_remaining * initial_value

        return func
    
    def on_policy_networks(self, network_type: str) -> dict:
        """
        Defining several network architectures for use in on-policy-algorithms like
        PPO, A2C.
        
        Number of hidden layers with number of nodes in policy-network (pi) and value network (vf). 
        More info: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
        
        Default network is like the "small" network defined below, ie. no shared layers, both nets with 2 hidden layers of 64 nodes.
        Default network is very basic, just combined blocks of linear layers and activation functions.
        ie. no dropout, no batch normalization etc.
        More info about architecure here: https://github.com/DLR-RM/stable-baselines3/blob/646d6d38b6ba9aac612d4431176493a465ac4758/stable_baselines3/common/policies.py#L379
        And here: https://github.com/DLR-RM/stable-baselines3/blob/646d6d38b6ba9aac612d4431176493a465ac4758/stable_baselines3/common/torch_layers.py#L136
        """
        assert network_type in ["small", "medium", "shared", None], f"{network_type} is not a valide network"
        
        networks = {
            "small": [dict(pi=[64, 64], vf=[64, 64])],
            "medium": [dict(pi=[256, 256], vf=[256, 256])],
            "shared": [64, 64, dict(pi=[64, 64], vf=[64, 64])],
            "default": None,
        }
        return networks[network_type]

    def off_policy_networks(self, network_type: str) -> dict:
        """
        Defining several network architectures for use in off-policy-algorithms like
        DDPG, SAC, TD3.
        """
        assert network_type in ["small", "medium", None], f"{network_type} is not a valide network"

        networks = {
            "small": [64, 64],
            "medium": [256, 256],
            "big": [400, 300],
        }
        return networks[network_type]

