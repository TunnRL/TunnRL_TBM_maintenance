# -*- coding: utf-8 -*-
"""
Collection of default hyperparameters for agents
Created on Sun Jul 24 16:19:57 2022

code contributors: Georg H. Erharter, Tom F. Hansen
"""

from typing import Callable

import gym
import numpy as np
import optuna
import torch.nn as nn
from numpy.typing import NDArray
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)


class Hyperparameters:
    """Class that bundle functionality to return a dictionary of suggested
    hyperparameters in Optuna trials."""

    def suggest_hyperparameters(self,
                                trial: optuna.trial.Trial,
                                algorithm: str,
                                environment: gym.Env,
                                steps_episode: int,
                                num_actions: int) -> dict:
        """Hyperparameter suggestions for optuna optimization of a chosen
        RL-architecture.
        Each lookup of algorithm returns a dictionary of parameters for that
        algorithm."""

        # suggesting different network architectures
        # TODO: num nodes should increase by log
        num_layers = trial.suggest_int("num_layers", low=1, high=5, step=1)
        num_nodes_layer = trial.suggest_categorical("n_nodes_layer", [8, 16, 32, 64, 128, 256, 512])
        num_shared_layers = trial.suggest_int("num_shared_layers", low=0, high=3, step=1)
        num_nodes_shared_layer = trial.suggest_categorical("n_nodes_shared_layer", [8, 16, 32, 64, 128, 256, 512])

        network_architecture = self._define_policy_network(algorithm,
                                                           num_layers,
                                                           num_nodes_layer,
                                                           num_shared_layers,
                                                           num_nodes_shared_layer)

        # suggesting different activation functions
        activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu", "leaky_relu"])
        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[activation_fn]

        # computing action noise
        action_noise = trial.suggest_categorical('action_noise', [None, 'NormalActionNoise', "OrnsteinUhlenbeckActionNoise"])
        action_noise = self._yield_action_noise(action_noise, num_actions)

        match algorithm:
            case "PPO":
                # adjusting the learning rate scheduler
                # learning_rate = trial.suggest_float('learning rate', low=1e-4, high=1e-3, log=True),
                # learning_scheduler = trial.suggest_categorical("learning_scheduler", ["constant","linear_decrease"])
                # if learning_scheduler == "linear_decrease":
                #     learning_rate = self.linear_schedule(learning_rate)
                params = dict(
                    policy='MlpPolicy',
                    env=environment,
                    n_steps=steps_episode,
                    batch_size=50,
                    n_epochs=10,
                    gamma=trial.suggest_float('discount', low=0.6, high=1),
                    gae_lambda=trial.suggest_float('gae lambda', low=0.75, high=1),
                    clip_range=trial.suggest_float('clip range', low=0.1, high=0.45),
                    learning_rate=trial.suggest_float('learning rate', low=1e-4, high=1e-3, log=True),
                    normalize_advantage=True,
                    ent_coef=trial.suggest_float('ent_coef', low=0.0, high=0.3),
                    vf_coef=trial.suggest_float('vf coef', low=0.4, high=0.9),
                    max_grad_norm=trial.suggest_float('max grad norm', low=0.3, high=0.7),
                    use_sde=False,
                    verbose=0,
                    policy_kwargs=dict(net_arch=network_architecture,
                                       activation_fn=activation_fn)
                )
            case "A2C":
                params = dict(
                    policy='MlpPolicy',
                    env=environment,
                    learning_rate=trial.suggest_float('learning rate', low=1e-5, high=1e-1, log=True),
                    n_steps=trial.suggest_int('n steps', low=1, high=20, step=1),
                    gamma=trial.suggest_float('discount', low=0.0, high=1),
                    gae_lambda=trial.suggest_float('gae lambda', low=0.0, high=1),
                    ent_coef=trial.suggest_float('ent_coef', low=0.0, high=1),
                    vf_coef=trial.suggest_float('vf coef', low=0.0, high=1),
                    max_grad_norm=trial.suggest_float('max grad norm', low=0, high=1),
                    rms_prop_eps=trial.suggest_float('rms_prop_eps', low=1e-6, high=1e-3, log=True),
                    verbose=0,
                    policy_kwargs=dict(net_arch=network_architecture,
                                       activation_fn=activation_fn)
                )
            case "DDPG":
                params = dict(
                    policy='MlpPolicy',
                    env=environment,
                    learning_rate=trial.suggest_float('learning rate', low=1e-5, high=1e-2, log=True),
                    batch_size=trial.suggest_int('batch_size', low=50, high=300, step=50),
                    learning_starts=trial.suggest_int('learning starts', low=50, high=1000, step=50),
                    tau=trial.suggest_float('tau', low=1e-4, high=1e-1, log=True),
                    gamma=trial.suggest_float('discount', low=0.0, high=1),
                    action_noise=action_noise,
                    # train_freq=trial.suggest_int('train_freq', low=1, high=5, step=1),
                    gradient_steps=trial.suggest_int('gradient_steps', low=1, high=10, step=1),
                    verbose=0,
                    policy_kwargs=dict(net_arch=network_architecture)
                )
            case "SAC":
                params = dict(
                    policy='MlpPolicy',
                    env=environment,
                    learning_rate=trial.suggest_float('learning rate', low=1e-5, high=1e-2, log=True),
                    learning_starts=trial.suggest_int('learning starts', low=50, high=1000, step=50),
                    batch_size=trial.suggest_int('batch_size', low=50, high=300, step=50),
                    gamma=trial.suggest_float('discount', low=0.0, high=1),
                    tau=trial.suggest_float('tau', low=1e-4, high=1, log=True),
                    train_freq=trial.suggest_int('train_freq', low=1, high=10, step=1),
                    gradient_steps=trial.suggest_int('gradient_steps', low=1, high=10, step=1),
                    action_noise=action_noise,
                    ent_coef=trial.suggest_float('ent_coef', low=0.0, high=1),
                    target_update_interval=trial.suggest_int('target_update_interval', low=1, high=10),
                    use_sde=trial.suggest_categorical('use sde', [True, False]),
                    use_sde_at_warmup=trial.suggest_categorical('use_sde_at_warmup', [True, False]),
                    verbose=0,
                    policy_kwargs=dict(net_arch=network_architecture)
                )
            case "TD3":
                params = dict(
                    policy='MlpPolicy',
                    env=environment,
                    learning_rate=trial.suggest_float('learning rate', low=1e-4, high=1e-1, log=True),
                    learning_starts=trial.suggest_int('learning starts', low=50, high=1000, step=50),
                    batch_size=trial.suggest_int('batch_size', low=50, high=300, step=50),
                    tau=trial.suggest_float('tau', low=1e-4, high=1e-1, log=True),
                    gamma=trial.suggest_float('discount', low=0.0, high=1),
                    # train_freq=trial.suggest_int('train_freq', low=1, high=10, step=1),
                    gradient_steps=trial.suggest_int('gradient_steps', low=1, high=10, step=1),
                    action_noise=action_noise,
                    policy_delay=trial.suggest_int('policy_delay', low=1, high=10, step=1),
                    target_policy_noise=trial.suggest_float('target_policy_noise', low=0.05, high=1),
                    target_noise_clip=trial.suggest_float('target_noise_clip', low=0.0, high=1),
                    verbose=0,
                    policy_kwargs=dict(net_arch=network_architecture)
                )
            case _:
                raise ValueError(f"{algorithm} is not implemented. These algorithms are implemented: PPO, DDPG, TD3, A2C, SAC")

        # print(f"Training agent with these parameters:\n {params}")

        return params

    def _define_policy_network(self, algorithm: str = "PPO",
                               num_layers: int = 2, num_nodes_layer: int = 64,
                               num_shared_layers: int = 0,
                               num_nodes_shared_layer: int = 0) -> list:
        """Setting up a policy network.

        Concretely as an input to policy_kwargs{net_arch:<dict>} in the RL-agent
        method. Shared layers are only an option for on-policy networks, eg.
        PPO, A2C.

        Number of hidden layers with number of nodes in policy-network (pi) and
        value network (vf).
        More info: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html

        Default network is no shared layers, both nets with 2 hidden layers of 64 nodes. Default
        network is very basic, just combined blocks of linear layers and
        activation functions.ie. no dropout, no batch normalization etc.

        More info about architecure here: https://github.com/DLR-RM/stable-baselines3/blob/646d6d38b6ba9aac612d4431176493a465ac4758/stable_baselines3/common/policies.py#L379
        And here: https://github.com/DLR-RM/stable-baselines3/blob/646d6d38b6ba9aac612d4431176493a465ac4758/stable_baselines3/common/torch_layers.py#L136

        Returns:
            List: network description
        """
        assert algorithm in ["PPO", "A2C", "DDPG", "TD3", "SAC"], f"{algorithm} is not a valid algorithm"

        network_description = []

        if algorithm in ["PPO", "A2C"]:
            for _ in range(num_shared_layers):
                network_description.append(num_nodes_shared_layer)

        policy_network = []
        value_network = []
        for _ in range(num_layers):
            policy_network.append(num_nodes_layer)
            value_network.append(num_nodes_layer)

        network_description.append(dict(pi=policy_network, vf=value_network))
        return network_description

    def _yield_action_noise(self, action_noise: str,
                            n_actions: int) -> NDArray:
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
        TODO: Not in use. Do this in the callback instead

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
