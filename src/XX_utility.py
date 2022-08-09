"""Utility functionality"""

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm


def load_best_model(agent_name: str, main_dir: str, agent_dir: str) -> BaseAlgorithm:
    """Load best model from a directory.

    Args:
        agent_name (str): name of RL-architecture (PPO, DDPG ...)
    """
    if agent_name == "DDP":
        agent_name = "DDPG"

    path = f'{main_dir}/{agent_dir}/best_model.zip'
    
    if agent_name == 'PPO':
        agent = PPO.load(path)
    elif agent_name == 'A2C':
        agent = A2C.load(path)
    elif agent_name == 'DDPG':
        agent = DDPG.load(path)
    elif agent_name == 'SAC':
        agent = SAC.load(path)
    elif agent_name == 'TD3':
        agent = TD3.load(path)

    return agent
