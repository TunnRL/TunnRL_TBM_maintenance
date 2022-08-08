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

    agents = dict(PPO=PPO(), A2C=A2C(), DDPG=DDPG(), SAC=SAC(), TD3=TD3())
    
    assert agent_name in agents.keys(), f"{agent_name} is not implemented."
    
    trained_agent = agents[agent_name].load(f'{main_dir}/{agent_dir}/best_model.zip')
    return trained_agent
