"""
Towards optimized TBM cutter changing policies with reinforcement learning
G.H. Erharter, T.F. Hansen
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Code that predict and action from a given state
TODO: make a streamlit version

code contributors: Tom F. Hansen
"""
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from XX_experiment_factory import load_best_model


def recommend(
    data_path: str,
    agent_name: str,
    agent_main_dir: str,
    agent_dir: str,
    save_path: str,
) -> None:
    """Reading a state vector for the TBM-cutters from a csv-file and
    recommend the next actions."""
    observation = pd.read_csv(data_path, header=None).values[0]
    broken_cutters = np.where(observation == 0)[0]
    good_cutters = np.where(observation > 0)[0]

    print("Cutter statistics:")
    print("Observation: ", observation)
    print("Good cutters: ", good_cutters)
    print("Broken cutters: ", broken_cutters)

    agent = load_best_model(
        agent_name=agent_name, main_dir=agent_main_dir, agent_dir=agent_dir
    )
    actions = agent.predict(observation, deterministic=False)[0]

    ac_reshaped = actions.reshape((N_C_TOT, N_C_TOT))
    fig, ax = plt.subplots()
    ax.imshow(ac_reshaped)
    plt.savefig(save_path)

    return actions


if __name__ == "__main__":
    MAIN_DIR = "checkpoints"  # 'checkpoints' 'optimization'
    AGENT_DIR = "PPO_f7169560-3771-44b6-95f0-8456e45e23f6"
    SAVEPATH = f"graphics/recommend_actions_{uuid.uuid4()}.svg"
    DATAPATH = "results/cutter_states.csv"
    N_C_TOT = 28
    agent_name = AGENT_DIR.split("_")[0]  # 'PPO' 'A2C' 'DDPG' 'TD3' 'SAC'

    actions = recommend(DATAPATH, agent_name, MAIN_DIR, AGENT_DIR, SAVEPATH)
    print(actions)
