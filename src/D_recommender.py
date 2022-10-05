"""
Towards optimized TBM cutter changing policies with reinforcement learning
G.H. Erharter, T.F. Hansen
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Code that predict and action from a given state
TODO: make a streamlit version

code contributors: Tom F. Hansen
"""
import uuid

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from utils.XX_experiment_factory import load_best_model


@click.command()
@click.option(
    "--n_c_tot",
    default=41,
    show_default=True,
    type=int,
    help="Number of cutters on cutterhead.",
)
@click.option(
    "-dp",
    "--data_path",
    default="results/cutter_states.csv",
    show_default=True,
    type=click.Path(exists=True, readable=True, dir_okay=True),
    help="Path to data file with TBM cutter life data",
)
@click.option(
    "-a",
    "--agent_name",
    default="PPO",
    show_default=True,
    type=click.Choice(["PPO", "DDPG", "A2C", "TD3", "SAC"], case_sensitive=False),
)
@click.option(
    "-ap",
    "--agent_path",
    default="checkpoints/PPO_f7169560-3771-44b6-95f0-8456e45e23f6",
    show_default=True,
    type=click.Path(exists=True),
    help="Path to trained agent",
)
@click.option(
    "-sd",
    "--save_dir",
    default="graphics",
    show_default=True,
    type=click.Path(exists=True),
    help="Dir to save heatmap of actions",
)
def recommend(
    n_c_tot: int, data_path: str, agent_name: str, agent_path: str, save_dir: str
) -> NDArray:
    """Reading a state vector for the TBM-cutters from a csv-file and
    recommend the next actions.
    Saves a heatmap. Returns an array of actions"""
    observation = pd.read_csv(data_path, header=None).values[0]
    broken_cutters = np.where(observation == 0)[0]
    good_cutters = np.where(observation > 0)[0]

    print("Cutter statistics:")
    print("Observation: ", observation)
    print("Good cutters: ", len(good_cutters))
    print("Broken cutters: ", len(broken_cutters))

    main_dir = "/".join(agent_path.split("/")[0:-1])
    agent_dir = agent_path.split("/")[-1]

    agent = load_best_model(
        agent_name=agent_name, main_dir=main_dir, agent_dir=agent_dir
    )
    actions = agent.predict(observation, deterministic=False)[0]

    ac_reshaped = actions.reshape((n_c_tot, n_c_tot))
    fig, ax = plt.subplots()
    ax.imshow(ac_reshaped)
    save_path = f"{save_dir}/recommend_actions_{uuid.uuid4()}.svg"
    plt.savefig(save_path)

    return actions


if __name__ == "__main__":
    actions = recommend()
    print(actions)
