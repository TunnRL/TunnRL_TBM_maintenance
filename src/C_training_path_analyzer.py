"""
Created on Tue Jul 26 13:32:03 2022

@author: Georg Erharter, Tom F. Hansen
"""

import click

from XX_plotting import Plotter


@click.command()
@click.option(
    "-a",
    "--agent_name",
    default="PPO",
    show_default=True,
    type=click.Choice(
        ["PPO", "DDPG", "A2C", "TD3", "SAC", "PPO-LSTM"], case_sensitive=False
    ),
)
@click.option(
    "-f",
    "--folder",
    default="checkpoints",
    show_default=True,
    type=click.Path(exists=True),
    help="Directory with training documentation. Values: training, optimization",
)
@click.option(
    "-sd",
    "--save_dir",
    default=None,
    show_default=True,
    type=click.Path(exists=False),
    help="Dir to save the graphic, eg: graphics/PPO_trainings_default.svg",
)
@click.option(
    "-vm",
    "--visualization_mode",
    default="rollout",
    show_default=True,
    type=click.Choice(["eval", "rollout"], case_sensitive=True),
)
@click.option(
    "-t",
    "--print_threshold",
    default=900,
    show_default=True,
    type=int,
    help="reward threshold to print trial name in VIS_MODE 'eval'",
)
@click.option(
    "-ylow",
    default=100,
    show_default=True,
    type=int,
    help="y low reward scale",
)
@click.option(
    "-yhigh",
    default=500,
    show_default=True,
    type=int,
    help="y high reward scale",
)
def main(
    agent_name: str,
    folder: str,
    save_dir: str,
    visualization_mode: str,
    print_threshold: int,
    ylow: int,
    yhigh: int,
) -> None:
    """Plot the training paths of all the trained models in a directory."""
    Plotter.custom_intermediate_values_plot(
        agent=agent_name,
        folder=folder,
        mode=visualization_mode,
        print_thresh=print_threshold,
        savepath=save_dir,
        y_low=ylow,
        y_high=yhigh,
    )


if __name__ == "__main__":
    main()
