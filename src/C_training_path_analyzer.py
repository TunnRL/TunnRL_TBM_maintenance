"""
Created on Tue Jul 26 13:32:03 2022

@author: Georg Erharter, Tom F. Hansen
"""

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import track
from rich.traceback import install
from utils.XX_general import parse_validate_hydra_config
from utils.XX_plotting import Plotter


def make_max_reward_list(root_directory: str, experiments_dir: str = None) -> None:
    """Find max reward value in each experiment and output a csv file with name of
    experiment directory and max reward value"""
    root_path = Path(root_directory, experiments_dir)
    filename = root_path.name.split("_")[0]
    output_path = Path(root_path.parent, f"{filename}_maxreward_experiment.csv")

    results = []

    print("First find number of experiments...")
    count = sum(entry.is_dir() for entry in root_path.iterdir())

    for directory in track(
        root_path.iterdir(),
        description="Finding max reward in each experiment",
        total=count,
    ):
        if directory.is_dir():
            csv_path = directory / "progress.csv"

            if csv_path.exists():
                df = pd.read_csv(csv_path)
                max_value = df["rollout/ep_rew_mean"].max()
                results.append([directory.name, max_value])

    result_df = pd.DataFrame(results, columns=["experiment_directory", "max_reward"])
    result_df.to_csv(output_path, index=False)


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3.2")
def main(cfg: DictConfig) -> None:
    """Plot training paths for exeriments, for one or many algorithms.

    In linux remember to mount P into wsl each time you restart the computer:
    sudo mount -t drvfs P: /mnt/P

    """

    console = Console()

    cfgs = parse_validate_hydra_config(cfg, console)

    # only make this list occasionally
    if cfgs.PLOT.MAKE_MAX_REWARD_LIST:
        experiments_dirs = [
            "PPO_2022_09_27_study",
            "DDPG_2022_10_03_study",
            "TD3_2022_09_27_study",
            "A2C_2022_11_30_study",
            "SAC_2022_10_05_study",
        ]
        for algorithm in experiments_dirs:
            make_max_reward_list(cfgs.PLOT.DATA_DIR, algorithm)

    for plot in cfgs.PLOT.PLOTS_TO_MAKE:
        match plot:
            case "training_path_experiments_single_algorithm":
                Plotter.custom_training_path_plot_algorithm(
                    agent=cfgs.PLOT.AGENT_NAME,
                    root_directory=cfgs.PLOT.DATA_DIR,
                    study_name=cfgs.PLOT.STUDY_NAME,
                    mode=cfgs.PLOT.VISUALIZATION_MODE,
                    print_thresh=cfgs.PLOT.PRINT_TRESH,
                    savepath=f"graphics/{cfgs.PLOT.AGENT_NAME}_learning_path_{cfgs.PLOT.VISUALIZATION_MODE}",
                    choose_num_best_rewards=cfgs.PLOT.CHOOSE_NUM_BEST_REWARDS,
                    filename_reward_list=f"{cfgs.PLOT.AGENT_NAME}_maxreward_experiment.csv",
                )
                console.print(
                    f"[green]Plotted learning path for {cfgs.PLOT.AGENT_NAME} \
                        using mode: {cfgs.PLOT.VISUALIZATION_MODE}"
                )

            case "training_path_experiments_algorithms":
                # plot a chosen set of algorithms
                algorithms = [
                    ("PPO", "PPO_2022_09_27_study", "red"),
                    # ("DDPG", "DDPG_2022_10_03_study", "green"),
                    # ("TD3","TD3_2022_09_27_study", "blue"),
                    ("A2C", "A2C_2022_11_30_study", "orange"),
                    ("SAC", "SAC_2022_10_05_study", "black"),
                ]

                Plotter.custom_training_path_plot_algorithms(
                    root_dir=cfgs.PLOT.DATA_DIR,
                    savepath=Path("graphics/learning_path_on_policy"),
                    algorithms=algorithms,
                    choose_num_best_rewards=cfgs.PLOT.CHOOSE_NUM_BEST_REWARDS,
                )
                console.print("[green]Plotted learning path for several algorithms")


if __name__ == "__main__":
    install()
    main()
