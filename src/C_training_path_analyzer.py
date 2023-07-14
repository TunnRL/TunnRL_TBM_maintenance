"""
Created on Tue Jul 26 13:32:03 2022

@author: Georg Erharter, Tom F. Hansen
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.traceback import install

from utils.XX_general import make_max_reward_list, parse_validate_hydra_config
from utils.XX_plotting import Plotter


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
                algorithms = dict(
                    on=[  # on-policy
                        ("PPO", "PPO_2022_09_27_study", "red"),
                        ("A2C", "A2C_2022_11_30_study", "orange"),
                        ("SAC", "SAC_2022_10_05_study", "black"),
                    ],
                    off=[  # off-policy
                        ("DDPG", "DDPG_2022_10_03_study", "green"),
                        ("TD3", "TD3_2022_09_27_study", "blue"),
                    ],
                    all=[
                        ("PPO", "PPO_2022_09_27_study", "red"),
                        ("A2C", "A2C_2022_11_30_study", "orange"),
                        ("SAC", "SAC_2022_10_05_study", "black"),
                        ("DDPG", "DDPG_2022_10_03_study", "green"),
                        ("TD3", "TD3_2022_09_27_study", "blue"),
                    ],
                )

                Plotter.custom_training_path_plot_algorithms(
                    root_dir=cfgs.PLOT.DATA_DIR,
                    savepath=Path("graphics/learning_path_off_policy"),
                    algorithms=algorithms,
                    policy="off",  # on or off or all
                    choose_num_best_rewards=cfgs.PLOT.CHOOSE_NUM_BEST_REWARDS,
                )
                console.print("[green]Plotted learning path for several algorithms")


if __name__ == "__main__":
    install()
    main()
