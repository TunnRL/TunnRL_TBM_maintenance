"""Analysis of training paths for experiments."""

from pathlib import Path

import hydra
from omegaconf import DictConfig
from rich.traceback import install

from tunnrl_tbm_maintenance.plotting import Plotter
from tunnrl_tbm_maintenance.utility import (
    make_max_reward_list,
    parse_validate_hydra_config,
)


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3.2")
def main(cfg: DictConfig) -> None:
    """Plot training paths for exeriments, for one or many algorithms.

    In linux remember to mount P into wsl each time you restart the computer:
    sudo mount -t drvfs P: /mnt/P

    """
    p_cfg, console = parse_validate_hydra_config(cfg, print_config=True)

    # only make this list occasionally
    if p_cfg.PLOT.MAKE_MAX_REWARD_LIST:
        for algorithm in p_cfg.OPT.STUDYS:
            make_max_reward_list(p_cfg.PLOT.DATA_DIR, algorithm)

    for plot in p_cfg.PLOT.PLOTS_TO_MAKE:
        match plot:
            case "training_progress_status_plot":
                import pandas as pd

                df_log = pd.read_csv(
                    Path(p_cfg.OPT.BEST_PERFORMING_ALGORITHM_PATH, "progress.csv")
                )
                df_log["episodes"] = (
                    df_log[r"time/total_timesteps"] / p_cfg.TBM.MAX_STROKES
                )
                df_env_log = pd.read_csv(
                    Path(p_cfg.OPT.BEST_PERFORMING_ALGORITHM_PATH, "progress_env.csv")
                )

                Plotter.training_progress_plot(
                    df_log=df_log,
                    df_env_log=df_env_log,
                    savepath=f"graphics/{p_cfg.PLOT.AGENT_NAME}_best_performing_progress",
                )
                console.print("[green]Plotted training progress status plot.")
            case "training_path_experiments_single_algorithm":
                Plotter.custom_training_path_plot_algorithm(
                    agent=p_cfg.PLOT.AGENT_NAME,
                    root_directory=p_cfg.PLOT.DATA_DIR,
                    study_name=p_cfg.PLOT.STUDY_NAME,
                    mode=p_cfg.PLOT.VISUALIZATION_MODE,
                    print_thresh=p_cfg.PLOT.PRINT_TRESH,
                    savepath=f"graphics/{p_cfg.PLOT.AGENT_NAME}_learning_path_{p_cfg.PLOT.VISUALIZATION_MODE}",
                    choose_num_best_rewards=p_cfg.PLOT.CHOOSE_NUM_BEST_REWARDS,
                    filename_reward_list=f"{p_cfg.PLOT.AGENT_NAME}_maxreward_experiment.csv",
                )
                console.print(
                    f"[green]Plotted learning path for {p_cfg.PLOT.AGENT_NAME} \
                        using mode: {p_cfg.PLOT.VISUALIZATION_MODE}"
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
                    root_dir=p_cfg.PLOT.DATA_DIR,
                    savepath=Path("graphics/learning_path_off_policy"),
                    algorithms=algorithms,
                    policy="off",  # on or off or all
                    choose_num_best_rewards=p_cfg.PLOT.CHOOSE_NUM_BEST_REWARDS,
                )
                console.print("[green]Plotted learning path for several algorithms")


if __name__ == "__main__":
    install()
    main()
