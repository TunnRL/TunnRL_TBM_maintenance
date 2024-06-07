"""Script that analyzes / visualizes the log of an OPTUNA hyperparameter study."""

import hydra
from omegaconf import DictConfig
from rich.traceback import install

from tunnrl_tbm_maintenance.plotting import Plotter
from tunnrl_tbm_maintenance.utility import (
    parse_validate_hydra_config,
    process_optuna_data,
)


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3.2")
def main(cfg: DictConfig) -> None:
    """Preprocessing and plotting optuna optimization data.

    In linux remember to mount P into wsl each time you restart the computer:
    sudo mount -t drvfs P: /mnt/P

    """
    p_cfg, console = parse_validate_hydra_config(cfg, print_config=True)

    for agent, study in zip(p_cfg.OPT.AGENTS, p_cfg.OPT.STUDYS):
        df_study, params, le_activation, le_noise = process_optuna_data(
            study, agent, p_cfg.PLOT.DATA_DIR
        )

        ###############################################################################
        # different visualizations of OPTUNA optimization

        for plot in p_cfg.PLOT.PLOTS_TO_MAKE:
            match plot:
                case "parallell_coordinate_plot":
                    console.print(f"[green]Plotting parallell coordinate plot: {study}")
                    Plotter.custom_parallel_coordinate_plot(
                        df_study,
                        params,
                        le_activation,
                        le_noise,
                        remove_negative_reward=False,
                        savepath=f"graphics/{study}_parallel_plot",
                    )

                case "optimization_history_plot":
                    console.print("[green]Plotting optimization history plot")
                    # plot that shows the progress of the optimization over the
                    # individual trials
                    Plotter.custom_optimization_history_plot(
                        df_study,
                        savepath=f"graphics/{study}_optimization_progress",
                    )

                case "slice_plot":
                    console.print("[green]Plotting custom slice plot")
                    # scatterplot of indivdual hyperparameters vs. reward
                    Plotter.custom_slice_plot(
                        df_study,
                        params,
                        le_activation=le_activation,
                        le_noise=le_noise,
                        savepath=f"graphics/{study}_optimization_scatter",
                    )


if __name__ == "__main__":
    install()
    main()