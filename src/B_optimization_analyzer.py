"""
Towards optimized TBM cutter changing policies with reinforcement learning
G.H. Erharter, T.F. Hansen
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Script that analyzes / visualizes the log of an OPTUNA study

Created on Thu Apr 14 13:28:07 2022
code contributors: Georg H. Erharter, Tom F. Hansen
"""

import click
import optuna
import pandas as pd
import yaml
from rich.traceback import install
from sklearn.preprocessing import LabelEncoder
from utils.XX_plotting import Plotter

install()


def process_optuna_data(study_name: str, agent: str, study_dirpath="results") -> tuple:
    """Process optuna data from optimization process.
    Returns a processed dataframe.
    """
    # load data from completed OPTUNA study
    # db_path = f"results/{study_name}.db"
    study_dirpath = f"{study_dirpath}/{study_name}.db"
    db_file = f"sqlite:///{study_dirpath}"
    study = optuna.load_study(study_name=study_name, storage=db_file)

    df_study: pd.DataFrame = study.trials_dataframe()

    print(df_study.tail(n=25))

    # some cleaning
    if "params_action_noise" in df_study.columns:
        le_noise = LabelEncoder()
        df_study["params_action_noise"] = le_noise.fit_transform(
            df_study["params_action_noise"]
        )
    else:
        le_noise = None
    if "params_activation_fn" in df_study.columns:
        le_activation = LabelEncoder()
        df_study["params_activation_fn"] = le_activation.fit_transform(
            df_study["params_activation_fn"]
        )

    # print values of best trial in study
    trial = study.best_trial
    print("\nHighest reward: {}".format(trial.value))
    print("Best hyperparameters:\n {}".format(trial.params))

    print("Saving best parameters to a yaml_file")
    with open(
        f"results/{study_name}_best_params_{study.best_value: .2f}.yaml", "w"
    ) as file:
        yaml.dump(study.best_params, file)

    params = [p for p in df_study.columns if "params_" in p]

    if agent == "SAC" or agent == "DDPG" or agent == "TD3":
        df_study["params_action_noise"].fillna(value="None", inplace=True)

    df_study = df_study.dropna().reset_index()

    df_study.to_excel("tmp/optuna_datafram.xlsx")

    return df_study, params, le_activation, le_noise


@click.command()
@click.option(
    "-sn",
    "--study_name",
    default="PPO_2022_09_27_study", show_default=True,
    help="Optuna study object with experiment information",
)
@click.option(
    "-sdp",
    "--study_dirpath",
    default="results", show_default=True,
    help="Directory path to directory containing study file. Eg. /mnt/P/2022/00/20220043/Calculations/",
)
def main(study_name: str) -> None:
    """Preprocessing and plotting optuna optimization data."""
    agent = study_name.split("_")[0]

    df_study, params, le_activation, le_noise = process_optuna_data(study_name, agent)

    ###############################################################################
    # different visualizations of OPTUNA optimization

    Plotter.custom_parallel_coordinate_plot(
        df_study,
        params,
        le_activation,
        savepath=f"graphics/{study_name}_parallel_plot.svg",
    )

    # plot that shows the progress of the optimization over the individual trials
    Plotter.custom_optimization_history_plot(
        df_study, savepath=f"graphics/{study_name}_optimization_progress.svg"
    )

    # scatterplot of indivdual hyperparameters vs. reward
    Plotter.custom_slice_plot(
        df_study,
        params,
        le_activation=le_activation,
        le_noise=le_noise,
        savepath=f"graphics/{study_name}_optimization_scatter.svg",
    )

    # plot intermediate steps of the training paths
    Plotter.custom_intermediate_values_plot(
        agent,
        folder="optimization",
        savepath=f"graphics/{study_name}_optimization_interms.svg",
    )


if __name__ == "__main__":
    main()
