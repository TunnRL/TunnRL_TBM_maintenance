"""Functionality for general use."""

from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import track
from rich.theme import Theme
from sklearn.preprocessing import LabelEncoder

from tunnrl_tbm_maintenance.config_schemas import Config


def parse_validate_hydra_config(
    cfg: DictConfig, print_config: bool = True
) -> tuple[Config, Console]:
    """Parses and validates a hydra dictconfig with Pydantic.
    Prints evaluated config to terminal upon run."""
    styles_dict = Theme(
        {
            "info": "bold blue",
            "warning": "bold yellow",
            "error": "bold red",
            "critical": "bold red",
            "debug": "bold green",
            "success": "bold green",
        }
    )
    console = Console(theme=styles_dict)  # instantiate rich console
    OmegaConf.resolve(cfg)  # resolves hydra interpolation in place
    cfg_dict: dict[str, Any] = OmegaConf.to_object(cfg)  # type:ignore
    cfgs = Config(**cfg_dict)  # parse config general
    if print_config:
        console.print(OmegaConf.to_yaml(cfg))
    return cfgs, console


def latest_subdirectory(directory: Path) -> str | None:
    """
    Find the latest made subdirectory in a directory.

    Args:
        directory (Path): The path to the directory to search.

    Returns:
        Optional[str]: The name of the latest made subdirectory, or None if no
        subdirectory exists.
    """
    if not directory.is_dir():
        return None

    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    if not subdirs:
        return None

    latest_subdir = max(subdirs, key=lambda d: d.stat().st_mtime)
    return latest_subdir.name


def make_max_reward_list(
    root_directory: Path, algorithm: str, study_name: str | None = None
) -> pd.DataFrame:
    """Find max reward value in each experiment for an algorithm and output a csv file
    with name of experiment directory and max reward value.

    If experiments are grouped under a study name, only these experiments are considered.

    """
    if study_name is not None:
        root_directory = Path(root_directory, study_name)
        output_path = Path(
            root_directory, study_name, f"{study_name}_maxreward_experiment.csv"
        )
        experiment_directories = [d for d in root_directory.iterdir() if d.is_dir()]
    else:  # search for directories starting with the algorithm name
        root_directory = Path(root_directory)
        output_path = Path(root_directory, f"{algorithm}_maxreward_experiment.csv")
        experiment_directories = [
            d
            for d in root_directory.iterdir()
            if d.is_dir() and d.name.startswith(algorithm)
        ]

    results = []

    print("This function take some time to run.")
    print("First find number of experiments...")
    count = len(experiment_directories)

    for directory in track(
        experiment_directories,
        description=f"Finding max reward in each experiment for {algorithm}",
        total=count,
    ):
        if directory.is_dir():
            csv_path = directory / "progress.csv"

            if csv_path.exists():
                df = pd.read_csv(csv_path)
                max_value = df["rollout/ep_rew_mean"].max()
                # max_value = df["eval/mean_reward"].max()
                results.append([directory.name, max_value])

    result_df = pd.DataFrame(results, columns=["experiment_directory", "max_reward"])
    result_df.to_csv(output_path, index=False)
    return result_df


def process_optuna_data(study_name: str, agent: str, study_dirpath="results") -> tuple:
    """Process optuna data from optimization process, the study object.
    - Returns a processed dataframe that can be utilized in plotting and analysis.
    - Saves a yaml file with best performing parameters
    - Print optimalization info
    """
    console = Console()

    # LOAD DATA FROM COMPLETED OPTUNA STUDY
    #############################################################
    study_dirpath = f"{study_dirpath}/{study_name}.db"
    if not (Path(study_dirpath).exists()):
        raise ValueError(
            f"studyobject is not found at: {study_dirpath}. Perhaps at P:?"
        )
    db_file = f"sqlite:///{study_dirpath}"
    study = optuna.load_study(study_name=study_name, storage=db_file)

    df_study: pd.DataFrame = study.trials_dataframe()

    console.print("25 tail values before preprocessing\n", df_study.tail(n=25))

    # PRINT AND SAVE VALUES OF BEST TRIAL IN STUDY
    ################################################
    print("\nSaving parameters for 10 best trials to yaml files")
    df_tmp = df_study.copy()
    new_names = [param[7:] for param in df_tmp.columns if "params" in param]
    old_names = [param for param in df_tmp.columns if "params" in param]
    param_name_change = {old: new for old, new in zip(old_names, new_names)}
    df_tmp = df_tmp.rename(columns=param_name_change)

    df_tmp = df_tmp.sort_values("value", ascending=False)
    df_tmp = df_tmp.drop(columns=["datetime_complete", "datetime_start", "duration"])

    for i in range(10):
        params = df_tmp.iloc[i].to_dict()
        with open(
            f"results/{study_name}_best_params_{params['value']: .2f}.yaml", "w"
        ) as file:
            yaml.dump(params, file)

    trial = study.best_trial
    console.print("\nHighest reward: {}".format(trial.value))
    console.print("Best hyperparameters:\n {}".format(trial.params))

    # PREPROCESSING DATA
    ##############################################################
    if agent in ["SAC", "DDPG", "TD3"]:
        df_study["params_action_noise"].fillna(value="None", inplace=True)

    # dropping na values in running or failed runs
    print(f"Num row before dropna: {df_study.shape[0]}")
    df_study = df_study.dropna().reset_index(drop=True)
    print(f"Num row after dropna: {df_study.shape[0]}")

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

    params = [p for p in df_study.columns if "params_" in p]

    for param in params:
        if df_study[param].dtype == float and df_study[param].max() > 0.01:
            df_study[param] = df_study[param].round(2)

    # Convert specific columns to int
    columns_to_convert = []
    if agent in ["SAC", "DDPG", "TD3", "PPO-LSTM"]:
        columns_to_convert = [
            "params_n_nodes_layer",
            "params_n_not_shared_layers",
            "params_batch_size",
        ]
    else:
        columns_to_convert = [
            "params_n_nodes_layer",
            "params_n_nodes_shared_layer",
            "params_n_not_shared_layers",
            "params_n_shared_layers",
        ]
    df_study[columns_to_convert] = df_study[columns_to_convert].astype(int)

    df_study.to_excel(f"tmp/optuna_dataframe_{agent}.xlsx")  # tmp check

    return df_study, params, le_activation, le_noise
