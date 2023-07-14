"""
Towards optimized TBM cutter changing policies with reinforcement learning
G.H. Erharter, T.F. Hansen
DOI: XXXXXXXXXXXXXXXXXXXXXXXXXX

Custom library for plotting.

code contributors: Georg H. Erharter, Tom F. Hansen
"""

from collections import Counter
from itertools import chain
from pathlib import Path

import gymnasium as gym
import matplotlib
import matplotlib as mpl
import matplotlib.cm as mplcm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import NDArray
from pandas.errors import EmptyDataError
from rich.console import Console
from rich.progress import track
from sklearn.preprocessing import LabelEncoder

# HELPER FUNCTIONS
#######################################################


def print_function_name(func):
    def wrapper(*args, **kwargs):
        console = Console()
        console.print(f"Function Name: {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


def plot_text(
    x_val: float,
    y_val: float,
    text_string: str,
    horizontal_align: str,
    vertical_align: str,
) -> None:
    """
    Plots the given text on the current axes at the specified coordinates.

    Args:
        x_val: X-coordinate for the text position.
        y_val: Y-coordinate for the text position.
        text_string: The text to be displayed.
        horizontal_align: Horizontal alignment for the text. Can be 'center', 'left',
        or 'right'.
        vertical_align: Vertical alignment for the text. Can be 'top', 'center', or
        'bottom'.

    Returns:
        None
    """
    ax = plt.gca()
    ax.text(
        x=x_val,
        y=y_val,
        s=text_string,
        horizontalalignment=horizontal_align,
        verticalalignment=vertical_align,
    )


# PLOTTING FUNCTIONS
#######################################################


class Plotter:
    """class that contains functions to visualize the progress of the
    training and / or individual samples of it

    Configuring of plots:
    - General rule: use config in figures_styles.mplstyle
      Eg. dpi and image-type is controlled in config
    - execptions for singular plots with matplotlib decorator:
    Eg. @mpl.rc_context({'lines.linewidth': 1.1, 'lines.markersize': 2, 'font.size':6})

    """

    plt.style.use("./src/config/figures_styles.mplstyle")
    FIGURE_WIDTH = 3.15

    @staticmethod
    def sample_ep_plot(
        states: list[NDArray],
        actions: list[NDArray],
        rewards: list[NDArray],
        replaced_cutters: list,
        moved_cutters: list,
        n_cutters: int,
        savepath: str | None = None,
        show: bool = True,
    ) -> None:
        """plot of different recordings of one exemplary episode"""

        n_replaced_cutters: list[int] = [len(cutters) for cutters in replaced_cutters]
        n_moved_cutters: list[int] = [len(cutters) for cutters in moved_cutters]
        strokes = np.arange(len(moved_cutters))

        cmap = mplcm.get_cmap("viridis")

        states_arr = np.vstack(states[:-1])
        actions_arr = np.vstack(actions)
        actions_arr = np.where(actions_arr > 0, 1, 0)  # binarize

        fig = plt.figure(tight_layout=True, figsize=(10.236, 7.126))
        gs = gridspec.GridSpec(nrows=4, ncols=2, width_ratios=[5, 1])

        # cutter life line plot
        ax = fig.add_subplot(gs[0, 0])
        for cutter in range(states_arr.shape[1]):
            rgba = cmap(cutter / states_arr.shape[1])
            ax.plot(strokes, states_arr[:, cutter], color=rgba, label=cutter)
        h_legend, l_legend = ax.get_legend_handles_labels()
        ax.set_xlim(left=-1, right=actions_arr.shape[0] + 1)
        ax.set_ylim(top=1.05, bottom=-0.05)
        ax.set_title("episode episode", fontsize=10)
        ax.set_ylabel("cutter life")
        ax.set_xticklabels([])
        ax.grid(alpha=0.5)

        # dedicated subplot for a legend to first axis
        lax = fig.add_subplot(gs[0, 1])
        id_middle = int(len(l_legend) / 2)
        lax.legend(
            list(np.take(h_legend, [0, id_middle, -1])),
            list(np.take(l_legend, [0, id_middle, -1])),
            borderaxespad=0,
            ncol=1,
            loc="upper left",
            fontsize=7.5,
            title="cutter positions",
        )
        lax.axis("off")

        # bar plot that shows how many cutters were moved
        ax = fig.add_subplot(gs[1, 0])
        ax.bar(x=strokes, height=n_moved_cutters, color="grey")
        avg_changed = np.mean(n_moved_cutters)
        ax.axhline(y=avg_changed, color="black")
        ax.text(
            x=950,
            y=avg_changed - avg_changed * 0.05,
            s=f"avg. moved cutters / stroke: {round(avg_changed, 2)}",
            color="black",
            va="top",
            ha="right",
            fontsize=7.5,
        )
        ax.set_xlim(left=-1, right=actions_arr.shape[0] + 1)
        ax.set_ylabel("cutter moves\nper stroke")
        ax.set_xticklabels([])
        ax.grid(alpha=0.5)

        # bar plot that shows n replacements over episode
        moved_count = Counter(chain(*map(set, moved_cutters)))
        ax = fig.add_subplot(gs[1, 1])
        ax.bar(
            x=list(moved_count.keys()),
            height=list(moved_count.values()),
            color="grey",
            edgecolor="black",
        )
        ax.set_xlim(left=0, right=n_cutters)
        ax.grid(alpha=0.5)
        ax.set_xlabel("n cutter moves per\nposition")

        # bar plot that shows how many cutters were replaced
        ax = fig.add_subplot(gs[2, 0])
        ax.bar(x=strokes, height=n_replaced_cutters, color="grey")
        avg_changed = np.mean(n_replaced_cutters)
        ax.axhline(y=avg_changed, color="black")
        ax.text(
            x=950,
            y=avg_changed - avg_changed * 0.05,
            s=f"avg. replacements / stroke: {round(avg_changed, 2)}",
            color="black",
            va="top",
            ha="right",
            fontsize=7.5,
        )
        ax.set_xlim(left=-1, right=actions_arr.shape[0] + 1)
        ax.set_ylabel("cutter replacements\nper stroke")
        ax.set_xticklabels([])
        ax.grid(alpha=0.5)

        # bar plot that shows n replacements over episode
        replaced_count = Counter(chain(*map(set, replaced_cutters)))
        ax = fig.add_subplot(gs[2, 1])
        ax.bar(
            x=list(replaced_count.keys()),
            height=list(replaced_count.values()),
            color="grey",
            edgecolor="black",
        )
        ax.set_xlim(left=0, right=n_cutters)
        ax.grid(alpha=0.5)
        ax.set_xlabel("n cutter replacements per\nposition")

        # plot that shows the reward per stroke
        ax = fig.add_subplot(gs[3, 0])
        ax.scatter(x=strokes, y=rewards, color="grey", s=1)
        ax.axhline(y=np.mean(rewards), color="black")
        ax.text(
            x=950,
            y=np.mean(rewards) - 0.05,
            s=f"avg. reward / stroke: {round(np.mean(rewards), 2)}",
            color="black",
            va="top",
            ha="right",
            fontsize=7.5,
        )
        ax.set_ylim(bottom=-1.05, top=1.05)
        ax.set_xlim(left=-1, right=actions_arr.shape[0] + 1)
        ax.set_ylabel("reward / stroke")
        ax.set_xlabel("strokes")
        ax.grid(alpha=0.5)

        plt.tight_layout()
        plt.savefig(savepath)

    @staticmethod
    def state_action_plot(
        states: list,
        actions: list,
        n_strokes: int,
        n_c_tot: int,
        rewards: list,
        savepath: str = None,
        show: bool = True,
    ) -> None:
        """plot that shows combinations of states and actions for the first
        n_strokes of an episode"""
        fig = plt.figure(figsize=(20, 9))

        ax = fig.add_subplot(311)
        ax.imshow(
            np.vstack(states[:n_strokes]).T,
            aspect="auto",
            interpolation="none",
            vmin=0,
            vmax=1,
            cmap="cividis",
        )
        ax.set_yticks(np.arange(-0.5, n_c_tot), minor=True)
        ax.set_xticks(np.arange(-0.5, n_strokes), minor=True)

        ax.set_xticks(np.arange(n_strokes), minor=False)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="minor", color="white")
        ax.tick_params(axis="x", which="major", length=10, color="lightgrey")

        ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
        ax.set_xlim(left=-1, right=n_strokes)
        ax.set_ylim(bottom=-0.5, top=n_c_tot - 0.5)
        ax.set_yticks
        ax.set_ylabel("cutter states on\ncutter positions")

        ax = fig.add_subplot(312)

        for stroke in track(
            range(n_strokes), description="Processing strokes in state_action_plot"
        ):
            for i in range(n_c_tot):
                # select cutter from action vector
                cutter = actions[stroke][i * n_c_tot : i * n_c_tot + n_c_tot]
                if np.max(cutter) < 0.9:
                    # cutter is not acted on
                    pass
                elif np.argmax(cutter) == i:
                    # cutter is replaced
                    ax.scatter(stroke, i, edgecolor="black", color="black", zorder=50)
                else:
                    # cutter is moved from original position to somewhere else
                    # original position of old cutter that is replaced
                    ax.scatter(stroke, i, edgecolor=f"C{i}", color="black", zorder=20)
                    # new position where old cutter is moved to
                    ax.scatter(
                        stroke,
                        np.argmax(cutter),
                        edgecolor=f"C{i}",
                        color=f"C{i}",
                        zorder=20,
                    )
                    # arrow / line that connects old and new positions
                    ax.arrow(
                        x=stroke,
                        y=i,
                        dx=0,
                        dy=-(i - np.argmax(cutter)),
                        color=f"C{i}",
                        zorder=10,
                    )
        ax.set_yticks(np.arange(n_c_tot), minor=True)
        ax.set_xticks(np.arange(n_strokes), minor=False)
        ax.tick_params(axis="x", which="minor", color="white")
        ax.tick_params(axis="x", which="major", length=10, color="lightgrey")
        ax.set_xticklabels([])
        ax.set_xlim(left=-1, right=n_strokes)
        ax.grid(zorder=0, which="both", color="grey")
        ax.set_ylabel("actions on\ncutter positions")

        ax = fig.add_subplot(313)
        ax.scatter(x=np.arange(n_strokes), y=rewards[:n_strokes], color="grey")
        ax.set_xticks(np.arange(n_strokes), minor=True)
        ax.set_xlim(left=-1, right=n_strokes)
        ax.set_ylim(bottom=-1.05, top=1.05)
        ax.grid(zorder=0, which="both", color="grey")
        ax.set_ylabel("reward per stroke")
        ax.set_xlabel("strokes")

        plt.tight_layout(h_pad=0)
        if savepath is not None:
            plt.savefig(savepath)
        if show is False:
            plt.close()

    @staticmethod
    def environment_parameter_plot(
        ep: int, env: gym.Env, savepath: str = None, show: bool = True
    ) -> None:
        """plot that shows the generated TBM parameters of the episode"""
        x = np.arange(len(env.Jv_s))  # strokes
        # count broken cutters due to blocky conditions
        n_brokens = np.count_nonzero(env.brokens, axis=1)

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6, figsize=(12, 9))

        ax1.plot(x, env.Jv_s, color="black")
        ax1.grid(alpha=0.5)
        ax1.set_ylabel("Volumetric Joint count\n[joints / m3]")
        ax1.set_xlim(left=0, right=len(x))
        ax1.set_title(f"episode {ep}", fontsize=10)
        ax1.set_xticklabels([])

        ax2.plot(x, env.UCS_s, color="black")
        ax2.grid(alpha=0.5)
        ax2.set_ylabel("Rock UCS\n[MPa]")
        ax2.set_xlim(left=0, right=len(x))
        ax2.set_xticklabels([])

        ax3.plot(x, env.FPIblocky_s, color="black")
        ax3.hlines([50, 100, 200, 300], xmin=0, xmax=len(x), color="black", alpha=0.5)
        ax3.set_ylim(bottom=0, top=400)
        ax3.set_ylabel("FPI blocky\n[kN/m/mm/rot]")
        ax3.set_xlim(left=0, right=len(x))
        ax3.set_xticklabels([])

        ax4.plot(x, env.TF_s, color="black")
        ax4.grid(alpha=0.5)
        ax4.set_ylabel("thrust force\n[kN]")
        ax4.set_xlim(left=0, right=len(x))
        ax4.set_xticklabels([])

        ax5.plot(x, env.penetration, color="black")
        ax5.grid(alpha=0.5)
        ax5.set_ylabel("penetration\n[mm/rot]")
        ax5.set_xlim(left=0, right=len(x))
        ax5.set_xticklabels([])

        ax6.plot(x, n_brokens, color="black")
        ax6.grid(alpha=0.5)
        ax6.set_ylabel("broken cutters\ndue to blocks")
        ax6.set_xlabel("strokes")
        ax6.set_xlim(left=0, right=len(x))

        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
        if show is False:
            plt.close()

    @staticmethod
    def action_visualization(
        action: NDArray, n_c_tot: int, savepath: str | None = None, binary: bool = False
    ) -> None:
        """plot that visualizes a single action"""
        if binary is True:
            action = np.where(action > 0.9, 1, -1)

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax1.imshow(np.reshape(action, (n_c_tot, n_c_tot)), vmin=-1, vmax=1)
        ax1.set_xticks(np.arange(-0.5, n_c_tot), minor=True)
        ax1.set_yticks(np.arange(-0.5, n_c_tot), minor=True)
        ax1.grid(which="minor", color="black", linestyle="-", linewidth=1)
        ax1.set_xlabel("cutters to move to")
        ax1.set_ylabel("cutters to acton on")
        fig.colorbar(im, cax=cax, orientation="vertical")
        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
            plt.close()

    @staticmethod
    @mpl.rc_context({"lines.linewidth": 1.1, "lines.markersize": 2, "font.size": 7})
    def custom_parallel_coordinate_plot(
        df_study: pd.DataFrame,
        params: list,
        le_activation: LabelEncoder,
        le_noise: LabelEncoder,
        remove_negative_reward: bool,
        savepath: str,
        figure_width=FIGURE_WIDTH,
    ) -> None:
        """custom implementation of the plot_parallel_coordinate() function of
        optuna:
        https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_parallel_coordinate.html#optuna.visualization.plot_parallel_coordinate
        """

        # dropping runs with values below zero
        if remove_negative_reward:
            print(f"Num row before removing negative values: {df_study.shape[0]}")
            df_study = df_study[df_study.value > 0]
            print(f"Num row after removing negative values: {df_study.shape[0]}")

        df_study["params_lr_schedule"] = np.where(
            df_study["params_lr_schedule"] == "constant", 0, 1
        )

        fig, ax = plt.subplots(figsize=(figure_width * 2.5, 0.6 * figure_width))

        mins = df_study[params].min().values
        f = df_study[params].values - mins
        maxs = np.max(f, axis=0)

        cmap = mpl.cm.get_cmap("cividis")
        norm = mpl.colors.Normalize(
            vmin=df_study["value"].min(), vmax=df_study["value"].max()
        )
        x = np.arange(len(params))

        # plotting the lines
        for t in range(len(df_study)):
            df_row = df_study.sort_values(by="value").iloc[t]
            y = df_row[params].values

            y = y - mins
            try:
                y = y / maxs
            except ZeroDivisionError:
                # print(f"maxs has value: {maxs}")
                print(
                    "ZeroDivisionError for one of the columns. Check values of maxs\
                      .This will lead to no plotting of lines"
                )
                continue

            if df_row["state"] == "FAIL":
                ax.plot(x, y, c="red", alpha=0.5)
            elif df_row["state"] == "RUNNING":
                pass
            else:
                if df_row["value"] < 600:
                    ax.plot(x, y, c=cmap(norm(df_row["value"])), alpha=0.2)
                else:
                    ax.plot(x, y, c=cmap(norm(df_row["value"])), alpha=1, zorder=10)

        # plotting the black points at bottom and top in plot
        ax.scatter(x, np.zeros(x.shape), color="black")
        ax.scatter(x, np.ones(x.shape), color="black")

        # plotting the colorbar
        # Add colorbar, specifying the mappable object and the axes to attach to
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # This line is necessary for the colorbar to appear
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("Reward")

        # plotting the text
        for param, x_val in zip(params, x):
            if param == "params_action_noise":
                noise_cats = [noise.split("Action")[0] for noise in le_noise.classes_]
                plot_text(x_val, -0.01, noise_cats[0], "center", "top")
                plot_text(x_val, 0.5, noise_cats[1], "center", "top")
                plot_text(x_val, 1.01, noise_cats[2], "center", "bottom")

            elif param == "params_activation_fn":
                plot_text(x_val, -0.01, le_activation.classes_[0], "center", "top")
                plot_text(x_val, 0.5, le_activation.classes_[1], "right", "top")
                plot_text(x_val, 1.01, le_activation.classes_[2], "center", "bottom")

            elif param == "params_lr_schedule":
                plot_text(x_val, -0.01, "constant", "center", "top")
                plot_text(x_val, 1.01, "lin. decr.", "center", "bottom")

            elif df_study[param].max() < 0.01:
                plot_text(x_val, -0.01, f"{df_study[param].min():.2E}", "center", "top")
                plot_text(
                    x_val, 1.01, f"{df_study[param].max():.2E}", "center", "bottom"
                )
            else:
                plot_text(x_val, -0.01, f"{df_study[param].min()}", "center", "top")
                plot_text(x_val, 1.01, f"{df_study[param].max()}", "center", "bottom")

        ax.set_xticks(x)
        ax.set_yticks([0, 1])
        ax.set_xticklabels([p[7:] for p in params], rotation=45, ha="center")
        ax.set_yticklabels([])
        ax.set_ylim(bottom=-0.1, top=1.1)
        ax.grid()
        score = df_study["value"].max()
        plt.title(
            f'{savepath.split("/")[1].split("_")[0]}. Best reward score: {score: .0f}'
        )

        # plt.tight_layout()
        plt.savefig(savepath, bbox_inches="tight")

    @staticmethod
    @mpl.rc_context({"lines.linewidth": 1.1})
    def custom_optimization_history_plot(
        df_study: pd.DataFrame,
        savepath: str,
        figure_width=FIGURE_WIDTH,
    ) -> None:
        """custom implementation of the plot_optimization_history() function of
        optuna:
        https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_optimization_history.html#optuna.visualization.plot_optimization_history
        """

        fig, ax = plt.subplots(figsize=(figure_width, figure_width))

        ax.scatter(
            df_study.loc[df_study["state"] == "COMPLETE", "number"],
            df_study.loc[df_study["state"] == "COMPLETE", "value"],
            s=30,
            alpha=0.7,
            color="grey",
            edgecolor="black",
            label="COMPLETE",
            zorder=10,
        )
        if (df_study["state"] == "FAIL").sum() > 0:
            ax.scatter(
                df_study[df_study["state"] == "FAIL"]["number"],
                np.full(
                    len(df_study[df_study["state"] == "FAIL"]), df_study["value"].min()
                ),
                s=30,
                alpha=0.7,
                color="red",
                edgecolor="black",
                label="FAIL",
                zorder=10,
            )
        if (df_study["state"] == "RUNNING").sum() > 0:
            ax.scatter(
                df_study[df_study["state"] == "RUNNING"]["number"],
                np.full(
                    len(df_study[df_study["state"] == "RUNNING"]),
                    df_study["value"].min(),
                ),
                s=30,
                alpha=0.7,
                color="white",
                edgecolor="black",
                label="RUNNING",
                zorder=10,
            )
        ax.plot(df_study["number"], df_study["value"].ffill().cummax(), color="red")
        ax.grid(alpha=0.5)
        ax.set_xlabel("trial number")
        ax.set_ylabel("reward")
        ax.legend()

        plt.title(savepath.split("/")[1].split("_")[0])
        plt.tight_layout()
        plt.savefig(savepath)

    @staticmethod
    def custom_slice_plot(
        df_study: pd.DataFrame,
        params: list,
        le_noise: LabelEncoder = None,
        le_activation: LabelEncoder = None,
        savepath: str = None,
        show: bool = True,
    ) -> None:
        fig = plt.figure(figsize=(20, 12))

        for i, param in enumerate(params):
            ax = fig.add_subplot(3, 6, i + 1)
            ax.scatter(
                df_study[df_study["state"] == "COMPLETE"][param],
                df_study[df_study["state"] == "COMPLETE"]["value"],
                s=20,
                color="grey",
                edgecolor="black",
                alpha=0.5,
                label="COMPLETE",
            )
            ax.scatter(
                df_study[df_study["state"] == "FAIL"][param],
                np.full(
                    len(df_study[df_study["state"] == "FAIL"]), df_study["value"].min()
                ),
                s=20,
                color="red",
                edgecolor="black",
                alpha=0.5,
                label="FAIL",
            )
            ax.grid(alpha=0.5)
            ax.set_xlabel(param[7:])
            if i == 0:
                ax.set_ylabel("reward")

            if "learning_rate" in param:
                ax.set_xscale("log")
            if "noise" in param:
                plt.xticks(range(len(le_noise.classes_)), le_noise.classes_)
            if "activation" in param:
                plt.xticks(range(len(le_activation.classes_)), le_activation.classes_)

        ax.legend()

        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
        if show is False:
            plt.close()

    @staticmethod
    def custom_training_path_plot_algorithms(
        root_dir: Path,
        savepath: Path,
        algorithms: dict[str, list[tuple]],
        policy: str,
        figure_width=FIGURE_WIDTH,
        choose_num_best_rewards: int | None = None,
    ):
        """
        Example root_dir:

        root_dir = /mnt/P/2022/00/20220043/Calculations/

        Example algorithms:

        algorithms = dict(
            on=[  # on-policy
                ("PPO", "PPO_2022_09_27_study", "red"),
                ("A2C", "A2C_2022_11_30_study", "orange"),
                ("SAC", "SAC_2022_10_05_study", "black"),
            ],
            off=[  # off-policy
                ("DDPG", "DDPG_2022_10_03_study", "green"),
                ("TD3", "TD3_2022_09_27_study", "blue"),
            ])
        """
        fig, ax = plt.subplots(figsize=(1 * figure_width, 1 * figure_width))

        # read in dataframe for each algorihtm
        for alg, alg_path, color in algorithms[policy]:
            max_reward_list_path = Path(root_dir, f"{alg}_maxreward_experiment.csv")
            df_max_rewards = pd.read_csv(max_reward_list_path).sort_values(
                "max_reward", ascending=False
            )
            df_max_rewards = df_max_rewards[0:choose_num_best_rewards]
            trials = [
                Path(root_dir, alg_path, experiment_dir)
                for experiment_dir in df_max_rewards["experiment_directory"]
            ]

            # plot the chosen n best runs with differenct colors and labels
            for trial in track(
                trials, description=f"Plotting {len(trials)} trials for {alg}"
            ):
                df_log = pd.read_csv(trial / "progress.csv")
                n_strokes = df_log[r"rollout/ep_len_mean"].median()
                df_log["episodes"] = df_log[r"time/total_timesteps"] / n_strokes

                ax.plot(
                    df_log["episodes"],
                    df_log[r"rollout/ep_rew_mean"],
                    # alpha=0.3,
                    color=color,
                )
            # Add a single label for the algorithm outside the inner loop
            ax.plot([], [], color=color, label=alg)

        # ax.set_title(agent)

        ax.legend()
        ax.grid(alpha=0.5)

        if policy == "off":
            ax.set_ylim(top=1000, bottom=-1000)
            ax.set_xlim(xmin=0, xmax=2000)
        elif policy == "on":
            ax.set_ylim(top=670, bottom=0)
            ax.set_xlim(xmin=0, xmax=8500)
        else:
            ax.set_ylim(top=1000, bottom=-1000)
            ax.set_xlim(xmin=0, xmax=8500)

        ax.set_xlabel("episodes")
        ax.set_ylabel("reward")

        plt.tight_layout()
        plt.savefig(savepath)

    @staticmethod
    def custom_training_path_plot_algorithm(
        agent: str,
        root_directory: Path,
        study_name: str,
        mode: str = "rollout",
        print_thresh: int = None,
        savepath: str = None,
        figure_width=FIGURE_WIDTH,
        choose_num_best_rewards: int | None = None,
        filename_reward_list: Path | None = None,
    ) -> None:
        """Plots the training path for completed and running experiments for a single
        algorithm.

        Args:
            agent (str): The name of the agent.
            root_directory (Path): The root directory containing the experiment data
            directories.
            mode (str, optional): The mode to plot ('rollout' or 'eval'). Defaults to
            'rollout'.
            print_thresh (int, optional): Threshold for printing evaluation metrics.
            Defaults to None.
            savepath (str, optional): The file path to save the plot. Defaults to None.
            figure_width (int, optional): Width of the plot figure. Defaults to
            FIGURE_WIDTH.
            choose_num_best_rewards (int | None, optional): Number of best rewards to
            consider. Defaults to None.
            path_reward_list (Path | None, optional): The path to the reward list file.
            Defaults to None.

        Note:
            - When choose_num_best_rewards is not None, the function selects the best
            rewards from the reward list file.
            - When choose_num_best_rewards is None, the function selects trials of the
            specified agent type from the root directory.

        Raises:
            EmptyDataError: If progress.csv is empty or not found for a trial.

        Returns:
            None
        """
        study_directory = Path(root_directory, study_name)
        path_reward_list = Path(root_directory, filename_reward_list)
        if choose_num_best_rewards is not None:
            df_max_rewards = pd.read_csv(path_reward_list).sort_values(
                "max_reward", ascending=False
            )
            df_max_rewards = df_max_rewards[0:choose_num_best_rewards]
            trials = [
                Path(study_directory, experiment_dir)
                for experiment_dir in df_max_rewards["experiment_directory"]
            ]
        else:
            # only get trials of one agent type
            trials = [
                experiment_dir
                for experiment_dir in study_directory.iterdir()
                if agent in experiment_dir.name
            ]

        # initialize y-limits
        y_low = 1000
        y_high = -1000

        fig, ax = plt.subplots(figsize=(2 * figure_width, 1 * figure_width))

        for trial in track(
            trials, description=f"Plotting {len(trials)} trials for {agent}"
        ):
            try:
                df_log = pd.read_csv(trial / "progress.csv")

                max_reward = df_log[r"rollout/ep_rew_mean"].max()
                min_reward = df_log[r"rollout/ep_rew_mean"].min()
                # automatically adjusts y-scale in plot
                y_high = max_reward if max_reward > y_high else y_high
                y_low = min_reward if min_reward < y_low else y_low

                n_strokes = df_log[r"rollout/ep_len_mean"].median()
                df_log["episodes"] = df_log[r"time/total_timesteps"] / n_strokes

                if mode == "rollout":
                    ax.plot(
                        df_log["episodes"],
                        df_log[r"rollout/ep_rew_mean"],
                        alpha=0.3,
                        color="black",
                    )
                elif mode == "eval":
                    try:
                        df_log.dropna(axis=0, subset=["eval/mean_reward"], inplace=True)
                        if print_thresh is not None:
                            if df_log[r"eval/mean_reward"].max() > print_thresh:
                                print(trial, df_log[r"eval/mean_reward"].max())
                        ax.plot(
                            df_log["episodes"],
                            df_log[r"eval/mean_reward"],
                            alpha=0.3,
                            color="black",
                        )
                    except KeyError:
                        pass
            except EmptyDataError:
                pass

        ax.set_title(agent)

        ax.grid(alpha=0.5)
        ax.set_xlabel("episodes")
        ax.set_ylabel("reward")
        ax.set_ylim(top=y_high, bottom=y_low)

        plt.tight_layout()
        plt.savefig(savepath)

    @staticmethod
    def training_progress_plot(
        df_log: pd.DataFrame,
        df_env_log: pd.DataFrame,
        savepath: str = None,
        show: bool = True,
    ) -> None:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))
        ax1.plot(
            df_log["episodes"],
            df_log[r"rollout/ep_rew_mean"],
            label=r"rollout/ep_rew_mean",
        )
        try:
            ax1.scatter(
                df_log["episodes"],
                df_log["eval/mean_reward"],
                label=r"eval/mean_reward",
            )
        except KeyError:
            pass
        ax1.legend()
        ax1.grid(alpha=0.5)
        ax1.set_ylabel("reward")

        # plotting environment statistics
        for logged_var in [
            "avg_replaced_cutters",
            "avg_moved_cutters",
            "avg_broken_cutters",
            "var_cutter_locations",
            "avg_inwards_moved_cutters",
            "avg_wrong_moved_cutters",
        ]:
            ax2.plot(df_env_log["episodes"], df_env_log[logged_var], label=logged_var)
        ax2.legend()
        ax2.grid(alpha=0.5)
        ax2.set_ylabel("count")

        # model specific visualization of loss
        for column in df_log.columns:
            if "train" in column and "loss" in column:
                ax3.plot(df_log["episodes"], df_log[column], label=column)

        ax3.legend()
        ax3.grid(alpha=0.5)
        ax3.set_xlabel("episodes")
        ax3.set_ylabel("loss")

        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
        if show is False:
            plt.close()

    @staticmethod
    def action_analysis_scatter_plotly(df: pd.DataFrame, savepath: str = None) -> None:
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="broken cutters",
            hover_data={
                "x": False,
                "y": False,
                "state": True,
                "replaced cutters": True,
                "moved cutters": True,
            },
        )
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        fig.write_html(savepath)

    @staticmethod
    def action_analysis_scatter(
        df: pd.DataFrame, savepath: str = None, show: bool = True
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 9))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        im = ax.scatter(df["x"], df["y"], c=df["broken cutters"], cmap="turbo")
        fig.colorbar(im, cax=cax, orientation="vertical", label="broken cutters")
        ax.set_title("TSNE mapping of actions")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath)
        if show is False:
            plt.close()
