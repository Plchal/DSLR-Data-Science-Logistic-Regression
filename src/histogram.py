"""."""

import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import load_csv
from src.variable import COLS_TO_DROP, DATASET, HOUSE_COLORS, HOUSES


def find_most_homogenous_feature(df: pd.DataFrame) -> str:
    """."""
    dict_means = {}
    dict_stds = {}
    for house in HOUSES:
        df_tmp = df[df["Hogwarts House"] == house].drop(columns=COLS_TO_DROP)
        dict_means[house] = df_tmp.mean(numeric_only=True)
        dict_stds[house] = df_tmp.std(numeric_only=True)
    df_means = pd.DataFrame(dict_means).T
    df_stds = pd.DataFrame(dict_stds).T
    var_means = df_means.var(ddof=0)
    var_stds = df_stds.var(ddof=0)

    score_homogeneite = var_means + var_stds
    return str(score_homogeneite.idxmin())


def plot_histogram(df: pd.DataFrame, features: list[str]) -> None:
    """."""
    ncases = math.ceil(math.sqrt(len(features)))
    _fig, axes = plt.subplots(nrows=ncases, ncols=ncases, figsize=(16, 16), squeeze=False)
    _fig.patch.set_facecolor("#F8F9FA")
    axes_liste = axes.flatten()

    for i, col in enumerate(features):
        ax = axes_liste[i]
        ax.set_facecolor("#FFFFFF")

        ax.grid(axis="y", linestyle="--", alpha=0.5, color="#CCCCCC", zorder=0)

        for house, color in HOUSE_COLORS.items():
            data = df[df["Hogwarts House"] == house][col].dropna()

            ax.hist(data, bins=20, alpha=0.5, color=color, label=house, edgecolor="white", linewidth=0.7, zorder=3)

        ax.set_title(col, fontsize=12, fontweight="bold", pad=10, color="#333333")
        ax.set_xlabel("Notes", fontsize=10, color="dimgray")
        ax.set_ylabel("Number of people", fontsize=10, color="dimgray")
        ax.tick_params(colors="gray", labelsize=9)

    for j in range(len(features), len(axes_liste)):
        axes_liste[j].axis("off")

    handles, labels = ax.get_legend_handles_labels()
    _fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.01))

    plt.subplots_adjust(bottom=0.1, top=0.95, hspace=0.5, wspace=0.3)
    plt.show()


def main() -> int:
    """Create histograms for each subject."""
    df = load_csv(DATASET)
    if df is None:
        return 1
    parser = argparse.ArgumentParser(description="A program that displays one or more histograms.")

    parser.add_argument("-s", "--subject", type=str, help="Show the school subject histogram you choose.")
    parser.add_argument(
        "-a", "--auto", action="store_true", help="Show the school subject histogram the most homogenous."
    )
    parser.add_argument("-f", "--full", action="store_true", help="Show all of school subject histogram.")
    args = parser.parse_args()
    if args.full:
        features = [col for col in df.columns if col not in COLS_TO_DROP]
        plot_histogram(df, features)
        return 0

    if args.auto:
        feature = [f"{find_most_homogenous_feature(df)}"]
        plot_histogram(df, feature)
        return 0

    if args.subject:
        if args.subject not in df.columns:
            print(f"Error : The subject '{args.subject}' does not exist in the dataset.")
            valid_subject = [col for col in df.columns if col not in COLS_TO_DROP]
            print(f"Subjects existing : {' , '.join(valid_subject)}")
            return 1
        feature = [args.subject]
        plot_histogram(df, feature)
        return 0

    feature = ["Care of Magical Creatures"]
    plot_histogram(df, feature)
    return 0


if __name__ == "__main__":
    main()
