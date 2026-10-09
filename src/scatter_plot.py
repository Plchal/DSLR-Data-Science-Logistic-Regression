"""."""

import argparse
from itertools import combinations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

from mathematics import calculate_pearson_correlation_coefficient
from utils import load_csv
from variable import COLS_TO_DROP, HOUSE_COLORS


def find_most_similarity_features(df: pd.DataFrame) -> list[str]:
    """."""
    cols_to_keep = df.drop(columns=COLS_TO_DROP).columns.tolist()
    number_of_features = 2
    if len(cols_to_keep) < number_of_features:
        return []

    max_corr = -1.0
    best_pair = []

    for col1, col2 in combinations(cols_to_keep, 2):
        corr = calculate_pearson_correlation_coefficient(df[col1], df[col2])
        if pd.isna(corr):
            continue

        if abs(corr) > max_corr:
            max_corr = abs(corr)
            best_pair = [col1, col2]
    return best_pair


def show_scatter_plot(df: pd.DataFrame, features: list[str]) -> None:
    """."""
    n_features = len(features)

    fig, axes = plt.subplots(nrows=n_features, ncols=n_features, figsize=(20, 20))
    fig.patch.set_facecolor("#F8F9FA")

    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]

            if j >= i:
                ax.set_visible(False)
                continue

            ax.set_facecolor("#FFFFFF")

            feat_y = features[i]
            feat_x = features[j]

            for house, color in HOUSE_COLORS.items():
                data = df[df["Hogwarts House"] == house][[feat_x, feat_y]].dropna()

                ax.scatter(data[feat_x], data[feat_y], alpha=0.6, color=color, s=2, linewidth=0)

            ax.set_xticks([])
            ax.set_yticks([])

            if j == 0:
                ax.set_ylabel(
                    feat_y.replace(" ", "\n"), fontsize=7, color="#333333", rotation=0, labelpad=40, ha="right"
                )
            if i == n_features - 1:
                ax.set_xlabel(feat_x.replace(" ", "\n"), fontsize=7, color="#333333", rotation=0)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#DDDDDD")
            ax.spines["bottom"].set_color("#DDDDDD")

    legend_handles = [
        mlines.Line2D([], [], color=color, marker="o", linestyle="None", markersize=10, label=house)
        for house, color in HOUSE_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=4, fontsize=16, frameon=False, bbox_to_anchor=(0.5, 0.98)
    )

    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08, wspace=0.05, hspace=0.05)
    plt.show()


def main() -> int:
    """."""
    df = load_csv("dataset_train.csv")
    if df is None:
        return 1

    parser = argparse.ArgumentParser(description="A program that displays one or more scatter plot.")
    parser.add_argument("-s1", "--subject1", type=str, help="Select the first subject you want to compare.")
    parser.add_argument("-s2", "--subject2", type=str, help="Select the second subject you want to compare.")

    parser.add_argument("-a", "--auto", action="store_true", help="Show the two subject with the most similarity.")
    parser.add_argument("-f", "--full", action="store_true", help="Show all combination of subject compare.")
    args = parser.parse_args()

    if args.full:
        features = [col for col in df.columns if col not in COLS_TO_DROP]
        show_scatter_plot(df, features)
        return 0

    if args.subject1 and args.subject2:
        if args.subject1 not in df.columns or args.subject2 not in df.columns:
            print(
                f"Error :  subject 1 '{args.subject1}' and or subject 2 '{args.subject2}'  does not exist in the dataset."
            )
            valid_subject = [col for col in df.columns if col not in COLS_TO_DROP]
            print(f"Subjects existing : {' , '.join(valid_subject)}")
            return 1

        features = [args.subject1, args.subject2]
        show_scatter_plot(df, features)
        return 0

    if args.auto:
        features = find_most_similarity_features(df)
        show_scatter_plot(df, features)
        return 0

    features = ["Astronomy", "Defense Against the Dark Arts"]
    show_scatter_plot(df, features)
    return 0


if __name__ == "__main__":
    main()
