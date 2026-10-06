"""."""

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from utils import load_csv


def main() -> None:
    """."""
    df = load_csv("dataset_train.csv")

    cols_to_drop = ["Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"]
    features = [col for col in df.columns if col not in cols_to_drop]

    n_features = len(features)

    house_colors = {"Ravenclaw": "#222F5B", "Slytherin": "#1A472A", "Gryffindor": "#740001", "Hufflepuff": "#D89E15"}

    fig, axes = plt.subplots(nrows=n_features, ncols=n_features, figsize=(20, 20))
    fig.patch.set_facecolor("#F8F9FA")

    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]
            ax.set_facecolor("#FFFFFF")

            feat_y = features[i]
            feat_x = features[j]

            for house, color in house_colors.items():
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
        for house, color in house_colors.items()
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=4, fontsize=16, frameon=False, bbox_to_anchor=(0.5, 0.98)
    )

    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08, wspace=0.05, hspace=0.05)
    plt.show()


if __name__ == "__main__":
    main()
