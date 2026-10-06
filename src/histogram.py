"""."""

import math

import matplotlib.pyplot as plt

from utils import load_csv


def main() -> None:
    """Create histograms for each subject."""
    df = load_csv("dataset_train.csv")

    house_colors = {"Ravenclaw": "#222F5B", "Slytherin": "#1A472A", "Gryffindor": "#740001", "Hufflepuff": "#D89E15"}

    cols_to_drop = ["Index", "First Name", "Last Name", "Birthday", "Best Hand", "Hogwarts House"]
    features = [col for col in df.columns if col not in cols_to_drop]

    ncases = math.ceil(math.sqrt(len(features)))
    _fig, axes = plt.subplots(nrows=ncases, ncols=ncases, figsize=(16, 16))
    _fig.patch.set_facecolor("#F8F9FA")
    axes_liste = axes.flatten()

    for i, col in enumerate(features):
        ax = axes_liste[i]
        ax.set_facecolor("#FFFFFF")

        ax.grid(axis="y", linestyle="--", alpha=0.5, color="#CCCCCC", zorder=0)

        for house, color in house_colors.items():
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


if __name__ == "__main__":
    main()
