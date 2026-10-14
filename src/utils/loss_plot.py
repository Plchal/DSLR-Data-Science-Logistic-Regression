"""TODO."""

import matplotlib.pyplot as plt

from .variable import HOUSE_COLORS


def plot_loss(histories: dict[str, list[float]], mode: str) -> None:
    """Display the evolution of the loss across the iterations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#FFFFFF")

    for name, history in histories.items():
        if not history:
            continue
        color = HOUSE_COLORS.get(name, "#333333")
        ax.plot(range(len(history)), history, color=color, linewidth=1.5, label=name)

    ax.set_title(f"Loss evolution ({mode})", fontsize=13, fontweight="bold", pad=10, color="#333333")
    ax.set_xlabel("Iteration", fontsize=10, color="dimgray")
    ax.set_ylabel("Loss", fontsize=10, color="dimgray")
    ax.grid(linestyle="--", alpha=0.5, color="#CCCCCC", zorder=0)
    ax.tick_params(colors="gray", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=10, frameon=False)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.14)
    plt.show()
