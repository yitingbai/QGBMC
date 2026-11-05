from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[6] / "sn-article-template"


def noise_curve():
    rates = np.array([0.0, 0.3, 0.6, 0.9])
    series = {
        "HandWritten, element": (
            np.array([0.8009, 0.8132, 0.8197, 0.5412]),
            np.array([0.0022, 0.0124, 0.0427, 0.0404]),
            "#1f77b4",
        ),
        "Hdigit, element": (
            np.array([0.8841, 0.9899, 0.9721, 0.5720]),
            np.array([0.0888, 0.0017, 0.0053, 0.0289]),
            "#d62728",
        ),
    }
    sample = {
        "HandWritten, sample": (0.5842, 0.0268, "#1f77b4"),
        "Hdigit, sample": (0.5871, 0.0489, "#d62728"),
    }

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for label, (mean, sd, color) in series.items():
        ax.errorbar(
            rates,
            mean,
            yerr=sd,
            marker="o",
            linewidth=2,
            capsize=3,
            color=color,
            label=label,
        )
    for i, (label, (mean, sd, color)) in enumerate(sample.items()):
        ax.errorbar(
            0.915 + i * 0.025,
            mean,
            yerr=sd,
            marker="s",
            linestyle="none",
            capsize=3,
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.8,
            label=label,
        )
    ax.set_xlabel("Corruption rate")
    ax.set_ylabel("Clustering accuracy")
    ax.set_xlim(-0.03, 0.98)
    ax.set_ylim(0.45, 1.04)
    ax.set_xticks(rates)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "revision_noise_curve.pdf", bbox_inches="tight")
    fig.savefig(OUT / "revision_noise_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def mechanism():
    datasets = ["HandWritten", "Hdigit"]
    occupancy = np.array([1.0, 1.0])
    gradient_ratio = np.array([0.135335, 0.135335])
    x = np.arange(len(datasets))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    b1 = ax.bar(
        x - width / 2,
        occupancy,
        width,
        color="#4c78a8",
        label="Decay-branch occupancy",
    )
    b2 = ax.bar(
        x + width / 2,
        gradient_ratio,
        width,
        color="#f58518",
        label="Modulated/raw gradient ratio",
    )
    ax.bar_label(b1, labels=["100%", "100%"], padding=3, fontsize=9)
    ax.bar_label(b2, labels=["0.135", "0.135"], padding=3, fontsize=9)
    ax.set_xticks(x, datasets)
    ax.set_ylabel("Proportion or ratio")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "revision_mechanism.pdf", bbox_inches="tight")
    fig.savefig(OUT / "revision_mechanism.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    noise_curve()
    mechanism()
