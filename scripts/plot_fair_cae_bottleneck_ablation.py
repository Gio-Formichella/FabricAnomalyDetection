import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_JSON = ROOT / "slides" / "cae_bottleneck_auroc.json"
OUT_FIG = ROOT / "slides" / "imgs" / "cae_bottleneck_auroc.png"


def main():
    with DATA_JSON.open("r") as f:
        payload = json.load(f)

    rows = payload["results"]
    best_idx = max(range(len(rows)), key=lambda idx: rows[idx]["auroc"])
    x = list(range(len(rows)))
    y = [row["auroc"] for row in rows]
    labels = [str(row["bottleneck_channels"]) for row in rows]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=180)
    ax.plot(
        x,
        y,
        color="#315c8a",
        linewidth=2.2,
        marker="o",
        markersize=6,
    )
    ax.scatter(
        [best_idx],
        [y[best_idx]],
        color="#d62728",
        edgecolor="white",
        linewidth=1.2,
        s=95,
        zorder=4,
        label="Best AUROC",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("Bottleneck capacity")
    ax.set_ylabel("Image-level AUROC")
    ax.set_ylim(max(0.45, min(y) - 0.04), min(1.0, max(y) + 0.06))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")


if __name__ == "__main__":
    main()
