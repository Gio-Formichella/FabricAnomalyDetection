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
    spatial_rows = payload.get("spatial_checks", [])
    all_rows = sorted(
        [
            *({"kind": "main", **row} for row in rows),
            *({"kind": "spatial", **row} for row in spatial_rows),
        ],
        key=lambda row: (row["latent_activations"], row["bottleneck_shape"]),
    )

    main_x = [idx for idx, row in enumerate(all_rows) if row["kind"] == "main"]
    main_y = [all_rows[idx]["auroc"] for idx in main_x]
    spatial_x = [idx for idx, row in enumerate(all_rows) if row["kind"] == "spatial"]
    spatial_y = [all_rows[idx]["auroc"] for idx in spatial_x]
    labels = [row["bottleneck_shape"] for row in all_rows]
    best_idx = max(main_x, key=lambda idx: all_rows[idx]["auroc"])
    best_y = all_rows[best_idx]["auroc"]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=180)
    ax.plot(
        main_x,
        main_y,
        color="#315c8a",
        linewidth=2.2,
        marker="o",
        markersize=6,
        label="8x8 sweep",
    )
    if spatial_rows:
        ax.scatter(
            spatial_x,
            spatial_y,
            color="#8a8f98",
            marker="D",
            s=48,
            label="Spatial check",
            zorder=3,
        )
    ax.scatter(
        [best_idx],
        [best_y],
        color="#d62728",
        edgecolor="white",
        linewidth=1.2,
        s=95,
        zorder=4,
        label="Best AUROC",
    )

    ax.set_xticks(list(range(len(all_rows))))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("Bottleneck capacity")
    ax.set_ylabel("Image-level AUROC")
    all_y = main_y + spatial_y
    ax.set_ylim(max(0.45, min(all_y) - 0.04), min(1.0, max(all_y) + 0.06))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")


if __name__ == "__main__":
    main()
