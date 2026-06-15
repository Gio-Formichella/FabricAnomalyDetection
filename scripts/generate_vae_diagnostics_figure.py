from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vae import (
    FeatureVAE,
    cache_images,
    crop_fraction,
    hidden_dim,
    image_size,
    latent_dim,
    patch_errors,
    patch_features,
    patch_size,
)


DATA_ROOT = ROOT / "data" / "fabric"
WEIGHTS_PATH = ROOT / "results" / "vae.pt"
OUTPUT_PATHS = [
    ROOT / "results" / "vae_reconstruction_comparison.png",
    ROOT / "slides" / "imgs" / "vae_reconstruction_comparison.png",
]

EXAMPLES = [
    (
        "Anomaly Type A (Large)",
        DATA_ROOT / "test_public" / "bad" / "000_overexposed.png",
        DATA_ROOT / "test_public" / "ground_truth" / "bad" / "000_overexposed_mask.png",
    ),
    (
        "Anomaly Type B (Small)",
        DATA_ROOT / "test_public" / "bad" / "012_shift_1.png",
        DATA_ROOT / "test_public" / "ground_truth" / "bad" / "012_shift_1_mask.png",
    ),
    (
        "Sane Image",
        DATA_ROOT / "test_public" / "good" / "000_overexposed.png",
        None,
    ),
]


def load_preprocessed_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    image = image.crop(
        (
            int(width * (1 - crop_fraction) / 2),
            int(height * (1 - crop_fraction) / 2),
            int(width * (1 + crop_fraction) / 2),
            int(height * (1 + crop_fraction) / 2),
        )
    )
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0


def load_preprocessed_mask(path: Path | None) -> np.ndarray:
    if path is None:
        return np.zeros((image_size, image_size), dtype=np.float32)

    mask = Image.open(path).convert("L")
    width, height = mask.size
    mask = mask.crop(
        (
            int(width * (1 - crop_fraction) / 2),
            int(height * (1 - crop_fraction) / 2),
            int(width * (1 + crop_fraction) / 2),
            int(height * (1 + crop_fraction) / 2),
        )
    )
    mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.float32) / 255.0


def stitch_gray_patches(features: torch.Tensor, grid_size: int) -> np.ndarray:
    gray = features[:, : patch_size * patch_size].reshape(
        grid_size, grid_size, patch_size, patch_size
    )
    image = gray.permute(0, 2, 1, 3).reshape(image_size, image_size).numpy()
    p1, p99 = np.percentile(image, [1, 99])
    return np.clip((image - p1) / max(p99 - p1, 1e-6), 0, 1)


def draw_top_patch_boxes(ax, heatmap: np.ndarray, fraction: float = 0.02) -> None:
    grid_size = heatmap.shape[0]
    k = max(1, int(heatmap.size * fraction))
    top_indices = np.argpartition(heatmap.ravel(), -k)[-k:]

    for idx in top_indices:
        row, col = divmod(int(idx), grid_size)
        ax.add_patch(
            patches.Rectangle(
                (col - 0.5, row - 0.5),
                1,
                1,
                linewidth=1.8,
                edgecolor="#45ff63",
                facecolor="none",
            )
        )


def main() -> None:
    device = "cpu"
    train_images, _, _ = cache_images(
        "train", str(DATA_ROOT), device=device, chunk_size=4
    )
    example_images = torch.stack(
        [load_preprocessed_image(path) for _, path, _ in EXAMPLES]
    )
    example_masks = [load_preprocessed_mask(mask_path) for _, _, mask_path in EXAMPLES]

    train_features_raw, n_train, train_ppi = patch_features(
        train_images, patch_size, device
    )
    example_features_raw, n_examples, example_ppi = patch_features(
        example_images, patch_size, device
    )
    feature_mean = train_features_raw.mean(dim=0, keepdim=True)
    feature_std = train_features_raw.std(dim=0, keepdim=True).clamp_min(1e-4)
    train_features = (train_features_raw - feature_mean) / feature_std
    example_features = (example_features_raw - feature_mean) / feature_std

    model = FeatureVAE(train_features.shape[1], hidden_dim, latent_dim).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    train_loss = patch_errors(model, train_features, n_train, train_ppi, device)
    normalizer = train_loss.mean(dim=0, keepdim=True).clamp_min(1e-6)
    normalized_loss = patch_errors(
        model, example_features, n_examples, example_ppi, device
    ) / normalizer

    with torch.no_grad():
        reconstructed_features = (
            model.reconstruct(example_features).cpu() * feature_std + feature_mean
        )

    grid_size = image_size // patch_size
    heatmaps = normalized_loss.reshape(n_examples, grid_size, grid_size).numpy()
    reconstructions = [
        stitch_gray_patches(
            reconstructed_features[i * example_ppi : (i + 1) * example_ppi],
            grid_size,
        )
        for i in range(n_examples)
    ]
    shared_vmax = float(np.ceil(heatmaps.max()))

    fig, axes = plt.subplots(3, 4, figsize=(18, 13), constrained_layout=True)
    heatmap_image = None

    for row, ((label, _, _), image, reconstruction, heatmap, mask) in enumerate(
        zip(EXAMPLES, example_images, reconstructions, heatmaps, example_masks)
    ):
        axes[row, 0].imshow(image.permute(1, 2, 0).numpy())
        axes[row, 0].set_title(f"{label}\nOriginal Image")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(reconstruction, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Feature Reconstruction")
        axes[row, 1].axis("off")

        heatmap_image = axes[row, 2].imshow(
            heatmap,
            cmap="hot",
            interpolation="nearest",
            vmin=0,
            vmax=shared_vmax,
        )
        draw_top_patch_boxes(axes[row, 2], heatmap)
        axes[row, 2].set_title("Anomaly Map (shared scale)")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].set_title("Ground Truth Mask")
        axes[row, 3].axis("off")

    assert heatmap_image is not None
    cbar = fig.colorbar(heatmap_image, ax=axes[:, 2], fraction=0.045, pad=0.02)
    cbar.set_label("Normalized MSE, shared across rows")

    for output_path in OUTPUT_PATHS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        print(f"Saved {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
