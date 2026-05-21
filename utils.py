import numpy as np
import torch
import random
from torch.utils.data import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, to_tensor
import json
import os


def fix_random(seed: int) -> None:
    """Fix all the possible sources of randomness.

    Args:
        seed: the seed to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class FabricDataset(Dataset):
    SPLITS = {"train": "train", "val": "validation", "test": "test_public"}

    def __init__(
        self, split, transforms=None, root="data/fabric", target_res=(2048, 2304)
    ):
        self.split = split
        self.transforms = transforms
        self.target_res = target_res  # Store explicitly (Height, Width)

        self.images = []
        self.labels = []
        self.masks = []

        split_folder = self.SPLITS[split]
        split_path = Path(root) / split_folder

        self._add_images(split_path / "good", 0)
        if split == "test":
            self._add_images(split_path / "bad", 1)

        # Pre-allocate a single static base mask to copy from (Massive CPU RAM saver)
        # For training, it matches the crop size; for val/test, it matches full res.
        if self.split == "train":
            self.cached_zero_mask = torch.zeros((1, 256, 256))
        else:
            self.cached_zero_mask = torch.zeros(
                (1, self.target_res[0], self.target_res[1])
            )

    def _add_images(self, folder, label):
        image_paths = sorted(folder.glob("*.png"))
        if not image_paths:
            raise FileNotFoundError(f"No png images found in {folder}")

        for image_path in image_paths:
            self.images.append(image_path)
            self.labels.append(label)
            mask = (
                folder.parent / "ground_truth" / "bad" / f"{image_path.stem}_mask.png"
            )
            self.masks.append(mask if label else None)

    def __getitem__(self, idx):
        # 1. Load Image
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.labels[idx]
        mask_path = self.masks[idx]

        # 2. Handle Mask efficiently
        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
            mask = to_tensor(mask)
            # Instead of sniffing the image, explicitly resize to target dimensions
            mask = resize(
                mask, self.target_res, interpolation=InterpolationMode.NEAREST
            )
        else:
            # Reusing the pre-allocated zero mask avoids CPU memory allocation thrashing
            mask = self.cached_zero_mask

        return img, label, mask

    def __len__(self):
        return len(self.images)


def plot_loss(run_id):
    history_path = os.path.join("results", run_id, "history.json")
    with open(history_path, "r") as f:
        history = json.load(f)

    train_metrics = history["train"]
    val_metrics = history["val"]

    # chronological plots
    train_metrics.sort(key=lambda x: x["step"])
    val_metrics.sort(key=lambda x: x["step"])

    plt.plot(
        [x["step"] for x in train_metrics],
        [x["train_loss"] for x in train_metrics],
        label="train loss",
    )
    plt.plot(
        [x["step"] for x in val_metrics],
        [x["val_loss"] for x in val_metrics],
        label="val loss",
    )
    plt.xlabel("steps")
    plt.legend()
    plt.show()


def plot_anomaly_results(
    original_image, reconstruction, anomaly_map, gt_mask, figsize=(15, 5)
):
    """
    Plots the original image, its reconstruction, the anomaly map, and the ground truth mask side-by-side.

    Args:
        original_image (torch.Tensor): The original input image (C, H, W).
        reconstruction (torch.Tensor): The reconstructed image (C, H, W).
        anomaly_map (torch.Tensor): The computed anomaly map (H, W).
        gt_mask (torch.Tensor): The ground truth anomaly mask (H, W).
        figsize (tuple): Figure size for matplotlib.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Original Image
    axes[0].imshow(original_image.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Reconstruction
    axes[1].imshow(reconstruction.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    # Anomaly Map
    # Normalize anomaly_map for better visualization if needed, or use a specific colormap
    im = axes[2].imshow(anomaly_map.cpu().numpy(), cmap="hot")
    axes[2].set_title("Anomaly Map (MSE)")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # Ground Truth Mask
    axes[3].imshow(gt_mask.squeeze().cpu().numpy(), cmap="gray")
    axes[3].set_title("Ground Truth Mask")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()
