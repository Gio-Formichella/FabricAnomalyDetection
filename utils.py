import numpy as np
import torch
import random
from torch.utils.data import Dataset
from pathlib import Path

from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, to_tensor


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

    def __init__(self, split, transforms=to_tensor, root="data/fabric"):
        split_folder = self.SPLITS[split]
        self.transforms = transforms
        self.images = []
        self.labels = []
        self.masks = []

        split_path = Path(root) / split_folder
        self._add_images(split_path / "good", 0)
        if split == "test":
            self._add_images(split_path / "bad", 1)

    def _add_images(self, folder, label):
        image_paths = sorted(folder.glob("*.png"))
        if not image_paths:
            raise FileNotFoundError(f"No png images found in {folder}")

        for image_path in image_paths:
            self.images.append(image_path)
            self.labels.append(label)
            mask = folder.parent / "ground_truth" / "bad" / f"{image_path.stem}_mask.png"
            self.masks.append(mask if label else None)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.labels[idx]

        mask = self.masks[idx]
        if mask is not None:
            mask = Image.open(mask).convert("L")
            mask = to_tensor(mask)
            if isinstance(img, torch.Tensor):
                mask = resize(mask, img.shape[1:], InterpolationMode.NEAREST)
        else:
            if isinstance(img, torch.Tensor):
                height, width = img.shape[1], img.shape[2]
            else:
                width, height = img.size
            mask = torch.zeros((1, height, width))

        return img, label, mask

    def __len__(self):
        return len(self.images)
