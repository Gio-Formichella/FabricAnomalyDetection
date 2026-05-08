import numpy as np
import torch
import random
from torch.utils.data import Dataset
import os
import cv2


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
    def __init__(self, dataset_path, split, transforms=None):
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms

        self.images = []  # Paths to images
        self.labels = []  # 0 for good and 1 bad (i.e. anomaly)
        self.masks = []  # Paths to ground truth masks

        if self.split in ("train", "val"):
            folder_name = "train" if self.split == "train" else "validation"
            split_folder = os.path.join(self.dataset_path, folder_name, "good")
            for img_id in os.listdir(split_folder):
                self.images.append(os.path.join(split_folder, img_id))
                # Training and validation images have no anomalies
                self.labels.append(0)
                self.masks.append(None)

        elif self.split == "test":
            split_folder = os.path.join(self.dataset_path, "test_public")

            good_folder = os.path.join(
                split_folder, "good"
            )  # Test data with no anomalies
            for img_id in os.listdir(good_folder):
                self.images.append(os.path.join(good_folder, img_id))
                self.labels.append(0)
                self.masks.append(None)

            bad_folder = os.path.join(split_folder, "bad")  # Test data with anomalies
            for img_id in os.listdir(bad_folder):
                self.images.append(os.path.join(bad_folder, img_id))
                self.labels.append(1)

                mask_id = img_id[:-4] + "_mask.png"
                self.masks.append(
                    os.path.join(split_folder, "ground_truth", "bad", mask_id)
                )
        else:
            raise ValueError("Split must be either train, val or test")

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx], cv2.IMREAD_COLOR_RGB)
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.labels[idx]

        mask = self.masks[idx]
        if mask is not None:
            mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)

        return img, label, mask

    def __len__(self):
        return len(self.images)
