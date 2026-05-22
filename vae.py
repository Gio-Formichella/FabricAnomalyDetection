import copy
import json
import os
import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from utils import FabricDataset, fix_random, plot_loss
from PIL import Image


data_root = "data/fabric"
image_size = 256
crop_fraction = 0.8
patch_size = 8
hidden_dim = 512
latent_dim = 32
beta = 1e-3
noise_std = 0.05 # Amount of noise to add to the input for regularization
learning_rate = 1e-3
epochs = 50
batch_size = 4096 # Number of patches per batch
seed = 123
topk = 0.002 # Fraction of patches to consider as anomalies
weights_path = "results/vae.pt"


class CenterCropFraction:
    """
    Helps preventing images' border noise to deteriorate performance
    """
    
    def __init__(self, fraction):
        self.fraction = fraction

    def __call__(self, image):
        width, height = image.size
        return TF.center_crop(
            image, [int(height * self.fraction), int(width * self.fraction)]
        )


class FeatureVAE(nn.Module):
    """
    A simple Variational Autoencoder using dense layers, designed to process 1D patch features.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        mid_dim = hidden_dim // 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(mid_dim, latent_dim)
        self.logvar = nn.Linear(mid_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-8, 8)
        
        # Reparameterization Trick: Allows backpropagation through random sampling.
        # Instead of sampling z directly from N(mu, sigma^2) which breaks gradients,
        # we sample epsilon from a standard normal N(0, 1) and shift/scale it.
        # sigma = exp(0.5 * logvar)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        
        return self.decoder(z), mu, logvar

    def reconstruct(self, x):
        return self.decoder(self.mu(self.encoder(x)))



def cache_images(split, root, device="cuda", chunk_size=32):
    """
    Returns a tensor of images, a tensor of labels, and a tensor of masks.
    Uses GPU-accelerated batch transforms for faster preprocessing.
    """
    dataset = FabricDataset(split, root=root)
    n = len(dataset)
    all_images, all_masks, labels = [], [], []
    chunk = []

    for idx in tqdm(range(n), desc=f"Cache {split}"):
        img = Image.open(dataset.images[idx]).convert("RGB")
        chunk.append(transforms.functional.to_tensor(img))
        labels.append(dataset.labels[idx])

        mask_path = dataset.masks[idx]
        if mask_path is not None:
            mask_pil = Image.open(mask_path).convert("L")
            w, h = mask_pil.size
            mask_pil = TF.center_crop(
                mask_pil, [int(h * crop_fraction), int(w * crop_fraction)]
            )
            mask_pil = mask_pil.resize((image_size, image_size), Image.NEAREST)
            all_masks.append(transforms.functional.to_tensor(mask_pil))
        else:
            all_masks.append(torch.zeros(1, image_size, image_size))

        # Flush chunk to GPU when full (avoids holding all full-res images in RAM)
        if len(chunk) == chunk_size:
            batch = torch.stack(chunk).to(device)
            _, _, h, w = batch.shape
            batch = TF.center_crop(
                batch, [int(h * crop_fraction), int(w * crop_fraction)]
            )
            batch = TF.resize(batch, [image_size, image_size])
            all_images.append(batch.cpu())
            chunk = []

    if chunk:
        batch = torch.stack(chunk).to(device)
        _, _, h, w = batch.shape
        batch = TF.center_crop(
            batch, [int(h * crop_fraction), int(w * crop_fraction)]
        )
        batch = TF.resize(batch, [image_size, image_size])
        all_images.append(batch.cpu())

    return torch.cat(all_images), torch.tensor(labels), torch.stack(all_masks)


def patch_features(images, patch_size, device="cuda"):
    images = images.to(device)
    # Split images into a grid of non-overlapping patches
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # from [Batch, Channels, GridHeight, GridWidth, PatchHeight, PatchWidth] to [Batch, GridHeight, GridWidth, Channels, PatchHeight, PatchWidth]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous() 
    n_images, grid_h, grid_w, _, _, _ = patches.shape
    # from [Batch, GridHeight, GridWidth, Channels, PatchHeight, PatchWidth] to [Batch * GridHeight * GridWidth, Channels, PatchHeight, PatchWidth]
    patches = patches.view(-1, 3, patch_size, patch_size)

    # Convert to grayscale and normalize each patch
    gray = 0.299 * patches[:, 0] + 0.587 * patches[:, 1] + 0.114 * patches[:, 2]
    gray = (gray - gray.mean(dim=(1, 2), keepdim=True)) / gray.std(
        dim=(1, 2), keepdim=True
    ).clamp_min(1e-4)
    
    # Calculate spatial gradients (horizontal and vertical edges)
    dx = gray[:, :, 1:] - gray[:, :, :-1]
    dy = gray[:, 1:, :] - gray[:, :-1, :]
    
    # Calculate frequency domain features (texture patterns)
    fft = torch.fft.rfft2(gray, norm="ortho").abs()
    fft = torch.log1p(fft[:, : patch_size // 2, : patch_size // 2])

    features = torch.cat(
        [gray.flatten(1), dx.flatten(1), dy.flatten(1), fft.flatten(1)],
        dim=1,
    )
    return features.cpu(), n_images, grid_h * grid_w


def standardize(train, *others):
    """
    Subtracts the mean and divides by the standard deviation for each feature.
    """
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1e-4)
    return (train - mean) / std, tuple((x - mean) / std for x in others)


def vae_loss(x_hat, x, mu, logvar, beta):
    """
    Calculates the VAE loss, which is the sum of the reconstruction loss and the KL divergence.
    """
    recon = nn.functional.mse_loss(x_hat, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld, recon, kld


def reconstruction_loss(model, loader, device):
    """
    Calculates the reconstruction loss for anomaly detection.
    """
    model.eval()
    total = 0.0
    feature_dim = loader.dataset.tensors[0].shape[1]
    with torch.no_grad():
        for (features,) in loader:
            x = features.to(device, non_blocking=True)
            total += nn.functional.mse_loss(
                model.reconstruct(x), x, reduction="sum"
            ).item()
    return total / len(loader.dataset) / feature_dim


def train(model, train_loader, val_loader, optimizer, device, epochs, beta, noise_std, run_id):
    """
    Trains the Denoising VAE. Saves the best model based on validation reconstruction error.
    """
    results_dir = os.path.join("results", run_id)
    os.makedirs(results_dir, exist_ok=True)
    history_path = os.path.join(results_dir, "history.json")
    weights_path = os.path.join(results_dir, "vae.pt")

    train_history = []
    val_history = []
    step = 0

    best_val = float("inf")
    best_state = None
    for epoch in tqdm(range(epochs), desc="Training VAE"):
        model.train()
        total_loss = total_recon = total_kld = 0.0
        for (features,) in train_loader:
            step += 1
            x = features.to(device, non_blocking=True)
            noisy_x = x + noise_std * torch.randn_like(x)
            x_hat, mu, logvar = model(noisy_x)
            loss, recon, kld = vae_loss(x_hat, x, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = x.size(0)
            total_loss += loss.item() * n
            total_recon += recon.item() * n
            total_kld += kld.item() * n

            train_history.append({"step": step, "train_loss": loss.item()})

        val_recon = reconstruction_loss(model, val_loader, device)
        if val_recon < best_val:
            best_val = val_recon
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, weights_path)
        val_history.append({"step": step, "val_loss": val_recon})

        history = {"train": train_history, "val": val_history}
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)

        count = len(train_loader.dataset)
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"loss={total_loss / count:.6f} "
            f"recon={total_recon / count:.6f} "
            f"kld={total_kld / count:.6f} "
            f"val_recon={val_recon:.6f}",
            flush=True,
        )
    model.load_state_dict(best_state)
    print(f"Loaded best validation checkpoint: val_recon={best_val:.6f}", flush=True)


def patch_errors(model, features, n_images, patches_per_image, device):
    """
    Calculates the Mean Squared Error (MSE) between the original and reconstructed patches.
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in DataLoader(TensorDataset(features), batch_size=8192):
            x = batch[0].to(device, non_blocking=True)
            recon = model.reconstruct(x)
            scores.append((x - recon).pow(2).mean(dim=1).cpu())
    return torch.cat(scores).view(n_images, patches_per_image)


def score_images(model, features, n_images, patches_per_image, device, topk, normalizer):
    """
    Aggregates patch-level errors into image-level anomaly scores using the top-k worst patches.
    """
    patch_scores = patch_errors(model, features, n_images, patches_per_image, device)
    patch_scores = patch_scores / normalizer
    k = max(1, int(patches_per_image * topk))
    return patch_scores.topk(k, dim=1).values.mean(dim=1)



def main():
    """
    Main pipeline: Data loading -> Feature extraction -> Model training -> Anomaly scoring.
    """
    fix_random(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if image_size % patch_size:
        raise ValueError("image_size must be divisible by patch_size")

    device = "cuda"
    train_images, _, _ = cache_images("train", data_root, device)
    val_images, _, _ = cache_images("val", data_root, device)
    test_images, test_labels, test_masks = cache_images("test", data_root, device)

    train_features, n_train, train_ppi = patch_features(train_images, patch_size, device)
    val_features, _, _ = patch_features(val_images, patch_size, device)
    test_features, n_test, test_ppi = patch_features(test_images, patch_size, device)
    train_features, (val_features, test_features) = standardize(
        train_features, val_features, test_features
    )

    train_loader = DataLoader(
        TensorDataset(train_features),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_features),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    model = FeatureVAE(train_features.shape[1], hidden_dim, latent_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    print(
        f"CUDA: {torch.cuda.get_device_name(0)} | "
        f"train={len(train_images)} val={len(val_images)} test={len(test_images)} | "
        f"features={train_features.shape[1]} train_patches={len(train_features)}",
        flush=True,
    )
    run_id = f"vae_latent{latent_dim}_beta{beta}"
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs,
        beta,
        noise_std,
        run_id,
    )
    plot_loss(run_id)
    train_errors = patch_errors(model, train_features, n_train, train_ppi, device)
    normalizer = train_errors.mean(dim=0, keepdim=True).clamp_min(1e-6)

    test_scores = score_images(
        model, test_features, n_test, test_ppi, device, topk, normalizer
    )
    test_labels_np = test_labels.numpy()
    test_scores_np = test_scores.numpy()
    print(
        f"Held-out test Image-Level AUROC: {roc_auc_score(test_labels_np, test_scores_np):.4f} "
        f"AUPRC: {average_precision_score(test_labels_np, test_scores_np):.4f}",
        flush=True,
    )
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    print(f"Saved weights to {weights_path}", flush=True)


if __name__ == "__main__":
    main()