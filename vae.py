import argparse
import copy
import os

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from utils import FabricDataset, fix_random


class CenterCropFraction:
    def __init__(self, fraction):
        self.fraction = fraction

    def __call__(self, image):
        width, height = image.size
        return TF.center_crop(
            image, [int(height * self.fraction), int(width * self.fraction)]
        )


class FeatureVAE(nn.Module):
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
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z), mu, logvar

    def reconstruct(self, x):
        return self.decoder(self.mu(self.encoder(x)))



def cache_images(split, transform, root):
    dataset = FabricDataset(split, transforms=transform, root=root)
    images, labels = [], []
    for image, label, _ in dataset:
        images.append(image)
        labels.append(label)
    return torch.stack(images), torch.tensor(labels)


def patch_features(images, patch_size):
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    n_images, grid_h, grid_w, _, _, _ = patches.shape
    patches = patches.view(-1, 3, patch_size, patch_size)

    gray = 0.299 * patches[:, 0] + 0.587 * patches[:, 1] + 0.114 * patches[:, 2]
    gray = (gray - gray.mean(dim=(1, 2), keepdim=True)) / gray.std(
        dim=(1, 2), keepdim=True
    ).clamp_min(1e-4)
    dx = gray[:, :, 1:] - gray[:, :, :-1]
    dy = gray[:, 1:, :] - gray[:, :-1, :]
    fft = torch.fft.rfft2(gray, norm="ortho").abs()
    fft = torch.log1p(fft[:, : patch_size // 2, : patch_size // 2])

    features = torch.cat(
        [gray.flatten(1), dx.flatten(1), dy.flatten(1), fft.flatten(1)],
        dim=1,
    )
    return features, n_images, grid_h * grid_w


def standardize(train, *others):
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1e-4)
    return (train - mean) / std, tuple((x - mean) / std for x in others)


def vae_loss(x_hat, x, mu, logvar, beta):
    recon = nn.functional.mse_loss(x_hat, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld, recon, kld


def reconstruction_loss(model, loader, device):
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


def train(model, train_loader, val_loader, optimizer, device, epochs, beta, noise_std):
    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = total_recon = total_kld = 0.0
        for (features,) in train_loader:
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

        val_recon = reconstruction_loss(model, val_loader, device)
        if val_recon < best_val:
            best_val = val_recon
            best_state = copy.deepcopy(model.state_dict())
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
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in DataLoader(TensorDataset(features), batch_size=8192):
            x = batch[0].to(device, non_blocking=True)
            recon = model.reconstruct(x)
            scores.append((x - recon).pow(2).mean(dim=1).cpu())
    return torch.cat(scores).view(n_images, patches_per_image)


def score_images(model, features, n_images, patches_per_image, device, topk, normalizer):
    patch_scores = patch_errors(model, features, n_images, patches_per_image, device)
    patch_scores = patch_scores / normalizer
    k = max(1, int(patches_per_image * topk))
    return patch_scores.topk(k, dim=1).values.mean(dim=1)


def select_topk(model, features, labels, n_images, patches_per_image, device, normalizer):
    candidates = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    best_topk = candidates[0]
    best_auroc = -1.0
    for topk in candidates:
        scores = score_images(
            model, features, n_images, patches_per_image, device, topk, normalizer
        )
        auroc = roc_auc_score(labels.numpy(), scores.numpy())
        if auroc > best_auroc:
            best_topk = topk
            best_auroc = auroc
    return best_topk, best_auroc


def make_synthetic_anomalies(images, patch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    anomalous = images.clone()
    _, _, height, width = anomalous.shape
    box = patch_size * 2
    for image in anomalous:
        y = torch.randint(0, height - box + 1, (1,), generator=generator).item()
        x = torch.randint(0, width - box + 1, (1,), generator=generator).item()
        image[:, y : y + box, x : x + box] = torch.rand(
            (3, box, box), generator=generator
        )
    labels = torch.cat(
        [
            torch.zeros(len(images), dtype=torch.long),
            torch.ones(len(anomalous), dtype=torch.long),
        ]
    )
    return torch.cat([images, anomalous]), labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/fabric")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--crop-fraction", type=float, default=0.8)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--topk", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--weights-path", default="results/vae.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    fix_random(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.size % args.patch_size:
        raise ValueError("--size must be divisible by --patch-size")

    device = "cuda"
    transform = transforms.Compose(
        [
            CenterCropFraction(args.crop_fraction),
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
        ]
    )
    train_images, _ = cache_images("train", transform, args.data_root)
    val_images, _ = cache_images("val", transform, args.data_root)
    test_images, test_labels = cache_images("test", transform, args.data_root)
    synth_images, synth_labels = make_synthetic_anomalies(
        val_images, args.patch_size, args.seed
    )

    train_features, n_train, train_ppi = patch_features(train_images, args.patch_size)
    val_features, _, _ = patch_features(val_images, args.patch_size)
    test_features, n_test, test_ppi = patch_features(test_images, args.patch_size)
    synth_features, n_synth, synth_ppi = patch_features(synth_images, args.patch_size)
    train_features, (val_features, test_features, synth_features) = standardize(
        train_features, val_features, test_features, synth_features
    )

    train_loader = DataLoader(
        TensorDataset(train_features),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_features),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    model = FeatureVAE(train_features.shape[1], args.hidden_dim, args.latent_dim).to(
        device
    )
    optimizer = AdamW(model.parameters(), lr=args.lr)
    print(
        f"CUDA: {torch.cuda.get_device_name(0)} | "
        f"train={len(train_images)} val={len(val_images)} test={len(test_images)} | "
        f"features={train_features.shape[1]} train_patches={len(train_features)}",
        flush=True,
    )
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        args.epochs,
        args.beta,
        args.noise_std,
    )
    train_errors = patch_errors(model, train_features, n_train, train_ppi, device)
    normalizer = train_errors.mean(dim=0, keepdim=True).clamp_min(1e-6)

    selected_topk, synth_auroc = select_topk(
        model, synth_features, synth_labels, n_synth, synth_ppi, device, normalizer
    )
    synth_scores = score_images(
        model, synth_features, n_synth, synth_ppi, device, selected_topk, normalizer
    )
    print(
        f"Synthetic validation AUROC: {synth_auroc:.4f} "
        f"selected_topk={selected_topk:.4f}",
        flush=True,
    )
    test_scores = score_images(
        model, test_features, n_test, test_ppi, device, selected_topk, normalizer
    )
    print(
        f"Held-out test Image-Level AUROC: "
        f"{roc_auc_score(test_labels.numpy(), test_scores.numpy()):.4f}",
        flush=True,
    )
    os.makedirs(os.path.dirname(args.weights_path), exist_ok=True)
    torch.save(model.state_dict(), args.weights_path)
    print(f"Saved weights to {args.weights_path}", flush=True)


if __name__ == "__main__":
    main()
