import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score
from pytorch_msssim import SSIM
import os
import json
from tqdm import tqdm

from utils import FabricDataset, fix_random


data_root = "data/fabric"
patch_size = 256
n_blocks = 4
selected_loss = "MSE"
learning_rate = 1e-3
epochs = 50
batch_size = 8
seed = 123
weights_path = "results/cae.pt"


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.first_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.first_bn = nn.BatchNorm2d(out_channels)

        self.second_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.second_bn = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.activation(x)

        x = self.second_conv(x)
        x = self.second_bn(x)

        x = identity + x  # skip connection

        x = self.activation(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.trans_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.first_bn = nn.BatchNorm2d(out_channels)

        self.conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.second_bn = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU()

        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.trans_conv(x)
        x = self.first_bn(x)
        x = self.activation(x)

        x = self.conv(x)
        x = self.second_bn(x)

        x = identity + x  # skip connection

        x = self.activation(x)

        return x


class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, n_blocks):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

        c_in = 16
        c_out = 32

        encoder_blocks = []
        for _ in range(n_blocks):
            encoder_blocks.append(
                EncoderBlock(c_in, c_out)
            )  # Each block halves spatial resolution
            c_in = c_out
            c_out *= 2  # Channels doubles across blocks
        self.encoder = nn.Sequential(*encoder_blocks)

        decoder_blocks = []
        for _ in range(n_blocks):
            c_out = c_in // 2  # Number of features halved after each block
            decoder_blocks.append(
                DecoderBlock(c_in, c_out)
            )  # Each block doubles the spatial resolution
            c_in = c_out
        self.decoder = nn.Sequential(*decoder_blocks)

        self.conv_1x1 = nn.Conv2d(
            in_channels=c_in, out_channels=3, kernel_size=1
        )  # Sets channels to RGB

    def forward(self, x):
        x = self.stem(x)
        x = self.encoder(x)
        x = self.decoder(x)

        x = self.conv_1x1(x)
        x = torch.sigmoid(x)

        return x


def train_cae(
    cae_model,
    train_dataloader,
    val_dataloader,
    num_epochs,
    lr,
    criterion,
    device,
    run_id,
):
    results_dir = os.path.join("./results", run_id)
    os.makedirs(results_dir, exist_ok=True)
    history_path = os.path.join(
        results_dir, "history.json"
    )  # Stores training/validation metrics
    weights_path = os.path.join(
        results_dir, "cae.pt"
    )  # Stores lowest validation loss weights

    train_history = []
    val_history = []

    optimizer = AdamW(cae_model.parameters(), lr=lr)

    step = 0  # optimization step count
    best_val_loss = float("inf")
    
    cae_model.to(device)

    for _ in tqdm(range(num_epochs)):
        # Training
        cae_model.train()
        for images, _, _ in train_dataloader:
            step += 1
            images = images.to(device)

            reconstructions = cae_model(images)
            loss = criterion(images, reconstructions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_history.append(
                {
                    "step": step,
                    "train_loss": loss.item(),
                }
            )

        # Validation (Sliding Window)
        cae_model.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            for images, _, _ in val_dataloader:
                images = images.to(device)

                reconstructions = sliding_window_inference(cae_model, images, patch_size)
                loss = criterion(images, reconstructions)

                val_epoch_loss += loss.item()
        
        avg_val_loss = val_epoch_loss / len(val_dataloader)  # Average validation loss per batch
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(cae_model.state_dict(), weights_path)
            print(f"--> Model Saved! Best Val Loss: {best_val_loss:.6f}")

        val_history.append(
            {"step": step, "val_loss": val_epoch_loss / len(val_dataloader)}
        )

        # Storing metrics after every epoch in case of run interruption
        history = {"train": train_history, "val": val_history}
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
            

def sliding_window_inference(model, image, patch_size=256):
    """
    Processes a large image patch-by-patch.
    Assumes image is (B, 3, H, W) and H, W are multiples of patch_size.
    """
    model.eval()
    B, C, H, W = image.shape
    reconstruction = torch.zeros_like(image)
    
    # Grid of patches
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = image[:, :, i:i+patch_size, j:j+patch_size]
            
            with torch.no_grad():
                recon_patch = model(patch)
                
            reconstruction[:, :, i:i+patch_size, j:j+patch_size] = recon_patch
            
    return reconstruction


def evaluate(model, dataloader, device, patch_size, criterion):
    """
    Computes per-image reconstruction error as anomaly scores.
    """
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for images, label, _ in dataloader:
            images = images.to(device)
            recon = sliding_window_inference(model, images, patch_size)
            # Per-image anomaly score
            for i in range(images.size(0)):
                scores.append(criterion(images[i:i+1], recon[i:i+1]).item())
            labels.extend(label.tolist())
    return scores, labels


def main():
    fix_random(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = "cuda"

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = FabricDataset("train", transforms=transform, root=data_root)
    val_dataset = FabricDataset("val", transforms=transform, root=data_root)
    test_dataset = FabricDataset("test", transforms=transform, root=data_root)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = ConvolutionalAutoEncoder(n_blocks)
    print(
        f"CUDA: {torch.cuda.get_device_name(0)} | "
        f"train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}",
        flush=True,
    )
    if selected_loss == "MSE":
        criterion = nn.MSELoss()
    elif selected_loss == "SSIM":
        ssim_module = SSIM(data_range=1.0, size_average=True, channel=3)
        criterion = lambda x, y: 1 - ssim_module(x, y)

    train_cae(model, train_loader, val_loader, epochs, learning_rate, criterion, device, "cae_run")

    # Load best checkpoint
    best_weights = os.path.join("results", "cae_run", "cae.pt")
    model.load_state_dict(torch.load(best_weights, map_location=device))
    model.to(device)

    test_scores, test_labels = evaluate(model, test_loader, device, patch_size, criterion)
    print(
        f"Held-out test Image-Level AUROC: {roc_auc_score(test_labels, test_scores):.4f} "
        f"AUPRC: {average_precision_score(test_labels, test_scores):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()