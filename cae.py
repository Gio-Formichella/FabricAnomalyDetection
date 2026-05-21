import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from sklearn.metrics import roc_auc_score, average_precision_score
from pytorch_msssim import SSIM
import os
import json
from tqdm import tqdm

from utils import FabricDataset, fix_random, plot_loss, plot_anomaly_results

data_root = "data/fabric"
patch_size = 256
n_blocks = 5
selected_loss = "SSIM+MSE"
learning_rate = 1e-3
epochs = 50
batch_size = 32
seed = 42
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

    def forward(self, x):
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.activation(x)
        x = self.second_conv(x)
        x = self.second_bn(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.trans_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.first_bn = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.second_bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.first_bn(self.trans_conv(x)))
        x = self.activation(self.second_bn(self.conv(x)))
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

                reconstructions = sliding_window_inference(
                    cae_model, images, patch_size
                )
                loss = criterion(images, reconstructions)

                val_epoch_loss += loss.item()

        avg_val_loss = val_epoch_loss / len(
            val_dataloader
        )  # Average validation loss per batch

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


def sliding_window_inference(model, image, patch_size=256, stride=128):
    B, C, H, W = image.shape
    reconstruction = torch.zeros_like(image)
    count = torch.zeros(B, 1, H, W, device=image.device)

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image[:, :, y : y + patch_size, x : x + patch_size]
            with torch.no_grad():
                recon_patch = model(patch)
            reconstruction[:, :, y : y + patch_size, x : x + patch_size] += recon_patch
            count[:, :, y : y + patch_size, x : x + patch_size] += 1

    reconstruction /= count.clamp(min=1)
    return reconstruction


def evaluate(model, dataloader, device, patch_size):
    model.eval()
    scores = []
    labels = []

    # element-wise MSE
    mse_elementwise = torch.nn.MSELoss(reduction="none")

    with torch.no_grad():
        for images, label, _ in dataloader:
            images = images.to(device)
            recon = sliding_window_inference(model, images, patch_size)

            # Compute element-wise squared error shape: (B, C, H, W)
            error_map = mse_elementwise(images, recon)

            # Average across channels to get a single spatial heatmap: (B, H, W)
            error_map = torch.mean(error_map, dim=1)

            # Flatten spatial dimensions to easily search pixels: (B, H*W)
            error_flat = error_map.view(images.size(0), -1)

            # Extract anomaly scores per image
            # STRATEGY A: Take the max error pixel (highly sensitive to tiny defects)
            # batch_scores = torch.max(error_flat, dim=1)[0]

            # STRATEGY B: Take the mean of the top 1% highest error pixels
            # (More robust to random noise than absolute Max)
            top_k = max(1, int(error_flat.size(1) * 0.01))
            top_errors, _ = torch.topk(error_flat, k=top_k, dim=1)
            batch_scores = torch.mean(top_errors, dim=1)

            scores.extend(batch_scores.cpu().tolist())
            labels.extend(label.tolist())

    return scores, labels


def main():
    fix_random(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = "cuda"

    resize_res = (2048, 2304)

    transforms = {
        "train": v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(resize_res),
                v2.RandomCrop(patch_size),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
            ]
        ),
        "test": v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize(resize_res)]
        ),
    }

    train_set = FabricDataset(
        root=data_root,
        split="train",
        transforms=transforms["train"],
        target_res=resize_res,
    )
    val_set = FabricDataset(
        root=data_root,
        split="val",
        transforms=transforms["test"],
        target_res=resize_res,
    )
    test_set = FabricDataset(
        root=data_root,
        split="test",
        transforms=transforms["test"],
        target_res=resize_res,
    )

    num_workers = 4

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )  # Images 256 x 256

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )  # Images 2304 x 2048

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )  # Images 2304 x 2048

    model = ConvolutionalAutoEncoder(n_blocks).to(device)
    print(
        f"CUDA: {torch.cuda.get_device_name(0)} | "
        f"train={len(train_set)} val={len(val_set)} test={len(test_set)}",
        flush=True,
    )
    run_id = f"sliding_window_cae_{selected_loss}-loss"
    if selected_loss == "MSE":
        criterion = nn.MSELoss()
    elif selected_loss == "SSIM+MSE":
        ssim_module = SSIM(data_range=1.0, size_average=True, channel=3)
        criterion = lambda x, y: 0.5 * nn.MSELoss()(x, y) + 0.5 * (
            1 - ssim_module(x, y)
        )

    if not os.path.exists(
        os.path.join("./results", run_id, "cae.pt")
    ):  # retrain if not already trained
        train_cae(
            model,
            train_loader,
            val_loader,
            epochs,
            learning_rate,
            criterion,
            device,
            run_id,
        )
    plot_loss(run_id)

    # Load best checkpoint
    best_weights = os.path.join("results", run_id, "cae.pt")
    model.load_state_dict(torch.load(best_weights, map_location=device))
    model.to(device)

    test_scores, test_labels = evaluate(model, test_loader, device, patch_size)
    print(
        f"Held-out test Image-Level AUROC: {roc_auc_score(test_labels, test_scores):.4f} "
        f"PR-AUC: {average_precision_score(test_labels, test_scores):.4f}",
        flush=True,
    )

    model.eval()
    with torch.no_grad():
        # Select an example from the test set (last are anomalies)
        image, label, gt_mask = test_set[-1]

        # Add batch dimension and move to device
        image_batch = image.unsqueeze(0).to(device)

        # Get reconstruction
        reconstruction_batch = sliding_window_inference(model, image_batch, patch_size)

        # Compute element-wise MSE for anomaly map
        mse_elementwise = torch.nn.MSELoss(reduction="none")
        error_map = mse_elementwise(image_batch, reconstruction_batch)
        anomaly_map = torch.mean(error_map, dim=1).squeeze(
            0
        )  # Average across channels, remove batch dim

        # Plot the results
        print(f"Displaying example: Label = {label} (0=good, 1=bad)")
        plot_anomaly_results(
            original_image=image,
            reconstruction=reconstruction_batch.squeeze(0).cpu(),
            anomaly_map=anomaly_map,
            gt_mask=gt_mask,
        )


if __name__ == "__main__":
    main()
