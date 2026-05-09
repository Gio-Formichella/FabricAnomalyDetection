import torch
import torch.nn as nn
from torch.optim import AdamW
import os
import json
from tqdm import tqdm


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
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
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
    selected_loss,
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

    if selected_loss == "MSE":
        criterion = nn.MSELoss()
    elif selected_loss == "SSIM":
        criterion = ""  # TODO

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

        # Validation
        cae_model.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            for images, _, _ in val_dataloader:
                images = images.to(device)

                reconstructions = cae_model(images)
                loss = criterion(images, reconstructions)

                val_epoch_loss += loss

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(cae_model.state_dict(), weights_path)

        val_history.append(
            {"step": step, "val_loss": val_epoch_loss.item() / len(val_dataloader)}
        )

        # Storing metrics after every epoch in case of run interruption
        history = {"train": train_history, "val": val_history}
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
