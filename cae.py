import torch
import torch.nn as nn


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
