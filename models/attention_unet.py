import torch
import torch.nn as nn


# ---------------------------------------
# Residual Block
# ---------------------------------------

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()

        if in_channels != out_channels:

            self.skip = nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    1,
                    bias=False
                ),

                nn.BatchNorm2d(out_channels)

            )

    def forward(self, x):

        identity = self.skip(x)

        x = self.relu(self.bn1(self.conv1(x)))

        x = self.bn2(self.conv2(x))

        x = x + identity

        return self.relu(x)


# ---------------------------------------
# Attention Gate
# ---------------------------------------

class AttentionGate(nn.Module):

    def __init__(self, Fg, Fl, Fint):

        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(Fg, Fint, 1),
            nn.BatchNorm2d(Fint)
        )

        self.Wx = nn.Sequential(
            nn.Conv2d(Fl, Fint, 1),
            nn.BatchNorm2d(Fint)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(Fint, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        psi = self.relu(
            self.Wg(g) + self.Wx(x)
        )

        psi = self.psi(psi)

        return x * psi


# ---------------------------------------
# Decoder Block
# ---------------------------------------

class DecoderBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            2,
            stride=2
        )

        self.att = AttentionGate(
            out_channels,
            skip_channels,
            out_channels // 2
        )

        self.res = ResidualBlock(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):

        x = self.up(x)

        skip = self.att(x, skip)

        x = torch.cat([x, skip], dim=1)

        return self.res(x)


# ---------------------------------------
# Network
# ---------------------------------------

class AttentionUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.e1 = ResidualBlock(2, 64)

        self.e2 = ResidualBlock(64, 128)

        self.e3 = ResidualBlock(128, 256)

        self.e4 = ResidualBlock(256, 512)

        self.bridge = ResidualBlock(512, 1024)

        self.d4 = DecoderBlock(
            1024,
            512,
            512
        )

        self.d3 = DecoderBlock(
            512,
            256,
            256
        )

        self.d2 = DecoderBlock(
            256,
            128,
            128
        )

        self.d1 = DecoderBlock(
            128,
            64,
            64
        )

        self.final = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(self, x):

        s1 = self.e1(x)

        s2 = self.e2(self.pool(s1))

        s3 = self.e3(self.pool(s2))

        s4 = self.e4(self.pool(s3))

        b = self.bridge(self.pool(s4))

        d4 = self.d4(b, s4)

        d3 = self.d3(d4, s3)

        d2 = self.d2(d3, s2)

        d1 = self.d1(d2, s1)

        return self.final(d1)