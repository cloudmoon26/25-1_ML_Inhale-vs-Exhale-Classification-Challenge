import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.down = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        )

    def forward(self, x):
        identity = self.down(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return self.relu(out + identity)


class BreathNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            ResidualBlock(1, 16),
            nn.MaxPool2d(2),      # (16, 64, 32)

            ResidualBlock(16, 32),
            nn.MaxPool2d(2),      # (32, 32, 16)

            ResidualBlock(32, 64),
            nn.MaxPool2d(2),      # (64, 16, 8)

            ResidualBlock(64, 128),
            nn.MaxPool2d(2),      # (128, 8, 4)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.cls(x)
