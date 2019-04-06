import torch
import torch.nn.functional as F
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.fcn = nn.Linear(in_features=1024, out_features=hidden_size)

    def forward(self, imgs):
        x = imgs
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = x.view(x.size(0), -1)
        x = self.fcn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.fcn = nn.Linear(in_features=hidden_size, out_features=1024)
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.conv1_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.upsampel = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, imgs):
        x = imgs
        x = self.fcn(x)
        x = x.view(x.size(0), 64, 4, 4)

        x = self.upsampel(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)

        x = self.upsampel(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)

        x = self.upsampel(x)
        x = self.conv1_1(x)
        x = self.conv1_2(x)

        return x


class AutoEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = Encoder(hidden_size)
        self.decoder = Decoder(hidden_size)

    def forward(self, imgs):
        x = imgs
        x = self.encoder(x)
        x = self.decoder(x)
        return x
