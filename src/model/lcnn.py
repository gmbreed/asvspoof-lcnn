import math

import torch
import torch.nn.functional as F
from torch import nn


class MFM(nn.Module):
    """
    Max-Feature-Map activation.
    Splits the channels into two halves and takes the max of two elements.

    Input:  (B, 2N, H, W)  or  (B, 2N)
    Output: (B, N,  H, W)  or  (B, N)
    """

    def forward(self, x):
        half = x.shape[1] // 2
        return torch.max(x[:, :half], x[:, half:])


class AngleLinear(nn.Module):
    """
    A-Softmax.

    Normalizes the class weights but KEEPS the feature norm ||x|| (this is what
    distinguishes A-Softmax from AM-Softmax). Returns TWO logits per class:
      - cos_theta * ||x||   : plain angular logit (used for scoring/inference)
      - phi_theta  * ||x||   : margined logit built from cos(m*theta) (training)

    The multiplicative margin m forces the true class to occupy an m-times
    smaller angular sector -> bonafide is squeezed into a tight cone and every
    spoof (incl. unseen attacks) is pushed out. m=4 is canonical SphereFace.
    """

    def __init__(self, in_features, out_features, m=4):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_normal_(self.weight)
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2 * x**2 - 1,
            lambda x: 4 * x**3 - 3 * x,
            lambda x: 8 * x**4 - 8 * x**2 + 1,
            lambda x: 16 * x**5 - 20 * x**3 + 5 * x,
        ]

    def forward(self, x):
        w = F.normalize(self.weight, dim=0)
        x_len = x.norm(dim=1, keepdim=True)

        cos_theta = (x / x_len) @ w
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)

        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.detach().acos()
        k = (self.m * theta / math.pi).floor()
        sign = 1.0 - 2.0 * (k % 2)
        phi_theta = sign * cos_m_theta - 2 * k

        cos_theta = cos_theta * x_len
        phi_theta = phi_theta * x_len
        return cos_theta, phi_theta


class LCNN(nn.Module):
    def __init__(
        self, n_class=2, input_dim=(257, 251), dropout=0.7, angular=False, angular_m=4
    ):
        super().__init__()
        self.angular = angular
        self.conv_layers = nn.Sequential(
            # block 1: 1 -> 32 channels, (H & W) // 2
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 2: 32 -> 48 channels, (H & W) // 2
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            # block 3: 48 -> 64 channels, (H & W) // 2
            nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0),
            MFM(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block 4: 64 -> 32 channels, (H & W) // 2
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            MFM(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # asking conv_layers what size it outputs
        # easily avoid hardcoding (gg ez)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_dim)
            n_flat = self.conv_layers(dummy).flatten(1).shape[1]

        fc_layers = [
            nn.Flatten(),
            nn.Linear(n_flat, 160),
            MFM(),  # 160 -> 80
            nn.Dropout(dropout),
            nn.BatchNorm1d(80),
        ]
        if angular:
            # embedding stops at BN(80); A-Softmax head is separate
            self.fc_layers = nn.Sequential(*fc_layers)
            self.head = AngleLinear(80, n_class, m=angular_m)
        else:
            # plain head kept INSIDE fc_layers -> identical to the CE baseline
            fc_layers.append(nn.Linear(80, n_class))
            self.fc_layers = nn.Sequential(*fc_layers)
            self.head = None

    def forward(self, data_object, **batch):
        x = self.conv_layers(data_object)
        out = self.fc_layers(x)
        if self.angular:
            cos_theta, phi_theta = self.head(out)
            return {"logits": cos_theta, "phi_theta": phi_theta}
        return {"logits": out}

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
