import torch
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


class LCNN(nn.Module):
    def __init__(self, n_class=2, input_dim=(257, 251), dropout=0.7):
        super().__init__()
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

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 160),
            MFM(),  # 160 -> 80
            nn.Dropout(dropout),
            nn.BatchNorm1d(80),
            nn.Linear(80, n_class),
        )

    def forward(self, data_object, **batch):
        x = self.conv_layers(data_object)
        logits = self.fc_layers(x)
        return {"logits": logits}

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
