import torch
from torch import nn


class CELoss(nn.Module):
    """
    Cross-Entropy loss with optional per-class weights (for class imbalance).
    """

    def __init__(self, weight=None):
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, labels, **batch):
        return {"loss": self.loss(logits, labels)}
