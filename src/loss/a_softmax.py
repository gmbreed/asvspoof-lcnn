import torch
from torch import nn


class ASoftmaxLoss(nn.Module):
    """
    A-Softmax loss (SphereFace, Liu et al. 2017) -- the angular-margin loss
    named in the assignment README (Hint 1: "A-Softmax or Cross-Entropy").

    Consumes the two logits from the model's AngleLinear head:
        logits    = cos(theta) * ||x||        (plain)
        phi_theta = psi(theta) * ||x||        (margined, from cos(m*theta))

    For the true class it blends the plain and margined logits with an annealed
    weight lambda:
        out_true = (lambda * logits + phi_theta) / (1 + lambda)
    lambda starts huge (margin almost off -> ordinary softmax, stable start)
    and decays each step toward `lambda_min`, so the angular margin gradually
    switches on. This is the standard SphereFace annealing.

    Args:
        lambda_min (float): floor lambda (full margin strength).
        lambda_max (float): starting lambda (margin nearly off).
        weight (list | None): per-class weights for imbalance (bonafide<<spoof).
    """

    def __init__(self, lambda_min=5.0, lambda_max=1500.0, weight=None):
        super().__init__()
        self.it = 0
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, phi_theta, labels, **batch):
        self.it += 1
        # lambda decays from lambda_max toward lambda_min as training proceeds
        lamb = max(self.lambda_min, self.lambda_max / (1 + 0.1 * self.it))

        index = torch.zeros_like(logits)
        index.scatter_(1, labels.view(-1, 1), 1.0)
        index = index.bool()

        output = logits.clone()
        # only the true class gets the (annealed) angular margin
        output[index] = (lamb * logits[index] + phi_theta[index]) / (1 + lamb)

        return {"loss": self.ce(output, labels)}
