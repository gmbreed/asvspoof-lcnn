import torch

from src.metrics.calculate_eer import compute_eer


class EERCalculator:
    """
    Accumulates CM scores over a full evaluation epoch and computes EER once.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.scores = []
        self.labels = []

    def update(self, logits, labels, **batch):
        score = torch.softmax(logits.detach(), dim=-1)[:, 1]
        self.scores.append(score.cpu())
        self.labels.append(labels.detach().cpu())

    def compute(self):
        scores = torch.cat(self.scores).numpy()
        labels = torch.cat(self.labels).numpy()

        bonafide_scores = scores[labels == 1]
        spoof_scores = scores[labels == 0]

        eer, _ = compute_eer(bonafide_scores, spoof_scores)
        return eer * 100
