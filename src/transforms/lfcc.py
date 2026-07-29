import torch
import torchaudio
from torch import nn


class LFCC(nn.Module):
    """
    LFCC front-end. Turns a raw waveform into linear-frequency cepstral
    coefficients (static + delta + delta-delta), stacked along frequency.
    Input:  (B, T)          -- batch of waveforms
    Output: (B, 1, F, time) -- cepstral features (image-like, for the CNN)
    """

    def __init__(
        self,
        sample_rate=16000,
        n_lfcc=20,
        n_fft=512,
        win_length=400,
        hop_length=160,
        with_deltas=True,
    ):
        super().__init__()
        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            n_lfcc=n_lfcc,
            speckwargs={
                "n_fft": n_fft,
                "win_length": win_length,
                "hop_length": hop_length,
            },
        )
        self.with_deltas = with_deltas
        self.delta = torchaudio.transforms.ComputeDeltas()

    def forward(self, x):
        feat = self.lfcc(x)  # (B, n_lfcc, time)
        if self.with_deltas:
            d1 = self.delta(feat)  # (Δ)
            d2 = self.delta(d1)  # (ΔΔ)
            feat = torch.cat([feat, d1, d2], dim=1)  # (B, 3*n_lfcc, time)
        feat = feat.unsqueeze(1)  # (B, 1, F, time)
        return feat
