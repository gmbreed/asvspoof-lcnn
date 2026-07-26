import torch
import torchaudio
from torch import nn


class LogSpectrogram(nn.Module):
    """
    STFT front-end. Turns a raw waveform into a log-spectrogram.
    Input:  (B, T)          -- batch of waveforms
    Output: (B, 1, F, time) -- log-spectrogram (image-like, for the CNN)
    """

    def __init__(self, n_fft=512, hop_length=256, win_length=512, power=2.0):
        super().__init__()
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=power,
        )

    def forward(self, x):
        spec = self.spectrogram(x)
        spec = torch.log(spec.clamp_(min=1e-9, max=1e9))
        spec = spec.unsqueeze(1)
        return spec
