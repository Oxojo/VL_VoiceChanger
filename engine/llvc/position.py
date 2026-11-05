"""Sinusoidal positional encoding; compatible with LLVC's SpeechBrain buffer.

Avoid importing SpeechBrain/torchaudio/fairseq for inference. The persistent
pe buffer and its shape intentionally match the pretrained checkpoint keys.
"""
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, input_size, max_len=2500):
        super().__init__()
        angles = torch.arange(max_len, dtype=torch.float32)[:, None] * torch.exp(
            torch.arange(0, input_size, 2, dtype=torch.float32) * (-math.log(10000.0) / input_size)
        )
        encoding = torch.empty(1, max_len, input_size)
        encoding[0, :, 0::2] = angles.sin()
        encoding[0, :, 1::2] = angles.cos()
        self.register_buffer('pe', encoding)

    def forward(self, x):
        return self.pe[:, :x.shape[1]].clone().detach()
