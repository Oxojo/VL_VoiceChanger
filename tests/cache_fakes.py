"""A real VoiceConverter pipeline with model-free feature/model backends."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from engine.convert import VoiceConverter


def make_converter(sample_rate=48000):
    converter = VoiceConverter.__new__(VoiceConverter)
    converter.device = "cpu"
    converter.info = SimpleNamespace(sample_rate=sample_rate)
    session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

    def f0(samples):
        return np.where(np.arange(samples.size // 160) % 2, 110, 0).astype(np.float32)

    def phone(samples, layer):
        return np.ones((1, max(1, samples.size // 320), 4), dtype=np.float32)

    def synth(phone, lengths, pitch, pitchf, sid, generator=None):
        return torch.zeros(1, 1, int(lengths.item()) * (sample_rate // 100))

    converter.rmvpe = SimpleNamespace(session=session, infer=Mock(side_effect=f0))
    converter.contentvec = SimpleNamespace(session=session, extract=Mock(side_effect=phone))
    converter.model = SimpleNamespace(infer=Mock(side_effect=synth))
    return converter
