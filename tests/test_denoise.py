"""ノイズ除去（engine/denoise.py）のテスト。

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.denoise import dehum, noise_floor_db, reduce_noise, spectral_gate  # noqa: E402

SR = 16000


def _tone(freq: float, seconds: float, amp: float = 0.2, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _magnitude_at(x: np.ndarray, freq: float, sr: int = SR) -> float:
    window = np.hanning(x.shape[0])
    spec = np.abs(np.fft.rfft(x.astype(np.float64) * window))
    freqs = np.fft.rfftfreq(x.shape[0], 1.0 / sr)
    return float(spec[np.argmin(np.abs(freqs - freq))])


class TestSpectralGate(unittest.TestCase):
    def test_off_returns_same_signal(self) -> None:
        x = _tone(220.0, 1.0)
        y = spectral_gate(x, SR, strength=0.0)
        np.testing.assert_allclose(x, y, atol=1e-6)

    def test_preserves_length(self) -> None:
        x = _tone(220.0, 1.0)
        y = spectral_gate(x, SR, strength=0.8)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(np.all(np.isfinite(y)))

    def test_reduces_stationary_noise(self) -> None:
        rng = np.random.default_rng(0)
        seconds = 3.0
        times = np.arange(int(SR * seconds)) / SR
        # 2 Hz でオンオフする「声」＋定常ノイズ
        gate = (np.sin(2 * np.pi * 2 * times) > 0).astype(np.float32)
        speech = (0.2 * np.sin(2 * np.pi * 220 * times) * gate).astype(np.float32)
        noise = (0.02 * rng.standard_normal(times.shape[0])).astype(np.float32)
        x = speech + noise

        before = noise_floor_db(x, SR)
        y = spectral_gate(x, SR, strength=0.8)
        after = noise_floor_db(y, SR)

        self.assertLess(after, before - 3.0, f"ノイズフロアが下がっていない: {before:.1f} → {after:.1f} dBFS")

        # 声の部分は壊していない（ノイズ除去後も元の波形と強く相関する）
        active = gate > 0
        corr = float(np.corrcoef(x[active], y[active])[0, 1])
        self.assertGreater(corr, 0.8, f"声の相関が低すぎる: {corr:.3f}")

    def test_silence_stays_silent(self) -> None:
        x = np.zeros(SR, dtype=np.float32)
        y = spectral_gate(x, SR, strength=0.9)
        self.assertLess(float(np.abs(y).max()), 1e-6)


class TestDehum(unittest.TestCase):
    def test_reduces_mains_hum_only(self) -> None:
        t = np.arange(int(SR * 2.0)) / SR
        speech = 0.1 * np.sin(2 * np.pi * 440 * t)
        hum = 0.2 * np.sin(2 * np.pi * 50 * t)
        x = (speech + hum).astype(np.float32)

        y = dehum(x, SR, mains=50.0)

        hum_before = _magnitude_at(x, 50.0)
        hum_after = _magnitude_at(y, 50.0)
        tone_before = _magnitude_at(x, 440.0)
        tone_after = _magnitude_at(y, 440.0)

        self.assertLess(hum_after, hum_before * 0.2, "50 Hz が十分に落ちていない")
        self.assertGreater(tone_after, tone_before * 0.9, "声の帯域まで削っている")


class TestNoiseFloor(unittest.TestCase):
    def test_quiet_is_lower_than_loud(self) -> None:
        rng = np.random.default_rng(1)
        quiet = (0.001 * rng.standard_normal(SR)).astype(np.float32)
        loud = (0.1 * rng.standard_normal(SR)).astype(np.float32)
        self.assertLess(noise_floor_db(quiet), noise_floor_db(loud) - 30.0)

    def test_reduce_noise_is_noop_when_disabled(self) -> None:
        x = _tone(220.0, 1.0)
        y = reduce_noise(x, SR, strength=0.0, dehum_hz=0.0)
        np.testing.assert_allclose(x, y)


if __name__ == "__main__":
    unittest.main()
