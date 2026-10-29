"""自前エンジンの最小テスト。

実行:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.audio import load_mono, resample, save_wav  # noqa: E402
from engine.convert import f0_to_coarse, interpolate_unvoiced  # noqa: E402
from engine.model.commons import weight_norm  # noqa: E402
from engine.model.synthesizer import SynthesizerTrnMs768NSFsid  # noqa: E402

import torch  # noqa: E402


class TestF0(unittest.TestCase):
    def test_coarse_range(self) -> None:
        f0 = np.array([0.0, 50.0, 1100.0, 2000.0], dtype=np.float64)
        coarse = f0_to_coarse(f0)
        self.assertEqual(coarse.min(), 1)
        self.assertEqual(coarse.max(), 255)
        self.assertEqual(coarse[1], 1)  # 下限は 1
        self.assertTrue(np.all(np.diff(coarse) >= 0))  # 単調増加

    def test_interpolate_unvoiced(self) -> None:
        f0 = np.array([0.0, 0.0, 200.0, 0.0, 300.0, 0.0], dtype=np.float32)
        out = interpolate_unvoiced(f0)
        self.assertTrue(np.all(out > 0))
        self.assertAlmostEqual(float(out[0]), 200.0, places=4)
        self.assertTrue(200.0 < out[3] < 300.0)

    def test_interpolate_all_unvoiced(self) -> None:
        out = interpolate_unvoiced(np.zeros(10, dtype=np.float32))
        self.assertTrue(np.all(out == 0))


class TestAudio(unittest.TestCase):
    def test_resample_identity(self) -> None:
        x = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
        np.testing.assert_array_equal(resample(x, 16000, 16000), x)

    def test_wav_roundtrip(self) -> None:
        sr = 16000
        t = np.arange(sr, dtype=np.float32) / sr
        x = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            save_wav(path, x, sr)
            y = load_mono(path, sr)
        self.assertEqual(y.shape[0], x.shape[0])
        self.assertLess(float(np.max(np.abs(x - y))), 0.01)


class TestWeightNorm(unittest.TestCase):
    def test_forward_matches_definition(self) -> None:
        torch.manual_seed(0)
        conv = torch.nn.Conv1d(4, 6, 3, padding=1)
        module = weight_norm(conv)
        self.assertIn("weight_g", dict(module.named_parameters()))
        self.assertIn("weight_v", dict(module.named_parameters()))

        g = module.weight_g
        v = module.weight_v
        expected = v * (g / torch.linalg.vector_norm(v, dim=(1, 2), keepdim=True))
        np.testing.assert_allclose(module.weight.detach().numpy(), expected.detach().numpy(), rtol=1e-6)

        x = torch.randn(2, 4, 8)
        np.testing.assert_allclose(module(x).detach().numpy(), torch.nn.functional.conv1d(x, expected, module.bias, padding=1).detach().numpy(), rtol=1e-5, atol=1e-6)


class TestSynthesizerShapes(unittest.TestCase):
    """極小構成で推論経路のテンソル形状だけを確認する。"""

    def test_infer_shape(self) -> None:
        torch.manual_seed(0)
        model = SynthesizerTrnMs768NSFsid(
            spec_channels=8,
            segment_size=2,
            inter_channels=8,
            hidden_channels=8,
            filter_channels=8,
            n_heads=2,
            n_layers=1,
            kernel_size=3,
            p_dropout=0.0,
            resblock="1",
            resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1, 3, 5]],
            upsample_rates=[2, 2],
            upsample_initial_channel=8,
            upsample_kernel_sizes=[4, 4],
            spk_embed_dim=2,
            gin_channels=4,
            sr=8000,
            with_posterior=False,
        ).eval()

        t = 5
        phone = torch.randn(1, t, 768)
        lengths = torch.tensor([t])
        pitch = torch.randint(1, 255, (1, t))
        f0 = torch.full((1, t), 200.0)
        sid = torch.tensor([0])

        with torch.no_grad():
            out = model.infer(phone, lengths, pitch, f0, sid, generator=torch.Generator().manual_seed(0))

        self.assertEqual(out.shape, (1, 1, t * 4))  # 8 kHz / 100 fps = 80 ではなく、hop=4 の極小構成
        self.assertTrue(torch.all(torch.abs(out) <= 1.0))


class TestChunkedConverter(unittest.TestCase):
    """区間の切り出しと連結の長さ・ずれを、合成器を使わずに確認する。"""

    class _FakeConverter:
        class _Info:
            sample_rate = 40000

        info = _Info()

        def __init__(self) -> None:
            self.segments: list[int] = []

        def convert_array(self, audio: np.ndarray, options) -> tuple[np.ndarray, None]:
            self.segments.append(int(audio.shape[0]))
            # 実機と同じく「10 ms フレーム数 × 400 サンプル」を返す
            frames = audio.shape[0] // 160
            # 区間の先頭が分かるよう、通し番号を振幅として埋め込む
            out = np.full(frames * 400, float(len(self.segments)), dtype=np.float32)
            return out, None

    def test_length_is_frame_exact(self) -> None:
        from engine.realtime import ChunkConfig, ChunkedConverter

        fake = self._FakeConverter()
        chunked = ChunkedConverter(
            fake,  # type: ignore[arg-type]
            ChunkConfig(hop_seconds=0.3, context_seconds=0.1, crossfade_seconds=0.05),
        )
        audio = np.zeros(16000, dtype=np.float32)  # 1.0 秒 = 100 フレーム
        out, stats = chunked.process(audio)

        self.assertEqual(out.shape[0], 100 * 400)  # 端数なく全フレームが埋まる
        self.assertEqual(stats.chunks, 4)  # 0.3 秒ごと → 30 + 30 + 30 + 10 フレーム
        self.assertAlmostEqual(stats.audio_seconds, 1.0, places=6)

    def test_segments_include_context_and_crossfade(self) -> None:
        from engine.realtime import ChunkConfig, ChunkedConverter

        fake = self._FakeConverter()
        chunked = ChunkedConverter(
            fake,  # type: ignore[arg-type]
            ChunkConfig(hop_seconds=0.5, context_seconds=0.1, crossfade_seconds=0.05),
        )
        audio = np.zeros(16000 * 2, dtype=np.float32)  # 2.0 秒
        chunked.process(audio)

        # 2 区間目以降は「文脈 0.1 + 本体 0.5 + 文脈 0.1 + クロスフェード 0.05」
        self.assertEqual(fake.segments[0], int(0.5 * 16000) + int(0.1 * 16000) + int(0.05 * 16000))
        self.assertEqual(fake.segments[1], int((0.1 + 0.5 + 0.1 + 0.05) * 16000))


if __name__ == "__main__":
    unittest.main()
