"""Web UI（engine/web）と評価指標（engine/metrics）のテスト。

サーバーは起動せず、純粋な関数だけを確認する。
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

from fastapi import HTTPException  # noqa: E402

from engine import metrics  # noqa: E402
from engine.web import server  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_f0_stats(self) -> None:
        f0 = np.array([0.0, 0.0, 200.0, 300.0], dtype=np.float32)
        median, ratio = metrics.f0_stats(f0)
        self.assertAlmostEqual(median, 250.0, places=4)
        self.assertAlmostEqual(ratio, 50.0, places=4)

    def test_f0_correlation_identity(self) -> None:
        rng = np.random.default_rng(0)
        f0 = rng.uniform(100, 300, 200).astype(np.float32)
        self.assertAlmostEqual(metrics.f0_correlation(f0, f0), 1.0, places=5)

    def test_envelope_correlation_identity(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(16000).astype(np.float32)
        self.assertAlmostEqual(metrics.envelope_correlation(x, x), 1.0, places=5)

    def test_compare_keys_and_clip(self) -> None:
        rng = np.random.default_rng(1)
        x = (0.1 * rng.standard_normal(16000)).astype(np.float32)
        loud = (x * 20.0).astype(np.float32)
        f = np.full(100, 200.0, dtype=np.float32)
        out = metrics.compare(x, loud, f, f)
        self.assertIn("f0_correlation", out)
        self.assertIn("envelope_correlation", out)
        self.assertTrue(out["clipped"])
        self.assertFalse(metrics.compare(x, x, f, f)["clipped"])

    def test_boundary_jumps(self) -> None:
        x = np.zeros(40000, dtype=np.float32)
        x[20000] = 0.5  # 0.5 秒の位置に段差
        mean_jump, max_jump = metrics.boundary_jumps(x, 40000, 0.5)
        self.assertAlmostEqual(max_jump, 0.5, places=5)
        self.assertGreaterEqual(mean_jump, 0.0)


class TestWebHelpers(unittest.TestCase):
    def test_list_models_has_ids(self) -> None:
        models = server.list_models()
        self.assertIsInstance(models, list)
        for m in models:
            self.assertIn("id", m)
            self.assertNotIn("\\", m["id"])  # 区切りは / に統一
            self.assertFalse(m["id"].endswith(".pth"))

    def test_model_path_rejects_traversal(self) -> None:
        for bad in ("../secrets", "..\\..\\windows\\system32\\config", "/etc/passwd"):
            with self.assertRaises(HTTPException) as ctx:
                server.model_path(bad)
            self.assertEqual(ctx.exception.status_code, 404)

    def test_model_path_accepts_listed_model(self) -> None:
        models = server.list_models()
        if not models:
            self.skipTest("models/ に .pth がありません")
        path = server.model_path(models[0]["id"])
        self.assertTrue(path.is_file())
        self.assertTrue(str(path).endswith(".pth"))


class _FakeUpload:
    """UploadFile の代わりに、用意したバイト列を返すだけの入れ物。"""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self) -> bytes:
        return self._data


def _wav_bytes(seconds: float = 1.0, sr: int = 16000) -> bytes:
    import io

    import soundfile as sf

    rng = np.random.default_rng(0)
    t = np.arange(int(sr * seconds)) / sr
    x = (0.2 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(t.shape[0])).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TestDenoiseEndpoint(unittest.TestCase):
    """入力音声だけにノイズ除去をかける API（変換前の任意ステップ）。"""

    def test_denoise_returns_metrics_and_audio(self) -> None:
        import asyncio

        result = asyncio.run(
            server.denoise_input(_FakeUpload(_wav_bytes()), strength=0.7, dehum=50.0)  # type: ignore[arg-type]
        )
        self.assertEqual(result["strength"], 0.7)
        self.assertEqual(result["dehum"], 50.0)
        self.assertAlmostEqual(result["duration"], 1.0, places=2)
        # ノイズフロアが下がっている
        self.assertLess(result["after_db"], result["before_db"] - 3.0)
        # 保存された音声を取得できる
        path = server.DENOISE_DIR / result["id"] / "clean.wav"
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 1000)

    def test_denoise_rejects_unknown_dehum(self) -> None:
        import asyncio

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                server.denoise_input(_FakeUpload(_wav_bytes()), strength=0.5, dehum=33.0)  # type: ignore[arg-type]
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_denoise_rejects_broken_audio(self) -> None:
        import asyncio

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                server.denoise_input(_FakeUpload(b"not a wav"), strength=0.5, dehum=0.0)  # type: ignore[arg-type]
            )
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
