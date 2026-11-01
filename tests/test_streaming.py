"""Contract tests for incrementally received audio; no model weights required."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from engine.convert import ConvertOptions
from engine.realtime import ChunkConfig, FRAME
from engine.streaming import StreamingChunkedConverter


class IdentityConverter:
    """Frame-preserving mock matching the RVC converter output contract."""

    def __init__(self) -> None:
        self.info = SimpleNamespace(sample_rate=16000)
        self.segments: list[int] = []

    def convert_array(self, samples: np.ndarray, options):
        self.segments.append(samples.size)
        return samples.astype(np.float32, copy=True), None


class TestStreamingChunkedConverter(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = ChunkConfig(hop_seconds=0.20, context_seconds=0.06, crossfade_seconds=0.04)
        self.fake = IdentityConverter()

    def make_stream(self, options: ConvertOptions | None = None) -> StreamingChunkedConverter:
        return StreamingChunkedConverter(self.fake, self.cfg, options)

    def test_arbitrary_input_boundaries_and_exact_output(self) -> None:
        # Ignore the 13 source samples that do not complete a 10 ms frame.
        audio = np.linspace(-0.5, 0.5, 16000 + 13, dtype=np.float32)
        stream = self.make_stream()
        pieces = []
        i = 0
        for length in [1, 37, 123, 319, 480, 117, 2500, 603, 1900, 2111, 202, 1300, 5500, 900]:
            if i >= audio.size:
                break
            pieces.append(stream.push(audio[i:i + length]))
            i += length
        pieces.append(stream.push(audio[i:]))
        pieces.append(stream.flush())
        output = np.concatenate(pieces)
        np.testing.assert_allclose(output, audio[:16000], atol=2e-7)
        self.assertEqual(stream.stats.input_frames, 100)
        self.assertEqual(stream.stats.emitted_frames, 100)
        self.assertEqual(stream.stats.chunks, 5)
        self.assertEqual(stream.flush().size, 0)
        self.assertGreater(stream.stats.process_seconds, 0)
        with self.assertRaises(RuntimeError):
            stream.push(np.zeros(FRAME, dtype=np.float32))

    def test_waits_for_lookahead(self) -> None:
        stream = self.make_stream()
        # Hop 20 frames + right context 6 frames + fade 4 frames.
        self.assertEqual(stream.push(np.zeros(FRAME * 29, dtype=np.float32)).size, 0)
        output = stream.push(np.ones(FRAME, dtype=np.float32))
        self.assertEqual(output.size, FRAME * 20)
        self.assertEqual(stream.stats.emitted_frames, 20)

    def test_only_recent_context_is_recomputed(self) -> None:
        stream = self.make_stream()
        for _ in range(200):
            stream.push(np.zeros(FRAME, dtype=np.float32))
        stream.flush()
        self.assertEqual(stream.stats.chunks, 10)
        self.assertTrue(self.fake.segments)
        self.assertLessEqual(max(self.fake.segments), FRAME * (20 + 6 * 2 + 4))
        self.assertLessEqual(stream._buffer.size, FRAME * 6)

    def test_short_final_tail_and_reset(self) -> None:
        self.cfg = ChunkConfig(hop_seconds=0.20, context_seconds=0, crossfade_seconds=0.05)
        stream = self.make_stream()
        audio = np.linspace(-0.3, 0.3, FRAME * 23, dtype=np.float32)
        output = np.concatenate((stream.push(audio), stream.flush()))
        np.testing.assert_allclose(output, audio, atol=2e-7)
        stream.reset()
        self.assertEqual(stream.stats.chunks, 0)
        np.testing.assert_allclose(
            np.concatenate((stream.push(audio), stream.flush())), audio, atol=2e-7
        )

    def test_rejects_offline_only_preprocessing_and_bad_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "offline denoise"):
            self.make_stream(ConvertOptions(denoise=0.5))
        with self.assertRaisesRegex(ValueError, "offline denoise"):
            self.make_stream(ConvertOptions(dehum=50))
        with self.assertRaisesRegex(ValueError, "hop must"):
            StreamingChunkedConverter(self.fake, ChunkConfig(hop_seconds=0))
        with self.assertRaisesRegex(ValueError, "divisible by 100"):
            self.make_stream(ConvertOptions(output_sr=22050))
        stream = self.make_stream()
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            stream.push(np.zeros((1, 160), dtype=np.float32))

    def test_zero_crossfade_and_empty_stream(self) -> None:
        self.cfg = ChunkConfig(hop_seconds=0.2, context_seconds=0, crossfade_seconds=0)
        stream = self.make_stream()
        self.assertEqual(stream.flush().size, 0)
        stream.reset()
        audio = np.ones(3200, dtype=np.float32)
        out = np.concatenate((stream.push(audio), stream.flush()))
        np.testing.assert_array_equal(out, audio)


if __name__ == "__main__":
    unittest.main()
