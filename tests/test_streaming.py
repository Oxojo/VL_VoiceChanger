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


class DifferingChunkConverter:
    """Each inference produces a distinct constant, revealing bad fades."""

    def __init__(self) -> None:
        self.info = SimpleNamespace(sample_rate=16000)
        self.calls: list[int] = []

    def convert_array(self, samples: np.ndarray, options):
        self.calls.append(int(samples.size))
        return np.full(samples.size, float(len(self.calls)), dtype=np.float32), None


class IdentityUpsampleConverter:
    """Mock a 16 kHz -> 48 kHz synthesis output without a neural model."""

    def __init__(self) -> None:
        self.info = SimpleNamespace(sample_rate=48000)

    def convert_array(self, samples: np.ndarray, options):
        return np.repeat(samples, 3).astype(np.float32), None


class BadLengthConverter:
    """A broken backend returning a waveform 50 ms too short."""

    def __init__(self) -> None:
        self.info = SimpleNamespace(sample_rate=16000)

    def convert_array(self, samples: np.ndarray, options):
        return np.zeros(max(0, samples.size - FRAME * 5), dtype=np.float32), None


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

    @staticmethod
    def _feed(stream: StreamingChunkedConverter, audio: np.ndarray, sizes: list[int]) -> np.ndarray:
        parts: list[np.ndarray] = []
        index = 0
        offset = 0
        while offset < audio.size:
            n = sizes[index % len(sizes)]
            parts.append(stream.push(audio[offset : offset + n]))
            offset += n
            index += 1
        parts.append(stream.flush())
        return np.concatenate(parts)

    def test_crossfade_with_different_chunk_outputs(self) -> None:
        """A/B and B/C boundaries interpolate once with exact endpoints."""
        fake = DifferingChunkConverter()
        stream = StreamingChunkedConverter(fake, self.cfg)
        output = self._feed(stream, np.zeros(FRAME * 60, dtype=np.float32), [1, 7, FRAME, 320, 1234])
        self.assertEqual(stream.stats.chunks, 3)
        self.assertEqual(output.size, FRAME * 60)
        self.assertEqual(len(fake.calls), 3)
        fade = np.linspace(0, 1, FRAME * 4, dtype=np.float32)
        np.testing.assert_allclose(output[:FRAME * 20], 1)
        np.testing.assert_allclose(output[FRAME * 20 : FRAME * 24], 1 + fade, atol=1e-6)
        np.testing.assert_allclose(output[FRAME * 24 : FRAME * 40], 2)
        np.testing.assert_allclose(output[FRAME * 40 : FRAME * 44], 2 + fade, atol=1e-6)
        np.testing.assert_allclose(output[FRAME * 44 :], 3)

    def test_matches_offline_chunk_assembly(self) -> None:
        """Compare to ChunkedConverter, including overlap and short tails."""
        from engine.realtime import ChunkedConverter

        for frames in (1, 5, 19, 20, 21, 23, 39, 40, 41, 44, 60, 61, 100):
            for fade in (0, 0.02, 0.04, 0.23):
                with self.subTest(frames=frames, fade=fade):
                    cfg = ChunkConfig(hop_seconds=0.2, context_seconds=0.06, crossfade_seconds=fade)
                    audio = np.zeros(FRAME * frames, dtype=np.float32)
                    # Mock accepts sub-100ms clips even though real RVC does not.
                    reference, _ = ChunkedConverter(DifferingChunkConverter(), cfg).process(audio)
                    stream = StreamingChunkedConverter(DifferingChunkConverter(), cfg)
                    output = self._feed(stream, audio, [1, 7, FRAME, 320, 911])
                    np.testing.assert_allclose(output, reference, atol=2e-6)
                    self.assertEqual(stream.stats.emitted_frames, frames)

    def test_output_is_invariant_under_input_block_partitioning(self) -> None:
        audio = np.linspace(-0.5, 0.5, FRAME * 61 + 103, dtype=np.float32)
        baseline = self._feed(self.make_stream(), audio, [audio.size])
        for sizes in ([1], [7], [160], [320], [17, 317, 199, 5]):
            with self.subTest(sizes=sizes):
                output = self._feed(self.make_stream(), audio, sizes)
                np.testing.assert_array_equal(output, baseline)
                self.assertEqual(output.size, FRAME * 61)

    def test_48khz_output_preserves_sample_timing(self) -> None:
        audio = np.sin(np.linspace(0, 10, FRAME * 61, dtype=np.float32))
        stream = StreamingChunkedConverter(IdentityUpsampleConverter(), self.cfg)
        output = self._feed(stream, audio, [1, 7, 160, 320])
        self.assertEqual(output.size, 61 * 480)
        np.testing.assert_allclose(output, np.repeat(audio, 3), atol=2e-7)

    def test_final_frame_shorter_than_model_minimum_is_padded_only_for_inference(self) -> None:
        self.cfg = ChunkConfig(hop_seconds=0.2, context_seconds=0, crossfade_seconds=0)
        stream = self.make_stream()
        audio = np.arange(FRAME, dtype=np.float32) / FRAME
        output = self._feed(stream, audio, [1, 7, 32])
        np.testing.assert_array_equal(output, audio)
        self.assertEqual(self.fake.segments, [FRAME * 10])
        self.assertEqual(stream.stats.emitted_frames, 1)

    def test_flush_discards_only_an_incomplete_10ms_frame(self) -> None:
        stream = self.make_stream()
        self.assertEqual(self._feed(stream, np.ones(159, dtype=np.float32), [1, 7]).size, 0)
        self.assertEqual(stream.stats.input_frames, 0)
        self.assertEqual(stream.stats.emitted_frames, 0)
        stream.reset()
        audio = np.arange(FRAME * 4 + 159, dtype=np.float32)
        np.testing.assert_array_equal(self._feed(stream, audio, [1, 7]), audio[:FRAME * 4])

    def test_backend_length_failure_does_not_commit_frames(self) -> None:
        stream = StreamingChunkedConverter(BadLengthConverter(), self.cfg)
        with self.assertRaisesRegex(ValueError, "more than one 10 ms output frame"):
            stream.push(np.ones(FRAME * 30, dtype=np.float32))
        self.assertEqual(stream.stats.chunks, 0)
        self.assertEqual(stream.stats.emitted_frames, 0)
        self.assertEqual(stream._next_frame, 0)

    def test_later_push_failure_preserves_unreturned_audio(self) -> None:
        class FailOnceConverter(IdentityConverter):
            def convert_array(self, samples, options):
                out, report = super().convert_array(samples, options)
                if len(self.segments) == 3:
                    out = out[:-FRAME * 5]
                return out, report

        stream = StreamingChunkedConverter(FailOnceConverter(), self.cfg)
        audio = np.linspace(-0.5, 0.5, FRAME * 70, dtype=np.float32)
        first = stream.push(audio[:FRAME * 30])
        before_buffer = stream._buffer.copy()
        before_fade = stream._pending_fade.copy()
        before_time = stream.stats.process_seconds
        with self.assertRaisesRegex(ValueError, "more than one 10 ms output frame"):
            stream.push(audio[FRAME * 30:])
        self.assertEqual(stream.stats.input_frames, 70)
        self.assertEqual(stream.stats.chunks, 1)
        self.assertEqual(stream.stats.emitted_frames, 20)
        self.assertEqual(stream._next_frame, 20)
        self.assertEqual(stream._buffer_start, 14)
        self.assertEqual(len(stream.stats.latencies), 1)
        self.assertEqual(stream.stats.process_seconds, before_time)
        np.testing.assert_array_equal(stream._pending_fade, before_fade)
        np.testing.assert_array_equal(
            stream._buffer, np.concatenate((before_buffer, audio[FRAME * 30:])),
        )
        output = np.concatenate((first, stream.push(np.empty(0)), stream.flush()))
        np.testing.assert_allclose(output, audio, atol=2e-7)
        self.assertEqual(stream.stats.chunks, 4)
        self.assertEqual(stream.stats.emitted_frames, 70)

    def test_later_flush_failure_preserves_unreturned_audio(self) -> None:
        class FailOnceConverter(IdentityConverter):
            def convert_array(self, samples, options):
                out, report = super().convert_array(samples, options)
                if len(self.segments) == 2:
                    out = out[:-FRAME * 5]
                return out, report

        stream = StreamingChunkedConverter(FailOnceConverter(), self.cfg)
        audio = np.linspace(-0.5, 0.5, FRAME * 25 + 13, dtype=np.float32)
        self.assertEqual(stream.push(audio).size, 0)
        before_buffer = stream._buffer.copy()
        with self.assertRaisesRegex(ValueError, "more than one 10 ms output frame"):
            stream.flush()
        self.assertEqual(stream.stats.chunks, 0)
        self.assertEqual(stream.stats.emitted_frames, 0)
        self.assertEqual(stream._next_frame, 0)
        self.assertEqual(stream._buffer_start, 0)
        self.assertEqual(stream.stats.latencies, [])
        self.assertEqual(stream.stats.process_seconds, 0)
        self.assertFalse(stream._closed)
        self.assertEqual(stream._unframed.size, 13)
        np.testing.assert_array_equal(stream._buffer, before_buffer)
        np.testing.assert_allclose(stream.flush(), audio[:FRAME * 25], atol=2e-7)
        self.assertEqual(stream.stats.chunks, 2)
        self.assertEqual(stream.stats.emitted_frames, 25)
        self.assertEqual(stream.flush().size, 0)

    def test_backend_length_tolerance_at_output_sample_rate(self) -> None:
        for sr in (16000, 48000):
            spf = sr // 100
            for frames in (1, 30):
                for delta in (-spf - 1, -spf, 0, spf, spf + 1):
                    with self.subTest(sr=sr, frames=frames, delta=delta):
                        class SizedConverter:
                            info = SimpleNamespace(sample_rate=sr)

                            def convert_array(self, samples, options):
                                n = samples.size // FRAME * spf + delta
                                return np.ones(n, dtype=np.float32), None

                        cfg = ChunkConfig(hop_seconds=1, context_seconds=0, crossfade_seconds=0)
                        stream = StreamingChunkedConverter(SizedConverter(), cfg)
                        stream.push(np.ones(FRAME * frames, dtype=np.float32))
                        if abs(delta) > spf:
                            with self.assertRaisesRegex(ValueError, "more than one 10 ms output frame"):
                                stream.flush()
                            self.assertEqual(stream.stats.chunks, 0)
                            self.assertEqual(stream.stats.emitted_frames, 0)
                        else:
                            output = stream.flush()
                            self.assertEqual(output.size, frames * spf)
                            valid = min(output.size, max(frames, 10) * spf + delta)
                            np.testing.assert_array_equal(output[:valid], 1)
                            np.testing.assert_array_equal(output[valid:], 0)
                            self.assertEqual(stream.stats.emitted_frames, frames)

    def test_empty_push_keeps_stream_open(self) -> None:
        stream = self.make_stream()
        self.assertEqual(stream.push(np.zeros(0, dtype=np.float32)).size, 0)
        audio = np.ones(FRAME * 20, dtype=np.float32)
        np.testing.assert_array_equal(
            np.concatenate((stream.push(audio), stream.flush())), audio
        )

    def test_stage_timings_rollback_with_unreturned_audio(self) -> None:
        class ProfiledConverter(IdentityConverter):
            def convert_array(self, samples, options):
                output, _ = super().convert_array(samples, options)
                if len(self.segments) == 3:
                    output = output[:-FRAME * 5]
                return output, SimpleNamespace(timings={"f0": 0.002, "contentvec": 0.003})

        stream = StreamingChunkedConverter(ProfiledConverter(), self.cfg)
        stream.push(np.zeros(FRAME * 30, dtype=np.float32))
        self.assertEqual(stream.stats.stage_seconds, {"f0": 0.002, "contentvec": 0.003})
        with self.assertRaisesRegex(ValueError, "more than one 10 ms output frame"):
            stream.push(np.zeros(FRAME * 40, dtype=np.float32))
        self.assertEqual(stream.stats.stage_seconds, {"f0": 0.002, "contentvec": 0.003})
        stream.push(np.empty(0))
        self.assertEqual(stream.stats.chunks, 3)
        self.assertAlmostEqual(stream.stats.stage_seconds["f0"], 0.006)
        stream.reset()
        self.assertEqual(stream.stats.stage_seconds, {})



if __name__ == "__main__":
    unittest.main()
