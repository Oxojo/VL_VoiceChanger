from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cache_fakes import make_converter
from engine.convert import ConvertOptions
from engine.features.cache import FeatureBatch, StreamingFeatureCache
from engine.realtime import ChunkConfig
from engine.streaming import StreamingChunkedConverter


class TestFeatureCache(unittest.TestCase):
    def setUp(self):
        self.cache = StreamingFeatureCache()
        self.calls = []

    def extract(self, audio, frames):
        self.calls.append((int(audio[0]), frames))
        f0 = audio[::160][:frames].copy()
        pairs = np.repeat(audio[::320], 2)[:frames]
        return FeatureBatch(torch.from_numpy(pairs.copy())[None, :, None], f0,
                            {"f0": 0.002, "contentvec": 0.003})

    def window(self, start, real=85):
        frames = real + 1
        audio = np.repeat(np.arange(start, start + frames, dtype=np.float32), 160)
        return self.cache.extract(audio, frames, start_frame=start, real_frames=real, extract=self.extract)

    def test_guarded_refresh_saves_extraction_without_frame_or_phase_shift(self):
        self.window(0)
        result = self.window(50)
        self.assertEqual(self.calls, [(0, 86), (64, 72)])
        np.testing.assert_array_equal(result.f0, np.arange(50, 136))
        np.testing.assert_array_equal(result.phone.numpy()[0, :, 0], np.repeat(np.arange(50, 136, 2), 2))
        summary = self.cache.summary()
        self.assertEqual(summary["avoided_frames"], 14)
        self.assertEqual(summary["reused_frames"], 25)
        self.assertEqual(summary["retained_frames"], 75)

    def test_odd_10ms_phase_shift_recomputes_full_window(self):
        self.window(0)
        result = self.window(51)
        self.assertEqual(self.calls[-1], (51, 86))
        self.assertEqual(self.cache.stats.reused_frames, 0)
        self.assertEqual(self.cache.stats.alignment_misses, 1)
        np.testing.assert_array_equal(result.phone.numpy()[0, :, 0], np.repeat(np.arange(51, 137, 2), 2))

    def test_no_savings_uses_fresh_features_and_short_tail_is_valid(self):
        self.window(0, real=20)
        self.window(10, real=20)
        self.assertEqual(self.calls, [(0, 21), (10, 21)])
        result = self.window(20, real=3)
        self.assertEqual(result.f0.size, 4)
        self.assertEqual(self.cache.f0.size, 0)

    def test_memory_is_bounded_and_cached_arrays_are_owned(self):
        first = self.window(0)
        first.f0[:] = -1
        first.phone[:] = -1
        next_window = self.window(50)
        self.assertTrue((next_window.f0 >= 0).all())
        self.assertTrue((next_window.phone >= 0).all())
        for start in range(100, 10000, 50):
            self.window(start)
            self.assertLessEqual(self.cache.f0.size, 75)
            self.assertLessEqual(self.cache.phone.shape[1], 75)

    def test_reset_and_snapshot_restore_include_counters(self):
        self.window(0)
        before = self.cache.checkpoint()
        counters = asdict(self.cache.stats)
        self.window(50)
        self.cache.restore(before)
        self.assertEqual(asdict(self.cache.stats), counters)
        self.assertEqual(self.cache.start, 0)
        self.cache.reset()
        self.assertEqual(self.cache.stats.calls, 0)
        self.assertEqual(self.cache.f0.size, 0)

    def test_invalid_feature_lengths_are_not_cached(self):
        with self.assertRaisesRegex(ValueError, "frame counts"):
            self.cache.extract(np.zeros(1600), 10, start_frame=0, real_frames=9,
                               extract=lambda *args: FeatureBatch(torch.zeros(1, 3, 2), np.zeros(4)))
        self.assertEqual(self.cache.stats.calls, 0)


class TestCachedConversion(unittest.TestCase):
    def test_extraction_alignment_matches_original_repeat_and_edge_padding(self):
        converter = make_converter()
        f0 = np.array([0, 110, 220, 0, 330], dtype=np.float32)
        phone = np.arange(24, dtype=np.float32).reshape(1, 6, 4)
        converter.rmvpe.infer.return_value = f0
        converter.rmvpe.infer.side_effect = None
        converter.contentvec.extract.return_value = phone
        converter.contentvec.extract.side_effect = None
        stages = []
        batch = converter.extract_features(np.zeros(160 * 13), 13,
                                            on_stage=lambda name, progress: stages.append(name))
        np.testing.assert_array_equal(batch.f0, np.pad(f0, (0, 8), mode="edge"))
        expected = np.repeat(phone, 2, axis=1)
        expected = np.concatenate((expected, expected[:, -1:]), axis=1)
        np.testing.assert_array_equal(batch.phone.numpy(), expected)
        self.assertEqual(stages, ["f0", "contentvec"])

    def test_refactored_full_window_matches_explicit_features_without_reuse(self):
        converter = make_converter()
        audio = np.zeros(160 * 85, dtype=np.float32)
        options = ConvertOptions(pitch_shift=12)
        baseline, report = converter.convert_array(audio, options)
        cached, cached_report = converter.convert_array(audio, options,
                                                       feature_cache=StreamingFeatureCache(), source_frame=0)
        np.testing.assert_array_equal(cached, baseline)
        self.assertEqual(report.frames, cached_report.frames)
        self.assertEqual(report.f0_voiced_ratio, cached_report.f0_voiced_ratio)

    def test_stream_reuses_raw_f0_without_double_pitch_shift_and_reset_is_clean(self):
        converter = make_converter()
        stream = StreamingChunkedConverter(converter,
            ChunkConfig(hop_seconds=0.3, context_seconds=0.15, crossfade_seconds=0.05),
            ConvertOptions(pitch_shift=12), feature_cache=True)
        audio = np.zeros(160 * 240 + 13, dtype=np.float32)
        pieces = [stream.push(audio[i:i + 320]) for i in range(0, audio.size, 320)]
        pieces.append(stream.flush())
        self.assertEqual(np.concatenate(pieces).size, 240 * 480)
        self.assertEqual(stream.stats.emitted_frames, 240)
        self.assertGreater(stream.feature_cache.stats.reused_frames, 0)
        self.assertGreater(stream.feature_cache.summary()["avoided_frames"], 0)
        for call in converter.model.infer.call_args_list:
            pitchf = call.args[3].numpy()
            self.assertTrue(np.isin(pitchf, [0, 220]).all())
        stream.reset()
        self.assertEqual(stream.feature_cache.stats.calls, 0)

    def test_failed_second_chunk_rolls_back_feature_state_and_statistics(self):
        converter = make_converter()
        stream = StreamingChunkedConverter(converter, feature_cache=True)
        original = converter.convert_array
        calls = 0

        def bad(*args, **kwargs):
            nonlocal calls
            calls += 1
            out, report = original(*args, **kwargs)
            if calls == 2:
                out = out[:-480 * 5]
            return out, report

        with patch.object(converter, "convert_array", side_effect=bad):
            with self.assertRaisesRegex(ValueError, "more than one"):
                stream.push(np.zeros(160 * 140, dtype=np.float32))
            self.assertEqual(stream.feature_cache.stats.calls, 0)
            self.assertEqual(stream.feature_cache.f0.size, 0)
            self.assertEqual(stream.stats.chunks, 0)
            first = stream.push(np.empty(0))
            end = stream.flush()
            self.assertEqual(first.size + end.size, 140 * 480)

    def test_cache_is_session_local_and_rejects_unsupported_backend(self):
        converter = make_converter()
        a = StreamingChunkedConverter(converter, feature_cache=True)
        b = StreamingChunkedConverter(converter, feature_cache=True)
        a.push(np.zeros(160 * 70, dtype=np.float32))
        self.assertEqual(b.feature_cache.stats.calls, 0)
        with self.assertRaisesRegex(ValueError, "does not support"):
            StreamingChunkedConverter(SimpleNamespace(info=converter.info), feature_cache=True)

    def test_nonstandard_padding_is_rejected_only_for_cached_path(self):
        converter = make_converter()
        converter.convert_array(np.zeros(1600), ConvertOptions(pad_samples=0))
        with self.assertRaisesRegex(ValueError, "default padding"):
            converter.convert_array(np.zeros(1600), ConvertOptions(pad_samples=0), feature_cache=StreamingFeatureCache())

    def test_stream_guard_never_reduces_configured_context(self):
        converter = make_converter()
        stream = StreamingChunkedConverter(converter,
            ChunkConfig(hop_seconds=0.3, context_seconds=0.15, crossfade_seconds=0.05), feature_cache=True)
        self.assertEqual(stream.feature_cache.guard_frames, 15)
        stream.push(np.zeros(160 * 110))
        self.assertGreater(stream.feature_cache.summary()["avoided_frames"], 0)

    def test_rollback_preserves_existing_cached_prefix_after_committed_audio(self):
        converter = make_converter()
        stream = StreamingChunkedConverter(converter, feature_cache=True)
        stream.push(np.zeros(160 * 70))
        checkpoint = stream.feature_cache.checkpoint()
        original = converter.convert_array

        def invalid(*args, **kwargs):
            out, report = original(*args, **kwargs)
            return out[:-480 * 5], report

        with patch.object(converter, "convert_array", side_effect=invalid):
            with self.assertRaises(ValueError):
                stream.push(np.zeros(160 * 50))
        self.assertEqual(stream.feature_cache.start, checkpoint[0])
        np.testing.assert_array_equal(stream.feature_cache.f0, checkpoint[2])
        torch.testing.assert_close(stream.feature_cache.phone, checkpoint[3])
        self.assertEqual(stream.feature_cache.stats, checkpoint[4])
        self.assertEqual(stream.stats.emitted_frames, 50)
