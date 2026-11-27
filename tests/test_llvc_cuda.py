"""CUDA Graph tests require real CUDA. CPU cases test preflight and selection."""
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from engine.llvc.cuda import CUDAGraphRuntime
from engine.llvc.model import Net
from engine.llvc.streaming import LLVCStream, check_realtime
from engine.web.stream import StreamRequest


def make_model():
    torch.manual_seed(32)
    return Net(**json.loads(Path('engine/llvc/config.json').read_text())['model_params']).eval()


class TestLLVCPreflight(unittest.TestCase):
    def test_llvc_auto_uses_cpu_even_when_cuda_available(self):
        manager = SimpleNamespace(_llvc_models={})
        request = StreamRequest(backend='llvc', model='llvc-default', device='auto')
        with patch('engine.web.stream.torch.cuda.is_available', return_value=True), \
             patch('engine.llvc.streaming.load_stream', return_value='sentinel') as load:
            self.assertEqual(request.build(manager), 'sentinel')
            load.assert_called_once_with('cpu', manager._llvc_models, warmup=True, model_id='llvc-default')

    def test_capacity_measurement_resets_audio_state_and_excludes_statistics(self):
        stream = LLVCStream(make_model())
        stream.push(np.zeros(320, dtype=np.float32))
        ticks = itertools.count()
        with patch('engine.llvc.streaming.time.perf_counter', side_effect=lambda: next(ticks)*.001):
            report = check_realtime(stream, packets=2)
        self.assertLess(report['wall_rtf'], 1)
        self.assertEqual(report['input_ms'], 40)
        self.assertEqual(stream.received, 0)
        self.assertEqual(stream.stats.chunks, 0)
        self.assertEqual(stream.pending.size, 0)

    def test_unsustainable_runtime_rejected_with_actionable_error_and_clean_state(self):
        stream = LLVCStream(make_model())
        ticks = itertools.count()
        with patch('engine.llvc.streaming.time.perf_counter', side_effect=lambda: next(ticks)*.1):
            with self.assertRaisesRegex(ValueError, 'マイクは開始していません.*CPU'):
                check_realtime(stream, packets=2)
        self.assertEqual(stream.received, 0)
        self.assertEqual(stream.stats.chunks, 0)
        self.assertEqual(stream.pending.size, 0)

    def test_graph_rejects_cpu_model(self):
        with self.assertRaisesRegex(ValueError, 'CUDA上のモデル'):
            CUDAGraphRuntime(make_model())


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA device unavailable: graph capture/replay unverified')
class TestLLVCCUDAGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = make_model().to('cuda')

    def test_replay_recurrent_state_audio_flush_and_reset_match_eager(self):
        runtime = CUDAGraphRuntime(self.model)
        graph, eager = LLVCStream(self.model, runtime), LLVCStream(self.model)
        source = np.random.default_rng(8).normal(0, .1, 16000+17).astype(np.float32)
        actual, expected = [], []
        for i in range(0, source.size, 319):
            actual.append(graph.push(source[i:i+319]))
            expected.append(eager.push(source[i:i+319]))
        actual.append(graph.flush())
        expected.append(eager.flush())
        np.testing.assert_allclose(np.concatenate(actual), np.concatenate(expected), rtol=1e-3, atol=1e-4)
        self.assertEqual(sum(len(x) for x in actual), source.size)
        graph.reset()
        eager.reset()
        np.testing.assert_allclose(graph.push(source[:320]), eager.push(source[:320]), rtol=1e-3, atol=1e-4)

    def test_distinct_sessions_do_not_share_graph_buffers(self):
        first = LLVCStream(self.model, CUDAGraphRuntime(self.model))
        second = LLVCStream(self.model, CUDAGraphRuntime(self.model))
        packet = np.linspace(-.1, .1, 320, dtype=np.float32)
        initial = first.push(packet)
        for _ in range(10):
            first.push(packet)
        np.testing.assert_allclose(second.push(packet), initial, rtol=1e-3, atol=1e-4)
