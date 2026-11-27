"""FP32 graph/state parity with test-only untrained weights, not voice quality."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from engine.llvc.cpu import prepare_cpu_runtime
from engine.llvc.model import Net
from engine.llvc.streaming import LLVCStream


class TestLLVCCPURuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cls.model = Net(**json.loads(Path('engine/llvc/config.json').read_text())['model_params']).eval()
        cls.directory = tempfile.TemporaryDirectory(prefix='llvc-cpu-test-')
        cls.path = Path(cls.directory.name)/'fixture.pth'
        torch.save({'model': cls.model.state_dict()}, cls.path)
        cls.runtime = prepare_cpu_runtime(cls.model, cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.runtime = None
        cls.directory.cleanup()
        torch.set_num_threads(cls.previous_threads)

    @torch.inference_mode()
    def test_encoder_matches_old_inplace_state_update_exactly(self):
        encoder = self.model.mask_gen.encoder
        x = torch.randn(1, encoder.channels, 13)
        previous = torch.randn_like(encoder.init_ctx_buf(1, torch.device('cpu')))
        actual, actual_state = encoder(x.clone(), previous.clone())
        expected, state = x.clone(), previous.clone()
        for i in range(encoder.num_layers):
            start = encoder.buf_indices[i]
            end = start + encoder.buf_lengths[i]
            joined = torch.cat((state[..., start:end], expected), dim=-1)
            state[..., start:end] = joined[..., -encoder.buf_lengths[i]:]
            expected = expected + encoder.dcc_layers[i](joined)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual_state, state, atol=0, rtol=0)

    def test_recurrent_output_reset_and_flush_match_torch(self):
        reference = LLVCStream(self.model)
        accelerated = LLVCStream(self.model, self.runtime)
        rng = np.random.default_rng(8)
        source = rng.normal(0, .08, 16000 + 17).astype(np.float32)
        offset, outputs, expected = 0, [], []
        sizes = [1, 0, 7, 319, 1600, 31, 640, 320, 100]
        while offset < source.size:
            for size in sizes:
                packet = source[offset:offset+size]
                outputs.append(accelerated.push(packet))
                expected.append(reference.push(packet))
                offset += size
        outputs.append(accelerated.flush())
        expected.append(reference.flush())
        result, eager = np.concatenate(outputs), np.concatenate(expected)
        self.assertEqual(result.size, source.size)
        np.testing.assert_allclose(result, eager, atol=3e-5, rtol=3e-4)
        self.assertEqual(accelerated.runtime_name, 'ONNX Runtime CPU / FP32')
        accelerated.reset()
        reference.reset()
        np.testing.assert_allclose(accelerated.push(source[:320]), reference.push(source[:320]), atol=3e-5, rtol=3e-4)

    def test_cached_graph_is_revalidated_without_reexport(self):
        with patch('torch.onnx.export', side_effect=AssertionError('cache unexpectedly reexported')):
            runtime = prepare_cpu_runtime(self.model, self.path)
        self.assertLess(runtime.validation_max_error, 1e-4)

    def test_corrupt_cache_is_rebuilt(self):
        path = next((self.path.parent/'onnx').glob('*.onnx'))
        path.write_bytes(b'broken cache')
        runtime = prepare_cpu_runtime(self.model, self.path)
        self.assertLess(runtime.validation_max_error, 1e-4)
        self.assertGreater(path.stat().st_size, 1000)
        self.assertFalse(list(path.parent.glob('tmp*')))

    @torch.inference_mode()
    def test_bad_runtime_output_is_not_accepted(self):
        with patch.object(self.runtime.session, 'run', side_effect=lambda names, values: [
                np.zeros((1,1,208), dtype=np.float32), values['enc'], values['dec'], values['out'], values['pre']]):
            with self.assertRaises(AssertionError):
                self.runtime.validate(self.model)
