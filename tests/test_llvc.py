"""LLVC structural tests use untrained weights; never establish voice quality."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from engine.llvc import assets
from engine.llvc.model import Net
from engine.llvc.streaming import LLVCStream, load_stream


def official_model():
    config = json.loads(Path('engine/llvc/config.json').read_text())
    torch.manual_seed(25)
    return Net(**config['model_params']).eval()


@torch.inference_mode()
def reference_stream(model, audio):
    """KoeAI/LLVC infer.py infer_stream input preparation, chunk_factor=1."""
    length, stride = len(audio), model.L * model.dec_chunk_size
    signal = torch.from_numpy(audio)
    signal = torch.nn.functional.pad(signal, (0, (-length) % stride))
    signal = torch.cat((signal[model.L:], torch.zeros(model.L)))
    buffers = model.init_buffers(1, torch.device('cpu'))
    prenet = model.convnet_pre.init_ctx_buf(1, torch.device('cpu'))
    outputs, previous = [], torch.zeros(2*model.L)
    for body in signal.split(stride):
        packet = torch.cat((previous, body))[None, None]
        out, *new = model(packet, *buffers, prenet, pad=False)
        buffers, prenet = tuple(new[:3]), new[3]
        previous = body[-2*model.L:]
        outputs.append(out.cpu().numpy().reshape(-1))
    return np.concatenate(outputs)[:length]


class TestLLVCStream(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = official_model()

    def test_matches_official_stream_with_arbitrary_packets_and_flush(self):
        rng = np.random.default_rng(4)
        for length in (1, 15, 16, 207, 208, 209, 420, 3200+13):
            with self.subTest(length=length):
                audio = rng.normal(0, .1, length).astype(np.float32)
                expected = reference_stream(self.model, audio)
                stream, pieces, offset = LLVCStream(self.model), [], 0
                for size in (1, 7, 0, 301, 5, 777, 1600, 1000):
                    pieces.append(stream.push(audio[offset:offset+size]))
                    offset += size
                    self.assertLess(stream.pending.size, stream.chunk_samples)
                pieces.append(stream.push(audio[offset:]))
                pieces.append(stream.flush())
                result = np.concatenate(pieces)
                self.assertEqual(result.size, length)
                np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-5)
                self.assertEqual(stream.flush().size, 0)
                self.assertEqual(stream.stats.emitted_frames, length/160)
                with self.assertRaises(RuntimeError):
                    stream.push(audio)

    def test_session_isolation_reset_and_invalid_input(self):
        one, two = LLVCStream(self.model), LLVCStream(self.model)
        chunk = np.linspace(-.2, .2, 320, dtype=np.float32)
        initial = one.push(chunk)
        for _ in range(15):
            one.push(chunk)
        np.testing.assert_allclose(two.push(chunk), initial, atol=1e-7)
        one.reset()
        np.testing.assert_allclose(one.push(chunk), initial, atol=1e-7)
        for bad in (np.array([np.nan]), np.zeros((2,2))):
            before = one.received
            with self.assertRaises(ValueError):
                one.push(bad)
            self.assertEqual(one.received, before)

    def test_bad_model_output_is_fatal_not_retried(self):
        stream = LLVCStream(self.model)
        with patch.object(self.model, 'forward', return_value=(torch.zeros(1,1,3), *stream.buffers, stream.pre_context)):
            with self.assertRaises(ValueError):
                stream.push(np.zeros(320, dtype=np.float32))
        with self.assertRaises(RuntimeError):
            stream.flush()
        with self.assertRaises(RuntimeError):
            stream.push(np.zeros(320, dtype=np.float32))

    def test_wav_verification_cli_preserves_resampled_length(self):
        from engine.llvc.__main__ import main
        import soundfile as sf
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'source.wav', Path(directory)/'output.wav'
            sf.write(source, np.zeros(800, dtype=np.float32), 8000)
            with patch('sys.argv', ['llvc', '--input', str(source), '--output', str(output)]), \
                 patch('engine.llvc.__main__.load_stream', return_value=LLVCStream(self.model)), \
                 redirect_stdout(io.StringIO()) as log:
                main()
            audio, sr = sf.read(output)
            self.assertEqual((len(audio), sr), (1600, 16000))
            self.assertEqual(json.loads(log.getvalue())['output_samples'], 1600)

    def test_loader_requires_checkpoint_and_strictly_loads_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pth'
            with patch.object(assets, 'CHECKPOINT', path):
                with self.assertRaises(ValueError):
                    load_stream('cpu', {})
                torch.save({'model': self.model.state_dict()}, path)
                cache = {}
                previous_threads = torch.get_num_threads()
                stream = load_stream('cpu', cache)
                self.assertEqual(torch.get_num_threads(), 1)
                self.assertEqual(stream.stats.chunks, 0)  # warmup excluded
                self.assertEqual(stream.received, 0)
                self.assertGreater(stream.warmup_seconds, 0)
                self.assertFalse(stream.model.training)
                self.assertEqual(len(cache), 1)
                stream.close()
                self.assertEqual(torch.get_num_threads(), previous_threads)
                torch.save({'model': {}}, path)
                with self.assertRaises(RuntimeError):
                    load_stream('cpu', {})


class TestLLVCDownload(unittest.TestCase):
    def test_sha256_download_and_failed_download_preserves_previous_file(self):
        blob = b'fixture-only-not-a-neural-checkpoint'
        metadata = {'sha': 'a'*40, 'siblings': [{'rfilename': assets.MODEL_FILE,
                    'lfs': {'sha256': hashlib.sha256(blob).hexdigest(), 'size': len(blob)}}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'G_500000.pth'
            with patch.object(assets, 'MODEL_DIR', Path(directory)), patch.object(assets, 'CHECKPOINT', path):
                with patch('engine.llvc.assets.urlopen', side_effect=[io.BytesIO(json.dumps(metadata).encode()), io.BytesIO(blob)]):
                    result = assets.install_checkpoint()
                self.assertEqual(path.read_bytes(), blob)
                self.assertEqual(result['revision'], 'a'*40)
                with patch('engine.llvc.assets.urlopen', side_effect=[io.BytesIO(json.dumps(metadata).encode()), io.BytesIO(b'x'*len(blob))]):
                    with self.assertRaises(ValueError):
                        assets.install_checkpoint()
                self.assertEqual(path.read_bytes(), blob)
                self.assertFalse(list(Path(directory).glob('*.part')))

    def test_unverified_metadata_rejected_before_download(self):
        for metadata in ({'sha': 'bad'}, {'sha': 'a'*40, 'siblings': []}):
            with patch('engine.llvc.assets.urlopen', return_value=io.BytesIO(json.dumps(metadata).encode())) as fetch:
                with self.assertRaises(ValueError):
                    assets.install_checkpoint()
                self.assertEqual(fetch.call_count, 1)


@unittest.skipUnless(assets.CHECKPOINT.is_file(), 'official LLVC checkpoint not available')
class TestLLVCOfficialWeights(unittest.TestCase):
    def test_continuous_real_inference(self):
        stream = load_stream('cpu', {})
        samples = np.random.default_rng(2).normal(0, .05, 16000).astype(np.float32)
        chunks = [stream.push(x) for x in np.array_split(samples, 50)]
        chunks.append(stream.flush())
        output = np.concatenate(chunks)
        self.assertEqual(output.size, samples.size)
        self.assertTrue(np.isfinite(output).all())
        self.assertFalse(np.allclose(output, samples))
