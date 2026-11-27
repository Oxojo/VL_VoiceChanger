"""Real LLVC architecture in a test-only untrained fixture; not audio-quality validation."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import URLError

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import torch

from engine.llvc.model import Net
from engine.llvc.streaming import LLVCStream
from engine.web.llvc import register_llvc_routes
from engine.web.stream import register_stream_routes


class TestLLVCWeb(unittest.TestCase):
    def setUp(self):
        self.manager = SimpleNamespace(_llvc_models={}, _executor=ThreadPoolExecutor(max_workers=1),
                                       reserve_stream=lambda: None, release_stream=lambda: None)
        self.app = FastAPI()
        register_stream_routes(self.app, self.manager)
        register_llvc_routes(self.app, self.manager)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.manager._executor.shutdown(wait=True)

    def test_llvc_binary_transport_debug_and_exact_flush(self):
        config = json.loads(Path('engine/llvc/config.json').read_text())
        model = Net(**config['model_params']).eval()
        stream = LLVCStream(model)
        released = threading.Event()
        with patch('engine.llvc.streaming.load_stream', return_value=stream), \
             patch.object(self.manager, 'release_stream', side_effect=released.set) as release:
            with self.client.websocket_connect('/api/stream') as ws:
                ws.send_json({'type': 'start', 'backend': 'llvc', 'model': 'llvc-default', 'device': 'cpu', 'debug': True})
                ready = ws.receive_json()
                self.assertEqual(ready['type'], 'ready')
                self.assertEqual(ready['backend'], 'llvc')
                self.assertEqual(ready['output_sr'], 16000)
                self.assertEqual(ready['playback_lead_ms'], 15)
                self.assertEqual(ready['feature_providers'], {})
                output = []
                for _ in range(8):
                    ws.send_bytes(np.zeros(320, dtype='<f4').tobytes())
                    output.append(np.frombuffer(ws.receive_bytes(), dtype='<f4'))
                    ack = ws.receive_json()
                    self.assertEqual(ack['type'], 'ack')
                    self.assertGreater(ack['debug']['output_frames'], 0)
                ws.send_json({'type': 'stop'})
                output.append(np.frombuffer(ws.receive_bytes(), dtype='<f4'))
                self.assertEqual(ws.receive_json()['type'], 'done')
                self.assertEqual(sum(len(x) for x in output), 320*8)
            self.assertTrue(released.wait(2))
            release.assert_called_once()

    def test_rvc_only_options_are_rejected_for_llvc(self):
        with patch('engine.llvc.streaming.load_stream') as load:
            with self.client.websocket_connect('/api/stream') as ws:
                ws.send_json({'type': 'start', 'backend': 'llvc', 'model': 'llvc-default',
                              'device': 'cpu', 'feature_cache': True})
                self.assertEqual(ws.receive_json()['type'], 'error')
            load.assert_not_called()

    def test_disconnect_closes_llvc_state(self):
        config = json.loads(Path('engine/llvc/config.json').read_text())
        stream = LLVCStream(Net(**config['model_params']).eval())
        released = threading.Event()
        with patch('engine.llvc.streaming.load_stream', return_value=stream), \
             patch.object(self.manager, 'release_stream', side_effect=released.set) as release:
            with self.client.websocket_connect('/api/stream') as ws:
                ws.send_json({'type': 'start', 'backend': 'llvc', 'model': 'llvc-default', 'device': 'cpu'})
                self.assertEqual(ws.receive_json()['type'], 'ready')
            self.assertTrue(released.wait(2), 'disconnect cleanup never completed')
            self.assertTrue(stream.closed)
            self.assertIsNone(stream.buffers)
            release.assert_called_once()

    def test_disconnect_during_loading_waits_for_worker_then_closes(self):
        config = json.loads(Path('engine/llvc/config.json').read_text())
        stream = LLVCStream(Net(**config['model_params']).eval())
        entered, continue_loading, released = threading.Event(), threading.Event(), threading.Event()
        def slow_load(*args, **kwargs):
            entered.set()
            if not continue_loading.wait(5):
                raise RuntimeError('test loader was never released')
            return stream
        with patch('engine.llvc.streaming.load_stream', side_effect=slow_load), \
             patch.object(self.manager, 'release_stream', side_effect=released.set):
            try:
                with self.client.websocket_connect('/api/stream') as ws:
                    ws.send_json({'type': 'start', 'backend': 'llvc', 'model': 'llvc-default', 'device': 'cpu'})
                    self.assertTrue(entered.wait(2))
                self.assertFalse(released.is_set(), 'reservation released while loader still running')
            finally:
                continue_loading.set()
            self.assertTrue(released.wait(2), 'loading cleanup never completed')
            self.assertTrue(stream.closed)
            self.assertIsNone(stream.buffers)

    def test_missing_checkpoint_reports_error_and_releases(self):
        released = threading.Event()
        with patch('engine.llvc.streaming.load_stream', side_effect=ValueError('LLVCモデルがありません')), \
             patch.object(self.manager, 'release_stream', side_effect=released.set) as release:
            with self.client.websocket_connect('/api/stream') as ws:
                ws.send_json({'type': 'start', 'backend': 'llvc', 'model': 'llvc-default', 'device': 'cpu'})
                error = ws.receive_json()
                self.assertEqual(error['type'], 'error')
                self.assertIn('LLVCモデルがありません', error['message'])
            self.assertTrue(released.wait(2))
            release.assert_called_once()

    def test_download_errors_release_and_cross_origin_rejected(self):
        with patch('engine.web.llvc.install_checkpoint', side_effect=URLError('403')), \
             patch.object(self.manager, 'release_stream') as release:
            response = self.client.post('/api/llvc/install')
            self.assertEqual(response.status_code, 502)
            self.assertIn('Hugging Face', response.json()['detail'])
            release.assert_called_once()
        with patch('engine.web.llvc.install_checkpoint') as install:
            response = self.client.post('/api/llvc/install', headers={'origin': 'https://other.example'})
            self.assertEqual(response.status_code, 403)
            install.assert_not_called()

    def test_successful_download_reports_status(self):
        with patch('engine.web.llvc.install_checkpoint', return_value={'sha256': 'verified'}), \
             patch('engine.web.llvc.model_status', return_value={'installed': True}), \
             patch.object(self.manager, 'release_stream') as release:
            response = self.client.post('/api/llvc/install')
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()['installed'])
            release.assert_called_once()
