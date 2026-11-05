"""API reservation/cleanup and truthful presentation of the untrained probe."""
import subprocess
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from engine.web import server


class TestStreamVCWeb(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)

    def test_entry_and_experimental_page(self):
        self.assertIn('/streamvc', self.client.get('/').text)
        page = self.client.get('/streamvc')
        self.assertEqual(page.status_code, 200)
        self.assertIn('未学習モデル', page.text)
        self.assertIn('声質変換・マイク入力・音声の試聴はできません', page.text)

    def test_success_reserves_and_releases(self):
        with patch.object(server.manager, 'reserve_stream') as reserve, \
             patch.object(server.manager, 'release_stream') as release, \
             patch('engine.web.streamvc.run_probe', return_value={'trained': False}) as run:
            response = self.client.post('/api/streamvc/probe', json={'device': 'cpu', 'threads': 1})
            self.assertEqual(response.json(), {'trained': False})
            self.assertEqual(response.status_code, 200)
            reserve.assert_called_once()
            release.assert_called_once()
            run.assert_called_once()

    def test_busy_does_not_run_or_release_someone_elses_reservation(self):
        with patch.object(server.manager, 'reserve_stream', side_effect=ValueError('busy')), \
             patch.object(server.manager, 'release_stream') as release, \
             patch('engine.web.streamvc.run_probe') as run:
            self.assertEqual(self.client.post('/api/streamvc/probe', json={}).status_code, 409)
            release.assert_not_called()
            run.assert_not_called()

    def test_errors_always_release(self):
        for failure, code in [(subprocess.TimeoutExpired('probe', 120), 504),
                              (subprocess.CalledProcessError(1, 'probe'), 500)]:
            with self.subTest(failure=failure), \
                 patch.object(server.manager, 'reserve_stream'), \
                 patch.object(server.manager, 'release_stream') as release, \
                 patch('engine.web.streamvc.run_probe', side_effect=failure):
                self.assertEqual(self.client.post('/api/streamvc/probe', json={}).status_code, code)
                release.assert_called_once()

    def test_unavailable_cuda_and_invalid_options(self):
        with patch('engine.web.streamvc.torch.cuda.is_available', return_value=False), \
             patch('engine.web.streamvc.run_probe') as run:
            self.assertEqual(self.client.post('/api/streamvc/probe', json={'device': 'cuda'}).status_code, 400)
            for invalid in [{'device': 'shell'}, {'threads': 999}, {'frames': 1000000}]:
                self.assertEqual(self.client.post('/api/streamvc/probe', json=invalid).status_code, 422)
            run.assert_not_called()
