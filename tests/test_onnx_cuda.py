"""Provider selection must be explicit, verified and isolated in the cache."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from engine.features.providers import feature_providers, ort_providers, verify_ort_cuda
from engine.web import server


def converter_with(cv, f0=None):
    def feature(providers):
        return SimpleNamespace(session=SimpleNamespace(get_providers=lambda: providers))
    return SimpleNamespace(contentvec=feature(cv), rmvpe=feature(cv if f0 is None else f0))


class TestONNXCUDA(unittest.TestCase):
    def test_cpu_mode_requires_no_cuda_provider(self):
        with patch("engine.features.providers.ort.get_available_providers", return_value=["CPUExecutionProvider"]):
            self.assertIsNone(ort_providers(False))
            verify_ort_cuda(converter_with(["CPUExecutionProvider"]), False)

    def test_missing_cuda_is_an_actionable_error(self):
        with patch("engine.features.providers.ort.get_available_providers", return_value=["CPUExecutionProvider"]):
            with self.assertRaisesRegex(RuntimeError, "onnxruntime-gpu"):
                ort_providers(True)

    def test_cuda_requires_both_sessions_to_register_cuda_first(self):
        with patch("engine.features.providers.ort.get_available_providers", return_value=["CUDAExecutionProvider"]):
            self.assertEqual(ort_providers(True), ["CUDAExecutionProvider", "CPUExecutionProvider"])
        for bad in (["CPUExecutionProvider"], ["CPUExecutionProvider", "CUDAExecutionProvider"], []):
            for failing_name in ("contentvec", "rmvpe"):
                good = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                cv = bad if failing_name == "contentvec" else good
                f0 = bad if failing_name == "rmvpe" else good
                with self.subTest(bad=bad, feature=failing_name):
                    with self.assertRaisesRegex(RuntimeError, failing_name):
                        verify_ort_cuda(converter_with(cv, f0), True)
        verify_ort_cuda(converter_with(["CUDAExecutionProvider", "CPUExecutionProvider"]), True)

    def test_provider_reporting_reflects_sessions(self):
        self.assertEqual(feature_providers(converter_with(["CPUExecutionProvider"])), {
            "contentvec": ["CPUExecutionProvider"], "rmvpe": ["CPUExecutionProvider"],
        })

    def test_cache_separates_cpu_and_cuda_feature_extractors(self):
        manager = server.JobManager()
        cpu = converter_with(["CPUExecutionProvider"])
        gpu = converter_with(["CUDAExecutionProvider", "CPUExecutionProvider"])
        try:
            with patch("engine.web.server.model_path", return_value=Path("test.pth")), \
                 patch("engine.web.server.VoiceConverter", side_effect=[cpu, gpu]) as build, \
                 patch("engine.features.providers.ort.get_available_providers", return_value=["CUDAExecutionProvider"]):
                job = server.Job(id="test", request={})
                self.assertIs(manager._converter("test", "cuda", job), cpu)
                self.assertIs(manager._converter("test", "cuda", job, ort_cuda=True), gpu)
                self.assertIs(manager._converter("test", "cuda", job), cpu)
                self.assertIs(manager._converter("test", "cuda", job, ort_cuda=True), gpu)
                self.assertEqual(build.call_count, 2)
                self.assertIsNone(build.call_args_list[0].kwargs["providers"])
                self.assertEqual(build.call_args_list[1].kwargs["providers"],
                                 ["CUDAExecutionProvider", "CPUExecutionProvider"])
        finally:
            manager._executor.shutdown()

    def test_silent_fallback_is_not_cached(self):
        manager = server.JobManager()
        try:
            with patch("engine.web.server.model_path", return_value=Path("test.pth")), \
                 patch("engine.web.server.VoiceConverter", return_value=converter_with(["CPUExecutionProvider"])), \
                 patch("engine.features.providers.ort.get_available_providers", return_value=["CUDAExecutionProvider"]):
                with self.assertRaisesRegex(RuntimeError, "expected CUDAExecutionProvider"):
                    manager._converter("test", "cuda", server.Job(id="test", request={}), ort_cuda=True)
                self.assertEqual(manager._converters, {})
        finally:
            manager._executor.shutdown()

    def test_file_conversion_receives_feature_cuda_option(self):
        from fastapi.testclient import TestClient

        with TestClient(server.app) as client, \
             patch("engine.web.server.model_path", return_value=Path("test.pth")), \
             patch.object(server.manager, "submit") as submit:
            result = client.post("/api/jobs", data={"model": "test", "ort_cuda": "true"},
                                 files={"audio": ("test.wav", b"test", "audio/wav")})
            self.assertEqual(result.status_code, 202)
            self.assertTrue(submit.call_args.args[0].request["ort_cuda"])
        # Existing requests without the option must keep the CPU default.
        with TestClient(server.app) as client, \
             patch("engine.web.server.model_path", return_value=Path("test.pth")), \
             patch.object(server.manager, "submit") as submit:
            result = client.post("/api/jobs", data={"model": "test"},
                                 files={"audio": ("test.wav", b"test", "audio/wav")})
            self.assertEqual(result.status_code, 202)
            self.assertFalse(submit.call_args.args[0].request["ort_cuda"])
