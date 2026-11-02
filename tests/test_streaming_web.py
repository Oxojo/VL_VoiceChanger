"""Exercise the actual WebSocket transport without downloading model weights."""

from __future__ import annotations

import io
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from engine.web import server
from engine.web.stream import register_stream_routes


class FakeConverter:
    info = SimpleNamespace(sample_rate=48000)

    def __init__(self):
        self.calls = []
        session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
        self.contentvec = self.rmvpe = SimpleNamespace(session=session)

    def convert_array(self, samples, options):
        self.calls.append((samples.copy(), options, threading.current_thread().name))
        return np.repeat(samples, 3), None


class TestStreamingWeb(unittest.TestCase):
    def setUp(self):
        self.manager = server.JobManager()
        self.fake = FakeConverter()
        self.converter_patch = patch.object(self.manager, "_converter", return_value=self.fake)
        self.converter_patch.start()
        self.app = FastAPI()
        register_stream_routes(self.app, self.manager)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.converter_patch.stop()
        self.manager._executor.shutdown(wait=True)

    def start(self, ws, **options):
        ws.send_json({"type": "start", "model": "test", "warmup": False, **options})
        ready = ws.receive_json()
        self.assertEqual(ready["type"], "ready", ready)
        self.assertEqual(ready["input_sr"], 16000)
        self.assertEqual(ready["output_sr"], 48000)
        return ready

    def receive(self, ws):
        message = ws.receive()
        output = np.empty(0, dtype=np.float32)
        if message.get("bytes") is not None:
            output = np.frombuffer(message["bytes"], dtype="<f4")
            message = ws.receive()
        import json
        return output, json.loads(message["text"])

    def test_pcm_roundtrip_arbitrary_packets_and_flush(self):
        audio = np.linspace(-0.5, 0.5, 160 * 61 + 103, dtype=np.float32)
        parts = []
        with self.client.websocket_connect("/api/stream") as ws:
            ready = self.start(ws, hop=0.2, context=0.06, crossfade=0.04, pitch=12)
            self.assertEqual(ready["buffer_ms"], 300)
            offset = index = 0
            while offset < audio.size:
                n = [1, 7, 160, 320, 911][index % 5]
                packet = audio[offset:offset + n]
                ws.send_bytes(packet.astype("<f4").tobytes())
                output, ack = self.receive(ws)
                self.assertEqual(ack["type"], "ack")
                self.assertEqual(ack["input_frames"], (offset + packet.size) // 160)
                parts.append(output)
                offset += packet.size
                index += 1
            ws.send_json({"type": "stop"})
            output, done = self.receive(ws)
            parts.append(output)
            self.assertEqual(done["type"], "done")
            self.assertEqual(done["emitted_frames"], 61)
            self.assertEqual(done["chunks"], 4)
        np.testing.assert_allclose(np.concatenate(parts), np.repeat(audio[:160 * 61], 3), atol=2e-7)
        self.assertFalse(self.manager._stream_active)
        self.assertTrue(all(call[1].pitch_shift == 12 for call in self.fake.calls))
        self.assertTrue(all(call[2].startswith("vc-job") for call in self.fake.calls))

    def test_short_tail_and_reconnect_have_independent_state(self):
        for n in (159, 160, 320):
            with self.subTest(n=n), self.client.websocket_connect("/api/stream") as ws:
                self.start(ws)
                ws.send_bytes(np.ones(n, dtype="<f4").tobytes())
                output, ack = self.receive(ws)
                self.assertEqual(output.size, 0)
                self.assertEqual(ack["emitted_frames"], 0)
                ws.send_json({"type": "stop"})
                output, done = self.receive(ws)
                self.assertEqual(output.size, n // 160 * 480)
                self.assertEqual(done["input_frames"], n // 160)
                np.testing.assert_array_equal(output, 1)

    def test_invalid_settings_release_session(self):
        for options in ({"hop": 0}, {"hop": float("nan")}, {"context": -1},
                        {"denoise": 0.5}, {"dehum": 50}, {"output_sr": 22050},
                        {"device": "invalid"}, {"type": "other"}, {"unknown": True}):
            with self.subTest(options=options), self.client.websocket_connect("/api/stream") as ws:
                ws.send_json({"type": "start", "model": "test", **options})
                self.assertEqual(ws.receive_json()["type"], "error")
            self.assertFalse(self.manager._stream_active)
        with self.client.websocket_connect("/api/stream") as ws:
            self.start(ws)

    def test_invalid_audio_is_rejected(self):
        for data in (b"", b"bad", bytes(6404), np.array([np.nan], dtype="<f4").tobytes(),
                     np.array([np.inf], dtype="<f4").tobytes()):
            with self.subTest(length=len(data)), self.client.websocket_connect("/api/stream") as ws:
                self.start(ws)
                ws.send_bytes(data)
                self.assertEqual(ws.receive_json()["type"], "error")
            self.assertFalse(self.manager._stream_active)

    def test_invalid_control_and_duration_limit(self):
        for control in ({"type": "reset"}, {"type": "stop", "extra": 1}, "not json"):
            with self.subTest(control=control), self.client.websocket_connect("/api/stream") as ws:
                self.start(ws)
                ws.send_json(control)
                self.assertEqual(ws.receive_json()["type"], "error")
        with patch("engine.web.stream.MAX_STREAM_FRAMES", 1):
            with self.client.websocket_connect("/api/stream") as ws:
                self.start(ws)
                ws.send_bytes(np.zeros(161, dtype="<f4").tobytes())
                self.assertEqual(ws.receive_json()["type"], "error")

    def test_busy_stream_blocks_other_sessions_and_offline_jobs(self):
        with self.client.websocket_connect("/api/stream") as first:
            self.start(first)
            with self.client.websocket_connect("/api/stream") as second:
                second.send_json({"model": "test"})
                self.assertEqual(second.receive_json()["type"], "error")
            self.assertTrue(self.manager._stream_active)
            with self.assertRaises(HTTPException) as error:
                self.manager.submit(server.Job(id="blocked", request={}),
                                    UploadFile(file=io.BytesIO(b""), filename="input.wav"))
            self.assertEqual(error.exception.status_code, 409)
        self.assertFalse(self.manager._stream_active)
        self.manager._jobs["busy"] = server.Job(id="busy", request={})
        with self.client.websocket_connect("/api/stream") as ws:
            ws.send_json({"model": "test"})
            self.assertEqual(ws.receive_json()["type"], "error")

    def test_missing_assets_and_backend_errors_are_visible(self):
        for exc in (FileNotFoundError("missing ONNX"), ValueError("bad model output length")):
            with self.subTest(exc=exc), patch.object(self.manager, "_converter", side_effect=exc):
                with self.client.websocket_connect("/api/stream") as ws:
                    ws.send_json({"model": "test"})
                    self.assertIn(str(exc), ws.receive_json()["message"])
            self.assertFalse(self.manager._stream_active)
        with patch.object(self.fake, "convert_array", side_effect=ValueError("bad model output length")):
            with self.client.websocket_connect("/api/stream") as ws:
                self.start(ws, hop=0.1, context=0, crossfade=0)
                ws.send_bytes(np.zeros(1600, dtype="<f4").tobytes())
                self.assertIn("bad model output length", ws.receive_json()["message"])

    def test_cross_origin_rejected_and_idle_session_released(self):
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect("/api/stream", headers={"origin": "https://other.example"}):
                pass
        with patch("engine.web.stream.IDLE_SECONDS", 0.01):
            with self.client.websocket_connect("/api/stream", headers={"origin": "http://testserver"}) as ws:
                self.start(ws)
                self.assertEqual(ws.receive_json()["type"], "error")
        self.assertFalse(self.manager._stream_active)

    def test_inference_does_not_block_http_event_loop(self):
        entered, release = threading.Event(), threading.Event()

        @self.app.get("/probe")
        async def probe():
            return {"ok": True}

        original = self.fake.convert_array

        def slow(*args):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("worker timeout")
            return original(*args)

        with patch.object(self.fake, "convert_array", side_effect=slow):
            with self.client.websocket_connect("/api/stream") as ws:
                self.start(ws, hop=0.1, context=0, crossfade=0)
                ws.send_bytes(np.zeros(1600, dtype="<f4").tobytes())
                try:
                    self.assertTrue(entered.wait(2))
                    self.assertTrue(self.client.get("/probe").json()["ok"])
                finally:
                    release.set()
                _, ack = self.receive(ws)
                self.assertEqual(ack["type"], "ack")

    def test_cuda_selection_and_stage_timings_reach_browser(self):
        session = SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.fake.contentvec = self.fake.rmvpe = SimpleNamespace(session=session)
        original = self.fake.convert_array

        def profiled(*args):
            out, _ = original(*args)
            return out, SimpleNamespace(timings={"f0": 0.002, "contentvec": 0.003, "synth": 0.004, "post": 0.001})

        with patch.object(self.fake, "convert_array", side_effect=profiled):
            with self.client.websocket_connect("/api/stream") as ws:
                ready = self.start(ws, ort_cuda=True, hop=0.1, context=0, crossfade=0)
                self.assertTrue(self.manager._converter.call_args.kwargs["ort_cuda"])
                self.assertEqual(ready["feature_providers"]["contentvec"][0], "CUDAExecutionProvider")
                for _ in range(2):
                    ws.send_bytes(np.zeros(1600, dtype="<f4").tobytes())
                    _, ack = self.receive(ws)
                    self.assertEqual(ack["stage_ms"], {"f0": 2, "contentvec": 3, "synth": 4, "post": 1})

    def test_debug_is_opt_in_and_preserves_audio_protocol(self):
        with self.client.websocket_connect("/api/stream") as ws:
            ready = self.start(ws, hop=0.1, context=0, crossfade=0)
            self.assertNotIn("debug", ready)
            ws.send_bytes(np.ones(1600, dtype="<f4").tobytes())
            output, ack = self.receive(ws)
            np.testing.assert_array_equal(output, 1)
            self.assertNotIn("debug", ack)

    def test_debug_correlates_requests_output_frames_and_server_durations(self):
        with self.client.websocket_connect("/api/stream") as ws:
            ready = self.start(ws, debug=True, hop=0.1, context=0, crossfade=0)
            self.assertEqual(ready["debug"], {"hop_ms": 100, "context_ms": 0, "crossfade_ms": 0})
            for i in range(1, 6):
                ws.send_bytes(np.ones(320, dtype="<f4").tobytes())
                output, ack = self.receive(ws)
                d = ack["debug"]
                self.assertEqual(d["request_seq"], i)
                self.assertEqual(d["output_start_frame"], 0)
                self.assertEqual(d["output_frames"], 10 if i == 5 else 0)
                self.assertEqual(output.size, 4800 if i == 5 else 0)
                self.assertGreaterEqual(d["server_ms"], d["worker_ms"])
                self.assertGreaterEqual(d["worker_ms"], d["inference_ms"])
                self.assertGreaterEqual(d["executor_wait_ms"], 0)
            ws.send_json({"type": "stop"})
            _, done = self.receive(ws)
            self.assertEqual(done["debug"]["request_seq"], 6)
            self.assertEqual(done["debug"]["output_start_frame"], 10)
            self.assertEqual(done["debug"]["output_frames"], 0)

    def test_debug_flush_keeps_short_tail_and_new_sessions_reset_sequence(self):
        for _ in range(2):
            with self.client.websocket_connect("/api/stream") as ws:
                self.start(ws, debug=True)
                ws.send_bytes(np.ones(160, dtype="<f4").tobytes())
                _, ack = self.receive(ws)
                self.assertEqual(ack["debug"]["request_seq"], 1)
                ws.send_json({"type": "stop"})
                out, done = self.receive(ws)
                self.assertEqual(out.size, 480)
                self.assertEqual(done["debug"]["request_seq"], 2)
                self.assertEqual(done["debug"]["output_frames"], 1)

    def test_debug_executor_wait_excludes_model_inference(self):
        import time

        release, queued = threading.Event(), threading.Event()
        with self.client.websocket_connect("/api/stream") as ws:
            self.start(ws, debug=True, hop=0.1, context=0, crossfade=0)
            blocker = self.manager._executor.submit(release.wait, 3)
            submit = self.manager._executor.submit

            def notify_submit(*args, **kwargs):
                future = submit(*args, **kwargs)
                queued.set()
                return future

            try:
                with patch.object(self.manager._executor, "submit", side_effect=notify_submit):
                    ws.send_bytes(np.ones(1600, dtype="<f4").tobytes())
                    self.assertTrue(queued.wait(2))
                    time.sleep(0.025)
                    release.set()
                    _, ack = self.receive(ws)
                    trace = ack["debug"]
                    self.assertGreaterEqual(trace["executor_wait_ms"], 20)
                    self.assertGreaterEqual(trace["server_ms"], trace["executor_wait_ms"] + trace["worker_ms"])
            finally:
                release.set()
                blocker.result(timeout=3)

    def test_warmup_covers_first_and_steady_shapes_without_committing_audio(self):
        # Context > hop exercises the full steady-state length immediately.
        with self.client.websocket_connect("/api/stream") as ws:
            ready = self.start(ws, warmup=True, debug=True, hop=0.1, context=0.3, crossfade=0.02)
            self.assertGreater(ready["warmup_ms"], 0)
            self.assertEqual([call[0].size for call in self.fake.calls], [160 * 42, 160 * 72])
            ws.send_bytes(np.ones(320, dtype="<f4").tobytes())
            _, ack = self.receive(ws)
            self.assertEqual(ack["input_frames"], 2)
            self.assertEqual(ack["chunks"], 0)
            self.assertEqual(ack["emitted_frames"], 0)
            self.assertEqual(ack["max_inference_ms"], 0)
            self.assertEqual(ack["stage_ms"], {})
            self.assertEqual(ack["debug"]["request_seq"], 1)
            ws.send_json({"type": "stop"})
            output, done = self.receive(ws)
            np.testing.assert_array_equal(output, 1)
            self.assertEqual(done["input_frames"], 2)
            self.assertEqual(done["chunks"], 1)

    def test_bad_output_during_warmup_is_reported_before_ready(self):
        with patch.object(self.fake, "convert_array", return_value=(np.empty(0), None)):
            with self.client.websocket_connect("/api/stream") as ws:
                ws.send_json({"model": "test"})
                error = ws.receive_json()
                self.assertEqual(error["type"], "error")
                self.assertIn("converter returned", error["message"])
        self.assertFalse(self.manager._stream_active)

    def test_low_latency_settings_preserve_frames_and_warm_both_windows(self):
        with self.client.websocket_connect("/api/stream") as ws:
            ready = self.start(ws, warmup=True, debug=True, hop=0.2, context=0.06, crossfade=0.02)
            self.assertEqual(ready["buffer_ms"], 280)
            self.assertEqual([call[0].size for call in self.fake.calls], [160 * 28, 160 * 34])
            pieces = []
            for i in range(40):
                ws.send_bytes(np.ones(320, dtype="<f4").tobytes())
                output, ack = self.receive(ws)
                pieces.append(output)
                if i < 13:
                    self.assertEqual(output.size, 0)
                if i == 13:
                    self.assertEqual(ack["emitted_frames"], 20)
                    self.assertEqual(output.size, 9600)
            ws.send_json({"type": "stop"})
            output, done = self.receive(ws)
            pieces.append(output)
            result = np.concatenate(pieces)
            self.assertEqual(result.size, 80 * 480)
            self.assertEqual(done["input_frames"], 80)
            self.assertEqual(done["emitted_frames"], 80)
            self.assertEqual(done["chunks"], 4)
            np.testing.assert_array_equal(result, 1)


if __name__ == "__main__":
    unittest.main()
