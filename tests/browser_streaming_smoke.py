"""Optional Chromium test: real capture/worklet/WebSocket/playback, fake RVC.

Run with `python tests/browser_streaming_smoke.py -v` after installing Playwright
and Chromium (`VLVC_TEST_CHROMIUM` can select a system executable). The fake
backend is installed only in this test process, never exposed by the application.
"""

from __future__ import annotations

import multiprocessing
import math
from array import array
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import time
import unittest
import wave
from types import SimpleNamespace
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def serve_fixture(port):
    import numpy as np
    import uvicorn
    from engine.web import server

    class Converter:
        info = SimpleNamespace(sample_rate=48000)

        def __init__(self, model, ort_cuda=False):
            self.model = model
            self.calls = 0
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort_cuda else ["CPUExecutionProvider"]
            session = SimpleNamespace(get_providers=lambda: providers)
            self.contentvec = self.rmvpe = SimpleNamespace(session=session)

        def convert_array(self, samples, options):
            self.calls += 1
            if self.model == "cold" and self.calls == 1:
                time.sleep(2.3)
            if self.model == "slow":
                time.sleep(3)
            if self.model == "broken":
                raise ValueError("test backend output failed")
            return np.repeat(samples * 0.5, 3), SimpleNamespace(timings={
                "f0": 0.002, "contentvec": 0.003, "synth": 0.004, "post": 0.001,
            })

    server.list_models = lambda: [
        {"id": name, "name": name, "size_mb": 0} for name in ["test", "slow", "broken", "cold"]
    ]
    server.manager.model_info = lambda model: {"sample_rate": 48000, "speakers": 1}
    server.manager._converter = lambda model, device, job, *, ort_cuda=False: Converter(model, ort_cuda)

    @server.app.get("/test/idle")
    async def idle():
        return {"idle": not server.manager._stream_active}

    uvicorn.run(server.app, host="127.0.0.1", port=port, log_level="warning",
                ws_max_size=6400, ws_max_queue=4)


class BrowserStreamingSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.url = f"http://127.0.0.1:{port}"
        cls.process = multiprocessing.get_context("spawn").Process(target=serve_fixture, args=(port,))
        cls.process.start()
        try:
            deadline = time.monotonic() + 20
            while True:
                try:
                    with urlopen(cls.url + "/api/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    if not cls.process.is_alive() or time.monotonic() > deadline:
                        raise RuntimeError("fixture server did not start")
                    time.sleep(0.1)
            cls.playwright = sync_playwright().start()
            cls.audio_dir = tempfile.TemporaryDirectory(prefix="vlvc-mic-")
            tone_path = str(Path(cls.audio_dir.name) / "microphone.wav")
            with wave.open(tone_path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(48000)
                wav.writeframes(array("h", (int(6000 * math.sin(2 * math.pi * 440 * i / 48000))
                                           for i in range(96000))).tobytes())
            executable = os.environ.get("VLVC_TEST_CHROMIUM") or shutil.which("chromium")
            cls.browser = cls.playwright.chromium.launch(
                executable_path=executable, headless=True,
                args=["--no-sandbox", "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                      f"--use-file-for-fake-audio-capture={tone_path}"],
            )
        except Exception:
            cls.process.terminate()
            cls.process.join(5)
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.audio_dir.cleanup()
        cls.process.terminate()
        cls.process.join(5)

    def setUp(self):
        self.context = self.browser.new_context(permissions=["microphone"])
        self.page = self.context.new_page()
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.add_init_script("""(() => {
          window.testPlayback = [];
          window.testTracks = [];
          window.testSockets = [];
          const OriginalWebSocket = window.WebSocket;
          window.WebSocket = class extends OriginalWebSocket {
            constructor(...args) { super(...args); window.testSockets.push(this); }
          };
          const original = AudioContext.prototype.createBufferSource;
          AudioContext.prototype.createBufferSource = function(...args) {
            const source = original.apply(this, args);
            const start = source.start.bind(source);
            source.start = (...times) => {
              const samples = source.buffer.getChannelData(0);
              let peak = 0;
              for (const x of samples) peak = Math.max(peak, Math.abs(x));
              window.testPlayback.push({length: samples.length, rate: source.buffer.sampleRate, peak});
              return start(...times);
            };
            return source;
          };
          const getMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
          navigator.mediaDevices.getUserMedia = async (...args) => {
            if (window.testDenyMic) throw new DOMException('denied for test', 'NotAllowedError');
            const media = await getMedia(...args);
            window.testTracks.push(...media.getTracks());
            if (window.testMicDelay) await new Promise(r => setTimeout(r, window.testMicDelay));
            return media;
          };
        })();""")
        self.page.goto(self.url)
        self.page.wait_for_function("!document.getElementById('liveStartBtn').disabled")
        self.page.locator('input[name="mode"][value="chunk"]').check()
        self.page.locator("#hop").fill("0.1")
        self.page.locator("#context").fill("0")
        self.page.locator("#crossfade").fill("0.02")

    def tearDown(self):
        self.context.close()
        deadline = time.monotonic() + 10
        while True:
            import json
            with urlopen(self.url + "/test/idle", timeout=1) as response:
                if json.load(response)["idle"]:
                    break
            if time.monotonic() > deadline:
                self.fail("stream reservation was not released")
            time.sleep(0.05)
        self.assertEqual(self.errors, [])

    def start(self, model="test"):
        self.page.locator("#modelSelect").select_option(model)
        self.page.locator("#liveStartBtn").click()

    def wait_stopped(self):
        self.page.wait_for_function("!document.getElementById('liveStartBtn').disabled", timeout=15000)
        self.assertTrue(self.page.evaluate("window.testTracks.every(t => t.readyState === 'ended')"))
        self.assertFalse(self.page.locator("#deviceSelect").is_disabled())

    def test_microphone_playback_stop_and_restart(self):
        for run in range(2):
            baseline = self.page.evaluate("window.testPlayback.length")
            self.start()
            self.page.wait_for_function(f"window.testPlayback.length >= {baseline + 3}", timeout=15000)
            self.assertTrue(self.page.locator("#modelSelect").is_disabled())
            self.page.locator("#liveStopBtn").click()
            self.wait_stopped()
            self.assertEqual(self.page.locator("#liveStatus").inner_text(), "停止しました")
        playback = self.page.evaluate("window.testPlayback")
        self.assertTrue(all(item["rate"] == 48000 for item in playback))
        self.assertTrue(any(item["peak"] > 0.001 for item in playback))
        self.assertIn("出力", self.page.locator("#liveMetrics").inner_text())

    def test_microphone_denial_and_recovery(self):
        self.page.evaluate("window.testDenyMic = true")
        self.start()
        self.page.wait_for_function("document.getElementById('liveStatus').textContent.includes('denied')")
        self.wait_stopped()
        self.page.evaluate("window.testDenyMic = false")
        self.start()
        self.page.wait_for_function("window.testPlayback.length > 0")
        self.page.locator("#liveStopBtn").click()
        self.wait_stopped()

    def test_backend_failure_is_visible(self):
        self.start("broken")
        self.page.wait_for_function("document.getElementById('liveStatus').textContent.includes('test backend output failed')")
        self.wait_stopped()
        self.assertEqual(self.page.evaluate("window.testPlayback.length"), 0)

    def test_slow_backend_stops_instead_of_unbounded_buffering(self):
        self.start("slow")
        self.page.wait_for_function("document.getElementById('liveStatus').textContent.includes('追いつかない')", timeout=15000)
        self.wait_stopped()

    def test_record_and_stream_controls_coexist(self):
        self.page.locator("#recordBtn").click()
        self.page.wait_for_function("document.getElementById('recordBtn').classList.contains('on')")
        self.start()
        self.assertIn("録音を停止", self.page.locator("#liveStatus").inner_text())
        self.page.locator("#recordBtn").click()
        self.page.wait_for_function("!document.getElementById('recordBtn').classList.contains('on')")

    def test_disconnect_releases_microphone_and_allows_restart(self):
        self.start()
        self.page.wait_for_function("window.testPlayback.length > 0")
        self.page.evaluate("window.testSockets.at(-1).close()")
        self.page.wait_for_function("document.getElementById('liveStatus').textContent.includes('接続が切れ')")
        self.wait_stopped()
        baseline = self.page.evaluate("window.testPlayback.length")
        self.start()
        self.page.wait_for_function(f"window.testPlayback.length > {baseline}")
        self.page.locator("#liveStopBtn").click()
        self.wait_stopped()

    def test_stop_while_microphone_permission_is_pending(self):
        self.page.evaluate("window.testMicDelay = 500")
        self.start()
        self.page.wait_for_function("window.testTracks.length > 0")
        self.page.locator("#liveStopBtn").click()
        self.page.wait_for_function("window.testTracks.every(t => t.readyState === 'ended')")
        self.wait_stopped()
        self.assertEqual(self.page.evaluate("window.testPlayback.length"), 0)

    def test_feature_cuda_setting_and_profile_display(self):
        self.page.locator("#ortDevice").select_option("cuda")
        self.start()
        self.page.wait_for_function("window.testPlayback.length > 0")
        self.assertIn("CUDA 優先", self.page.locator("#liveProviders").inner_text())
        self.assertIn("ピッチ推定 2.0 ms", self.page.locator("#liveStages").inner_text())
        self.assertIn("音声特徴 3.0 ms", self.page.locator("#liveStages").inner_text())
        self.assertTrue(self.page.locator("#ortDevice").is_disabled())
        self.page.locator("#liveStopBtn").click()
        self.wait_stopped()
        self.assertFalse(self.page.locator("#ortDevice").is_disabled())

    def test_debug_mode_records_latency_and_exports_without_audio(self):
        import json

        self.page.locator("#liveDebug").check()
        self.assertTrue(self.page.locator("#debugPanel").is_visible())
        self.start()
        self.page.wait_for_function("document.getElementById('debugTotal').textContent.includes('ms')")
        self.assertTrue(self.page.locator("#liveDebug").is_disabled())
        self.assertIn("ピッチ推定", self.page.locator("#debugRows").inner_text())
        self.assertIn("未測定", self.page.locator("#debugPanel").inner_text())
        self.assertIn("hop 100", self.page.locator("#debugConfig").inner_text())
        self.page.screenshot(path=str(Path(__file__).resolve().parents[1] / "logs" / "debug-mode.png"), full_page=True)
        self.page.locator("#liveStopBtn").click()
        self.wait_stopped()
        with self.page.expect_download() as result:
            self.page.locator("#debugExport").click()
        export = json.loads(Path(result.value.path()).read_text())
        self.assertTrue(export["settings"]["debug"])
        self.assertEqual(export["reason"], "停止しました")
        self.assertGreater(len(export["records"]), 0)
        self.assertLessEqual(len(export["records"]), 300)
        self.assertGreater(export["records"][0]["graph_capture_to_schedule_ms"], 0)
        self.assertEqual(export["records"][0]["output_start_frame"], 0)
        self.assertAlmostEqual(export["records"][0]["accumulation_ms"], 120, places=3)
        self.assertEqual(export["client_state"]["pending_audio_ms"], 0)
        self.assertIsNone(export["client_state"]["inflight_elapsed_ms"])
        self.assertNotIn("samples", export["records"][0])
        self.page.locator("#liveDebug").uncheck()
        self.assertFalse(self.page.locator("#debugPanel").is_visible())

    def test_normal_mode_does_not_show_debug_panel(self):
        self.assertFalse(self.page.locator("#liveDebug").is_checked())
        self.start()
        self.page.wait_for_function("window.testPlayback.length > 0")
        self.assertFalse(self.page.locator("#debugPanel").is_visible())
        self.assertTrue(self.page.locator("#debugExport").is_disabled())
        self.page.locator("#liveStopBtn").click()
        self.wait_stopped()

    def test_debug_exports_failure_before_first_audio_output(self):
        import json

        self.page.locator("#liveDebug").check()
        self.start("slow")
        self.page.wait_for_function("document.getElementById('liveStatus').textContent.includes('追いつかない')", timeout=15000)
        self.wait_stopped()
        with self.page.expect_download() as result:
            self.page.locator("#debugExport").click()
        export = json.loads(Path(result.value.path()).read_text())
        self.assertEqual(export["records"], [])
        self.assertIn("追いつかない", export["reason"])
        self.assertGreater(export["client_state"]["pending_audio_ms"], 2000)
        self.assertGreater(export["client_state"]["inflight_elapsed_ms"], 0)
        self.assertIsNotNone(export["client_state"]["inflight_request_seq"])

    def test_cold_inference_finishes_before_microphone_opens(self):
        self.start("cold")
        self.page.wait_for_timeout(400)
        self.assertEqual(self.page.evaluate("window.testTracks.length"), 0)
        self.assertIn("初回推論", self.page.locator("#liveStatus").inner_text())
        self.page.wait_for_function("window.testPlayback.length >= 2", timeout=15000)
        self.assertIn("マイク開始前", self.page.locator("#liveWarmup").inner_text())
        self.assertNotIn("追いつかない", self.page.locator("#liveStatus").inner_text())
        self.page.locator("#liveStopBtn").click()
        self.wait_stopped()


if __name__ == "__main__":
    unittest.main()
