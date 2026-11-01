"""Start the real npm dev server and verify its HTTP/WS entry points."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import unittest
from urllib.request import urlopen

import websockets
import psutil


class TestDevLauncher(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the npm development launcher")
    def test_dev_server_lifecycle(self):
        root = Path(__file__).resolve().parents[1]
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        command = ([shutil.which("npm"), "run", "dev", "--"] if os.name != "nt" and shutil.which("npm")
                   else [shutil.which("node"), "scripts/dev.mjs"])
        launch_env = {**os.environ, "VIRTUAL_ENV": sys.prefix}
        launch_env.pop("VLVC_PYTHON", None)
        process = subprocess.Popen(
            [*command, "--port", str(port), "--no-open"],
            cwd=root, env=launch_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            start_new_session=os.name != "nt",
        )
        try:
            deadline = time.monotonic() + 20
            while True:
                try:
                    with urlopen(base + "/api/health", timeout=1) as response:
                        self.assertEqual(json.load(response)["app"], "VL VoiceChanger")
                    break
                except OSError:
                    if process.poll() is not None or time.monotonic() > deadline:
                        self.fail("dev server did not become ready")
                    time.sleep(0.1)
            with urlopen(base) as response:
                self.assertIn("liveStartBtn", response.read().decode())
            with urlopen(base + "/static/live.js") as response:
                self.assertEqual(response.status, 200)
            with urlopen(base + "/api/models") as response:
                self.assertIn("models", json.load(response))
            self.assertTrue(any(
                child.cmdline() and Path(child.cmdline()[0]).parent == Path(sys.executable).parent
                for child in psutil.Process(process.pid).children(recursive=True)
            ), "the activated environment must take priority over a repository .venv")

            async def ws_request():
                async with websockets.connect(f"ws://127.0.0.1:{port}/api/stream", origin=base) as ws:
                    await ws.send(json.dumps({"model": "__missing_launcher_test_model__"}))
                    error = json.loads(await ws.recv())
                    self.assertEqual(error["type"], "error")
            asyncio.run(ws_request())
        finally:
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/pid", str(process.pid), "/t", "/f"], capture_output=True)
                else:
                    os.killpg(process.pid, signal.SIGINT)
            try:
                output, _ = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                output, _ = process.communicate(timeout=5)
                self.fail("dev server did not stop: " + output[-2000:])
        self.assertIn("Started reloader process", output)
        self.assertIn("Finished server process", output)
        # The listening socket must be released after Ctrl+C.
        with socket.socket() as sock:
            self.assertNotEqual(sock.connect_ex(("127.0.0.1", port)), 0)


if __name__ == "__main__":
    unittest.main()
