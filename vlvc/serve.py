"""ブラウザ UI の起動（旧 ``Start-Web.ps1``）。

依存関係を確認してから FastAPI サーバーを前面で起動し、既定のブラウザで開く。
停止は Ctrl+C。変換はこの PC 上で実行され、音声は外部に送信されない。
"""

from __future__ import annotations

import threading
import webbrowser

from . import pyproc, ui


def run(python: str | None = None, port: int = 8765, host: str = "127.0.0.1", no_browser: bool = False) -> int:
    ui.header("VL_VoiceChanger / Web UI")
    try:
        pyproc.ensure_web_deps(python)
    except Exception as exc:  # noqa: BLE001 - 依存関係の失敗はメッセージで返す
        ui.err(str(exc))
        return 1

    url = f"http://{host}:{port}"
    ui.info(f"URL: {url}")
    if not no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    ui.info("停止は Ctrl+C（このウィンドウを閉じてもよい）")

    try:
        proc = pyproc.engine(["serve", "--host", host, "--port", str(port)], python=python, capture=False)
    except KeyboardInterrupt:
        ui.info("停止しました。")
        return 0
    return proc.returncode
