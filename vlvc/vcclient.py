"""VC Client（外部アプリ）との連携。

ポート監視・REST API・プロセス検索・起動待ちを OS 非依存でまとめる。
VC Client 本体は Windows 向け配布だが、この CLI 自体はどの OS でも動く
（VC Client を使わない自前エンジンだけなら Windows 以外でも問題ない）。
"""

from __future__ import annotations

import fnmatch
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from . import ui

IS_WINDOWS = os.name == "nt"


class VcApiError(RuntimeError):
    """VC Client の API 呼び出しに失敗した。"""

    def __init__(self, message: str, status: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


# ------------------------------------------------------------------ ポート
def port_open(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_port(
    port: int,
    timeout: float,
    host: str = "127.0.0.1",
    interval: float = 0.5,
    label: str = "",
    on_tick: Callable[[float], None] | None = None,
) -> bool:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if port_open(port, host, timeout=0.5):
            return True
        if on_tick:
            on_tick(time.monotonic() - started)
        time.sleep(interval)
    return port_open(port, host, timeout=0.5)


# ------------------------------------------------------------------ API
def _request(
    port: int,
    path: str,
    method: str = "GET",
    body: Any = None,
    timeout: float = 60,
    host: str = "127.0.0.1",
) -> Any:
    url = f"http://{host}:{port}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise VcApiError(f"HTTP {exc.code} ({method} {path})", status=exc.code, body=detail) from exc
    except urllib.error.URLError as exc:
        raise VcApiError(f"接続できません ({method} {path}): {exc.reason}") from exc
    except TimeoutError as exc:
        raise VcApiError(f"タイムアウト ({method} {path})") from exc

    text = raw.decode("utf-8", "replace").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise VcApiError(f"JSON として読めません ({method} {path}): {text[:200]}") from exc


def api(
    settings: Any,
    path: str,
    method: str = "GET",
    body: Any = None,
    timeout: float = 60,
) -> Any:
    return _request(settings.port, path, method=method, body=body, timeout=timeout, host=settings.host)


def api_ok(settings: Any) -> bool:
    try:
        _request(settings.port, "/api/hello", timeout=8, host=settings.host)
        return True
    except VcApiError:
        return False


def module_status(settings: Any) -> tuple[int, int]:
    """(有効なモジュール数, 全モジュール数)。取得できなければ (0, 0)。"""
    try:
        modules = _request(settings.port, "/api/module-manager/modules", timeout=10, host=settings.host)
    except VcApiError:
        return 0, 0
    items = modules if isinstance(modules, list) else []
    valid = sum(1 for m in items if isinstance(m, dict) and m.get("valid"))
    return valid, len(items)


# ------------------------------------------------------------------ プロセス
def _psutil():
    try:
        import psutil

        return psutil
    except ImportError as exc:  # pragma: no cover - 環境依存
        raise RuntimeError("psutil が必要です。`vlvc setup` か `pip install psutil` を実行してください。") from exc


def find_processes(root: str | Path = "", patterns: str = "") -> list[Any]:
    """VC Client のプロセスを探す。

    まず ``root`` 配下の実行ファイルから起動されたものを Path で判定し、
    見つからなければプロセス名のパターン（カンマ区切り・ワイルドカード可）で探す。
    """
    psutil = _psutil()
    found: dict[int, Any] = {}

    root_text = str(root or "")
    if root_text in {"", "."}:
        root_text = ""
    if root_text:
        try:
            resolved = Path(root_text).resolve()
        except OSError:
            resolved = Path(root_text)
        for proc in psutil.process_iter(["pid", "name", "exe"]):
            try:
                exe = proc.info.get("exe") or ""
                if exe and Path(exe).resolve().is_relative_to(resolved):
                    found[proc.pid] = proc
            except (psutil.Error, OSError, ValueError):
                continue

    if not found and patterns:
        for pattern in patterns.split(","):
            pattern = pattern.strip()
            if not pattern:
                continue
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    name = (proc.info.get("name") or "").lower()
                    if fnmatch.fnmatch(name, pattern.lower()) or fnmatch.fnmatch(
                        name.removesuffix(".exe"), pattern.lower()
                    ):
                        found[proc.pid] = proc
                except psutil.Error:
                    continue

    return sorted(found.values(), key=lambda p: _safe_memory(p), reverse=True)


def _safe_memory(proc: Any) -> float:
    try:
        return float(proc.memory_info().rss)
    except Exception:  # noqa: BLE001 - 取得できないプロセスは 0 扱い
        return 0.0


def listener_pid(port: int) -> int | None:
    """``port`` を待ち受けているプロセスの PID。取得できなければ None。"""
    try:
        psutil = _psutil()
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                return conn.pid
    except Exception:  # noqa: BLE001 - 権限が無い環境では判定しない
        return None
    return None


# ------------------------------------------------------------------ 起動
def launcher_path(settings: Any) -> Path:
    return settings.vc_root / settings.launcher


def launch(settings: Any) -> subprocess.Popen:
    """VC Client を起動する（別ウィンドウ／別セッション）。"""
    launcher = launcher_path(settings)
    if not launcher.is_file():
        raise FileNotFoundError(f"起動スクリプトが見つかりません: {launcher}")

    if IS_WINDOWS:
        # .bat は CreateProcess で直接実行できないため cmd 経由。別コンソールで開く。
        return subprocess.Popen(
            ["cmd", "/c", str(launcher)],
            cwd=str(settings.vc_root),
            creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP,  # type: ignore[attr-defined]
        )

    if launcher.suffix in {".sh", ".bash"}:
        return subprocess.Popen(["bash", str(launcher)], cwd=str(settings.vc_root), start_new_session=True)
    if not os.access(launcher, os.X_OK):
        os.chmod(launcher, 0o755)
    return subprocess.Popen([str(launcher)], cwd=str(settings.vc_root), start_new_session=True)


def _log_progress(log_path: Path) -> tuple[int, int]:
    """VC Client のログから必須モジュールのダウンロード進捗を数える。"""
    if not log_path.is_file():
        return -1, -1
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return -1, -1
    return text.count("Downloading completed:"), text.count("Start downloading:")


def wait_gui(
    settings: Any,
    timeout: float,
    log_path: Path | None = None,
    tick_seconds: float = 60,
) -> bool:
    """GUI（ポート）が応答するまで待つ。

    初回起動は必須モジュール（合計約 3.3 GB）のダウンロードで数十分かかる。
    ダウンロードが進んでいる間はタイムアウトを延長する（30 分進まなければ打ち切り）。
    """
    started = time.monotonic()
    deadline = started + timeout
    last_done, last_total = -1, -1
    last_progress = started
    last_tick = started

    while True:
        if port_open(settings.port, settings.host, timeout=0.5):
            return True

        if log_path is not None:
            done, total = _log_progress(log_path)
        else:
            done, total = -1, -1

        now = time.monotonic()
        if done >= 0 and (done != last_done or total != last_total):
            if last_done >= 0 and done != last_done and total > 0:
                ui.info(f"必須モジュールのダウンロード: {done}/{total} 件")
            last_done, last_total, last_progress = done, total, now

        downloading = done >= 0 and total > 0 and done < total and (now - last_progress) < 1800
        if not downloading and now >= deadline:
            return False

        if now - last_tick >= tick_seconds:
            ui.info(f"起動待機中… 経過 {int((now - started) / 60)} 分 / モジュール {done}/{total} 件")
            last_tick = now

        time.sleep(2)
