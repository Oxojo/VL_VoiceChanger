"""ハードウェア・OS 情報の収集（Windows / macOS / Linux）。

取れない項目は無理に埋めず、空欄・None のまま返す。環境確認（``vlvc doctor``）で使う。
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

IS_WINDOWS = os.name == "nt"
IS_MAC = sys.platform == "darwin"


def _run(command: list[str], timeout: float = 15) -> str:
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout or ""


def _psutil():
    try:
        import psutil

        return psutil
    except ImportError:
        return None


# ------------------------------------------------------------------ 基本情報
def hostname() -> str:
    return platform.node() or "unknown"


def os_summary() -> tuple[str, str]:
    system = platform.system() or os.name
    if IS_WINDOWS:
        release, version, _, _ = platform.win32_ver()
        return f"Windows {release}".strip(), version or platform.version()
    if IS_MAC:
        return f"macOS {platform.mac_ver()[0]}", platform.release()
    return system, f"{platform.release()} ({platform.machine()})"


def _cpu_name() -> str:
    if IS_WINDOWS:
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return re.sub(r"\s+", " ", str(name)).strip()
        except OSError:
            return platform.processor() or "unknown"
    if IS_MAC:
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def cpu_summary() -> tuple[str, int, int]:
    psutil = _psutil()
    name = _cpu_name()
    if psutil is not None:
        cores = psutil.cpu_count(logical=False) or 0
        threads = psutil.cpu_count(logical=True) or os.cpu_count() or 0
    else:
        cores = os.cpu_count() or 0
        threads = cores
    return name, cores, threads


def memory_gb() -> float:
    psutil = _psutil()
    if psutil is not None:
        return round(psutil.virtual_memory().total / 1024**3, 1)
    return 0.0


def disk_free_gb(path: Path) -> float | None:
    try:
        usage = shutil.disk_usage(str(path))
    except OSError:
        return None
    return round(usage.free / 1024**3, 1)


# ------------------------------------------------------------------ GPU
def _nvidia_gpus() -> list[dict[str, Any]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    text = _run(
        [
            exe,
            "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, Any]] = []
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 7:
            continue
        try:
            gpus.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "driver": fields[2],
                    "vram_mb": float(fields[3]),
                    "vram_used_mb": float(fields[4]),
                    "util_pct": float(fields[5]),
                    "temp_c": float(fields[6]),
                    "source": "nvidia-smi",
                }
            )
        except ValueError:
            continue
    return gpus


def _windows_gpus() -> list[dict[str, Any]]:
    script = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,DriverVersion,AdapterRAM | ConvertTo-Json -Compress"
    )
    text = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script])
    if not text.strip():
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    items = data if isinstance(data, list) else [data]
    gpus = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        vram = entry.get("AdapterRAM") or 0
        gpus.append(
            {
                "index": None,
                "name": entry.get("Name") or "unknown",
                "driver": entry.get("DriverVersion") or "",
                "vram_mb": round(float(vram) / 1024**2) if vram else None,
                "vram_used_mb": None,
                "util_pct": None,
                "temp_c": None,
                "source": "Win32_VideoController",
            }
        )
    return gpus


def _mac_gpus() -> list[dict[str, Any]]:
    text = _run(["system_profiler", "SPDisplaysDataType"], timeout=30)
    gpus = []
    name = ""
    vram = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("Chipset Model:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("VRAM"):
            match = re.search(r"(\d+)\s*(MB|GB)", line)
            if match:
                value = float(match.group(1))
                vram = value * 1024 if match.group(2) == "GB" else value
        elif line.startswith("Metal") and name:
            gpus.append(
                {
                    "index": None,
                    "name": name,
                    "driver": line.split(":", 1)[1].strip(),
                    "vram_mb": vram,
                    "vram_used_mb": None,
                    "util_pct": None,
                    "temp_c": None,
                    "source": "system_profiler",
                }
            )
            name = ""
    return gpus


def _linux_gpus() -> list[dict[str, Any]]:
    if not shutil.which("lspci"):
        return []
    text = _run(["lspci"])
    gpus = []
    for line in text.splitlines():
        if re.search(r"(vga|3d|display)", line, re.IGNORECASE):
            name = line.split(":", 2)[-1].strip()
            gpus.append(
                {
                    "index": None,
                    "name": name,
                    "driver": "",
                    "vram_mb": None,
                    "vram_used_mb": None,
                    "util_pct": None,
                    "temp_c": None,
                    "source": "lspci",
                }
            )
    return gpus


def gpus() -> list[dict[str, Any]]:
    found = _nvidia_gpus()
    if found:
        return found
    if IS_WINDOWS:
        return _windows_gpus()
    if IS_MAC:
        return _mac_gpus()
    return _linux_gpus()


def gpu_sample() -> dict[str, float] | None:
    """計測用の 1 サンプル（使用率・VRAM・温度）。nvidia-smi が無ければ None。"""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    text = _run(
        [exe, "--query-gpu=utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader,nounits"]
    )
    line = next((line for line in text.splitlines() if line.strip()), "")
    if not line:
        return None
    fields = [f.strip() for f in line.split(",")]
    if len(fields) < 3:
        return None
    try:
        return {"util_pct": float(fields[0]), "mem_mb": float(fields[1]), "temp_c": float(fields[2])}
    except ValueError:
        return None


# ------------------------------------------------------------------ 音声デバイス
def _windows_audio() -> list[dict[str, Any]]:
    script = (
        "Get-PnpDevice -Class AudioEndpoint -Status OK | "
        "Select-Object FriendlyName,InstanceId,Status | ConvertTo-Json -Compress"
    )
    text = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout=30)
    devices: list[dict[str, Any]] = []
    if text.strip():
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = []
        items = data if isinstance(data, list) else [data]
        for entry in items:
            if not isinstance(entry, dict):
                continue
            instance = str(entry.get("InstanceId") or "")
            # MMDevAPI: {0.0.1.x} が録音（input）、{0.0.0.x} が再生（output）
            kind = "input" if "{0.0.1." in instance else "output"
            devices.append(
                {
                    "name": entry.get("FriendlyName") or "unknown",
                    "type": kind,
                    "status": entry.get("Status") or "",
                    "source": "Get-PnpDevice",
                }
            )
    if devices:
        return devices

    script = "Get-CimInstance Win32_SoundDevice | Select-Object Name,Status | ConvertTo-Json -Compress"
    text = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout=30)
    if text.strip():
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = []
        items = data if isinstance(data, list) else [data]
        for entry in items:
            if isinstance(entry, dict):
                devices.append(
                    {
                        "name": entry.get("Name") or "unknown",
                        "type": "unknown",
                        "status": entry.get("Status") or "",
                        "source": "Win32_SoundDevice",
                    }
                )
    return devices


def _mac_audio() -> list[dict[str, Any]]:
    text = _run(["system_profiler", "SPAudioDataType", "-json"], timeout=30)
    if not text.strip():
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    devices: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            name = node.get("_name")
            if name and any(key.startswith("coreaudio_device_") for key in node):
                kind = "unknown"
                if node.get("coreaudio_device_input") or node.get("coreaudio_default_audio_input_device"):
                    kind = "input"
                elif node.get("coreaudio_device_output") or node.get("coreaudio_default_audio_output_device"):
                    kind = "output"
                devices.append({"name": str(name), "type": kind, "status": "", "source": "system_profiler"})
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return devices


def _linux_audio() -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for kind, subcommand in (("input", "sources"), ("output", "sinks")):
        if not shutil.which("pactl"):
            break
        text = _run(["pactl", "list", "short", subcommand])
        for line in text.splitlines():
            fields = line.split("\t")
            if len(fields) >= 2:
                devices.append({"name": fields[1], "type": kind, "status": "", "source": "pactl"})
    if devices:
        return devices

    try:
        text = Path("/proc/asound/cards").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        match = re.match(r"\s*\d+\s+\[([^\]]+)\]:\s*(.+)", line)
        if match:
            devices.append(
                {
                    "name": f"{match.group(1).strip()} - {match.group(2).strip()}",
                    "type": "unknown",
                    "status": "",
                    "source": "/proc/asound/cards",
                }
            )
    return devices


def audio_devices() -> list[dict[str, Any]]:
    if IS_WINDOWS:
        return _windows_audio()
    if IS_MAC:
        return _mac_audio()
    return _linux_audio()
