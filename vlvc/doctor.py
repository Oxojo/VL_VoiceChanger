"""環境確認（旧 ``Check-Environment.ps1``）。

機材・実行環境を調べ、Markdown と JSON のレポートを ``reports/`` に出す。
エディション判定の考え方は従来どおり（``web_front/assets/gui_settings`` を一次情報とし、
無ければフォルダ名から推定）。
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import re
from pathlib import Path
from typing import Any

from . import sysinfo, ui, vcclient
from .config import Settings, format_size, resolve_path

KIND_LABEL = {"input": "入力（録音）", "output": "出力（再生）", "unknown": "不明"}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _detect_edition(root: Path) -> tuple[str, str]:
    gui_dir = root / "web_front" / "assets" / "gui_settings"
    edition = _read_text(gui_dir / "edition.txt")
    version = _read_text(gui_dir / "version.txt")
    leaf = root.name
    if not edition:
        if re.search(r"cuda", leaf, re.IGNORECASE):
            edition = "cuda"
        elif re.search(r"dml|directml", leaf, re.IGNORECASE):
            edition = "dml"
        elif re.search(r"onnx", leaf, re.IGNORECASE):
            edition = "onnx"
    if not version:
        match = re.search(r"(\d+\.\d+\.\d+[-\w.]*)", leaf)
        if match:
            version = match.group(1)
    return edition, version


def collect(settings: Settings, vc_root_override: str | None = None) -> dict[str, Any]:
    findings: list[dict[str, str]] = []

    def finding(level: str, message: str) -> None:
        findings.append({"level": level, "message": message})

    vc_root_value = vc_root_override or str(settings.get("vcClient.root", "") or "")
    vc_root = resolve_path(vc_root_value) if vc_root_value else Path()
    port = settings.port

    os_name, os_version = sysinfo.os_summary()
    cpu_name, cores, threads = sysinfo.cpu_summary()
    ram_gb = sysinfo.memory_gb()
    gpus = sysinfo.gpus()
    free_gb = sysinfo.disk_free_gb(Path.cwd())

    devices = sysinfo.audio_devices()
    inputs = [d for d in devices if d["type"] == "input"]
    outputs = [d for d in devices if d["type"] == "output"]

    if not devices:
        finding("warn", "音声デバイスを列挙できませんでした")
    elif not inputs:
        finding("warn", "入力デバイスが 0 件")
    elif not outputs:
        finding("warn", "出力デバイスが 0 件")

    # ---------------------------------------------------------------- VC Client
    launcher_name = settings.launcher
    vc_info: dict[str, Any] = {
        "root": str(vc_root) if vc_root_value else "",
        "exists": False,
        "launcher": None,
        "port": port,
        "port_open": False,
        "edition": "",
        "version": "",
        "has_cuda": False,
        "has_onnx": False,
        "model_count": 0,
        "api_ok": None,
        "modules_valid": None,
        "modules_total": None,
    }

    if not vc_root_value:
        finding("warn", "vcClient.root が未設定")
    elif not vc_root.is_dir():
        finding("error", "VC Client フォルダが存在しない")
    else:
        vc_info["exists"] = True
        launcher = vc_root / launcher_name
        if launcher.is_file():
            vc_info["launcher"] = str(launcher)
        else:
            finding("warn", f"{launcher_name} が見つからない")

        edition, version = _detect_edition(vc_root)
        vc_info["edition"] = edition
        vc_info["version"] = version
        vc_info["has_cuda"] = bool(re.search(r"cuda", edition, re.IGNORECASE))
        vc_info["has_onnx"] = bool(re.search(r"onnx|dml", edition, re.IGNORECASE))

        if vc_info["has_cuda"] or vc_info["has_onnx"]:
            pass
        elif re.search(r"std", edition, re.IGNORECASE):
            finding("error", f"RVC 非対応エディション: {edition}")
        elif not edition:
            finding("warn", "エディション判定不可（RVC 対応版か要確認）")

        vc_info["model_count"] = sum(1 for _ in vc_root.rglob("*.pth"))

    vc_info["port_open"] = vcclient.port_open(port, settings.host, timeout=1.0)
    if vc_info["port_open"]:
        try:
            hello = vcclient.api(settings, "/api/hello", timeout=8)
            vc_info["api_ok"] = True
            vc_info["api_message"] = (hello or {}).get("message") if isinstance(hello, dict) else None
        except vcclient.VcApiError:
            vc_info["api_ok"] = False
        valid, total = vcclient.module_status(settings)
        if total:
            vc_info["modules_valid"] = valid
            vc_info["modules_total"] = total
            if valid < total:
                finding("warn", f"必須モジュール未完了 ({valid}/{total})")

    # ---------------------------------------------------------------- 判定
    nvidia = next((g for g in gpus if re.search(r"nvidia", str(g.get("name")), re.IGNORECASE)), None)
    if not nvidia:
        finding("warn", "NVIDIA GPU 未検出")
    vram_gb = None
    if nvidia and nvidia.get("vram_mb"):
        vram_gb = round(float(nvidia["vram_mb"]) / 1024, 1)
        if vram_gb < 4:
            finding("warn", f"VRAM 不足の可能性 ({vram_gb} GB)")
    if ram_gb and ram_gb < 8:
        finding("warn", "メモリが 8 GB 未満")

    return {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": sysinfo.hostname(),
        "platform": {"system": os_name, "version": os_version, "python": platform.python_version()},
        "cpu": {"name": cpu_name, "cores": cores, "threads": threads},
        "memory_gb": ram_gb,
        "free_disk_gb": free_gb,
        "gpus": gpus,
        "audio_devices": devices,
        "vc_client": vc_info,
        "findings": findings,
    }


def render_markdown(data: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append

    add("# 環境確認レポート")
    add("")
    add(f"- 生成日時: {data['generated_at']}")
    add(f"- ホスト名: `{data['host']}`")
    add(f"- Python: {data['platform']['python']}")
    add("")

    add("## OS / ハードウェア")
    add("")
    add("| 項目 | 値 |")
    add("| :--- | :--- |")
    add(f"| OS | {data['platform']['system']} |")
    add(f"| OS バージョン | {data['platform']['version']} |")
    add(f"| CPU | {data['cpu']['name']} |")
    add(f"| コア / スレッド | {data['cpu']['cores']} / {data['cpu']['threads']} |")
    add(f"| メモリ | {data['memory_gb']} GB |")
    add(f"| 空きディスク | {data['free_disk_gb']} GB |")
    add("")

    add("## GPU")
    add("")
    if not data["gpus"]:
        add("GPU を検出できませんでした。")
    else:
        add("| # | 名前 | VRAM | ドライバ | 取得元 |")
        add("| :--- | :--- | :--- | :--- | :--- |")
        for gpu in data["gpus"]:
            vram = f"{float(gpu['vram_mb']) / 1024:.1f} GB" if gpu.get("vram_mb") else "不明"
            index = gpu.get("index")
            add(f"| {index if index is not None else '-'} | {gpu['name']} | {vram} | {gpu.get('driver') or '-'} | {gpu.get('source')} |")
        add("")
        add("> Windows の `Win32_VideoController` の VRAM は 32 bit 制限で 4 GB 超が不正確な場合がある。`nvidia-smi` の値を優先する。")
    add("")

    add("## 音声デバイス")
    add("")
    if not data["audio_devices"]:
        add("デバイスを列挙できませんでした。OS のサウンド設定で確認してください。")
    else:
        for kind in ("input", "output", "unknown"):
            items = [d["name"] for d in data["audio_devices"] if d["type"] == kind]
            if not items:
                continue
            add(f"### {KIND_LABEL[kind]}")
            add("")
            for name in items:
                add(f"- {name}")
            add("")

    add("## VC Client")
    add("")
    vc = data["vc_client"]
    add("| 項目 | 値 |")
    add("| :--- | :--- |")
    add(f"| 配置フォルダ | `{vc['root'] or '（未設定）'}` |")
    add(f"| フォルダ存在 | {vc['exists']} |")
    add(f"| 起動スクリプト | {'あり' if vc['launcher'] else '未検出'} |")
    add(f"| エディション（推定） | {vc['edition'] or '不明'} |")
    add(f"| バージョン（推定） | {vc['version'] or '不明'} |")
    add(f"| ポート {vc['port']} | {'応答あり' if vc['port_open'] else '閉'} |")
    add(f"| 配下の .pth | {vc['model_count']} 件 |")
    if vc.get("modules_total"):
        add(f"| 必須モジュール | {vc['modules_valid']}/{vc['modules_total']} 件有効 |")
    add("")
    add("> エディションとバージョンは推定。実際の配布名は GUI の Server Info で確認し、docs/00-phase0-environment.md に記録する。")
    add("")

    add("## 検出した注意点")
    add("")
    if not data["findings"]:
        add("- なし")
    else:
        for entry in data["findings"]:
            add(f"- [{entry['level']}] {entry['message']}")
    add("")

    add("## 次の作業（Phase 0 完了条件）")
    add("")
    add("- [ ] 機材台帳（マイク / オーディオ IF / ヘッドホン / PC）を確定した")
    add("- [ ] VC Client の採用エディション・バージョンを確定し、ダウンロード元を記録した")
    add("- [ ] 使用する RVC モデル 1 種類を確定し、利用規約を記録した（docs/05-models.md）")
    add("- [ ] 上記を docs/00-phase0-environment.md に転記した")
    add("")
    return "\n".join(lines)


def run(settings: Settings, vc_root: str | None = None, output_dir: str | None = None) -> int:
    data = collect(settings, vc_root)

    ui.step("OS / ハードウェア")
    ui.info(f"OS: {data['platform']['system']} / {data['platform']['version']}")
    ui.info(f"CPU: {data['cpu']['name']} ({data['cpu']['cores']} コア / {data['cpu']['threads']} スレッド)")
    ui.info(f"メモリ: {data['memory_gb']} GB")
    for gpu in data["gpus"]:
        vram = f"{float(gpu['vram_mb']) / 1024:.1f} GB" if gpu.get("vram_mb") else "不明"
        ui.info(f"GPU: {gpu['name']} / VRAM {vram} / driver {gpu.get('driver') or '-'}")
    ui.info(f"空きディスク: {data['free_disk_gb']} GB")

    ui.step("音声入出力デバイス")
    if not data["audio_devices"]:
        ui.warn("音声デバイスを列挙できませんでした")
    else:
        for device in data["audio_devices"]:
            ui.info(f"{KIND_LABEL[device['type']]}: {device['name']}")

    ui.step("VC Client の配置")
    vc = data["vc_client"]
    if not vc["root"]:
        ui.warn("vcClient.root が未設定です（自前エンジンだけなら不要）")
    elif not vc["exists"]:
        ui.err(f"VC Client フォルダが見つかりません: {vc['root']}")
    else:
        ui.ok(f"VC Client フォルダ: {vc['root']}")
        if vc["launcher"]:
            ui.ok("起動スクリプトを確認")
        ui.info(f"エディション: {vc['edition'] or '不明'} / バージョン: {vc['version'] or '不明'}")
        if vc["port_open"]:
            ui.ok(f"ポート {vc['port']} は応答あり（起動中）")
            if vc.get("modules_total"):
                if vc["modules_valid"] == vc["modules_total"]:
                    ui.ok(f"必須モジュール: {vc['modules_total']}/{vc['modules_total']} 件有効")
                else:
                    ui.warn(f"必須モジュール: {vc['modules_valid']}/{vc['modules_total']} 件のみ有効")
        else:
            ui.info(f"ポート {vc['port']} は閉じています（未起動）")

    ui.step("判定")
    if any(re.search(r"nvidia", str(g.get("name")), re.IGNORECASE) for g in data["gpus"]):
        ui.ok("NVIDIA GPU を検出（CUDA が利用可能）")
    else:
        ui.warn("NVIDIA GPU を検出できませんでした（CPU 動作になります）")
    for entry in data["findings"]:
        if entry["level"] == "error":
            ui.err(entry["message"])
        else:
            ui.warn(entry["message"])

    # ---------------------------------------------------------------- 出力
    out_dir = resolve_path(output_dir) if output_dir else settings.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    md_path = out_dir / f"environment-{stamp}.md"
    latest_path = out_dir / "environment-latest.md"
    json_path = out_dir / f"environment-{stamp}.json"

    markdown = render_markdown(data)
    md_path.write_text(markdown, encoding="utf-8")
    latest_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    ui.step("出力")
    ui.ok(f"レポート: {md_path}")
    ui.ok(f"最新版  : {latest_path}")
    ui.ok(f"JSON    : {json_path}")
    ui.note("次は docs/00-phase0-environment.md の台帳に転記してください。")
    return 0
