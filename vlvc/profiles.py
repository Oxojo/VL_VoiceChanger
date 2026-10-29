"""プロファイルの検証・表示・VC Client への適用。

旧 ``Start-VoiceChanger.ps1``（検証と GUI チェックリスト）と
旧 ``Set-VoiceProfile.ps1``（REST での一括反映）をまとめたもの。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from . import sysinfo, ui, vcclient
from .config import Settings, dot_get, load_profile, profile_id, profile_name, resolve_path

#: REST では変更できず、GUI で設定する項目
MANUAL_KEYS = (
    ("convert.pitch", "TUNE（pitch_shift）"),
    ("convert.indexRatio", "INDEX（index_ratio）"),
    ("convert.f0Detector", "F0 Det"),
    ("convert.chunk", "CHUNK"),
)


def _device_names() -> list[str]:
    return [d["name"] for d in sysinfo.audio_devices()]


def match_device(known: list[str], query: str) -> tuple[str, str | None]:
    """(状態, 候補) を返す。状態は exact / partial / missing。"""
    if query in known:
        return "exact", query
    for name in known:
        if query and query.lower() in name.lower():
            return "partial", name
    return "missing", None


def validate(settings: Settings, name_or_path: str) -> dict[str, Any]:
    path, profile = load_profile(name_or_path, settings)
    errors: list[str] = []
    warnings: list[str] = []

    # ---------------------------------------------------------------- VC Client
    vc_root = settings.vc_root
    launcher = vc_root / settings.launcher
    if not str(vc_root) or str(vc_root) == ".":
        errors.append("vcClient.root が未設定です（config/settings.json）")
    elif not vc_root.is_dir():
        errors.append(f"VC Client フォルダが存在しません: {vc_root}")
    elif not launcher.is_file():
        errors.append(f"起動スクリプトが見つかりません: {launcher}")

    # ---------------------------------------------------------------- モデル
    model_pth = str(dot_get(profile, "model.pth", "") or "")
    model_index = str(dot_get(profile, "model.index", "") or "")
    model_status: list[tuple[str, str]] = []
    if not model_pth:
        warnings.append("model.pth が未設定です（モデル未選定）")
    else:
        pth = resolve_path(model_pth)
        if pth.is_file():
            model_status.append(("モデル", model_pth))
        else:
            warnings.append(f"モデルファイルが見つかりません: {model_pth}（GUI のスロット登録でも可）")
    if model_index:
        idx = resolve_path(model_index)
        if idx.is_file():
            model_status.append(("index", model_index))
        else:
            warnings.append(f"index ファイルが見つかりません: {model_index}")

    # ---------------------------------------------------------------- デバイス
    known = _device_names()
    devices: list[dict[str, Any]] = []
    for label, key in (
        ("入力", "audio.inputDevice"),
        ("出力", "audio.outputDevice"),
        ("モニター", "audio.monitorDevice"),
    ):
        value = str(dot_get(profile, key, "") or "")
        if not value:
            continue
        state, candidate = match_device(known, value)
        if state == "exact":
            devices.append({"label": label, "value": value, "state": "ok", "candidate": None})
        elif state == "partial":
            warnings.append(f"{label}デバイス名が完全一致しません: '{value}' → 候補 '{candidate}'")
            devices.append({"label": label, "value": value, "state": "partial", "candidate": candidate})
        else:
            warnings.append(f"{label}デバイスが見つかりません: '{value}'")
            devices.append({"label": label, "value": value, "state": "missing", "candidate": None})

    checklist = {
        "mode": str(dot_get(profile, "audio.mode", "server")),
        "pitch": dot_get(profile, "convert.pitch", 0),
        "index_ratio": dot_get(profile, "convert.indexRatio", 0),
        "f0": str(dot_get(profile, "convert.f0Detector", "rmvpe")),
        "chunk": dot_get(profile, "convert.chunk", 128),
        "extra": dot_get(profile, "convert.extra", 8192),
        "noise_gate": dot_get(profile, "convert.noiseGateDb", -60),
        "gain_in": dot_get(profile, "convert.gainIn", 0),
        "gain_out": dot_get(profile, "convert.gainOut", 0),
        "volume": dot_get(profile, "convert.volume", 1.0),
        "gpu": dot_get(profile, "gpu.deviceId", 0),
        "input_device": dot_get(profile, "audio.inputDevice", "（未設定）"),
        "output_device": dot_get(profile, "audio.outputDevice", "（未設定）"),
        "monitor_device": dot_get(profile, "audio.monitorDevice", ""),
        "model_pth": model_pth or "（未設定）",
        "model_index": model_index or "（なし）",
    }

    return {
        "path": path,
        "profile": profile,
        "id": profile_id(profile, path),
        "name": profile_name(profile, path),
        "errors": errors,
        "warnings": warnings,
        "devices": devices,
        "model_status": model_status,
        "checklist": checklist,
        "vc_root": vc_root,
        "launcher": launcher,
    }


def print_checklist(result: dict[str, Any]) -> None:
    c = result["checklist"]
    ui.step("GUI で設定する値")
    ui.note("  --- モデル ---", "gray")
    ui.item("モデル選択エリア", result["name"])
    ui.item("モデルファイル (RVC)", c["model_pth"])
    ui.item("index", c["model_index"])
    ui.note("  --- 音声入出力（詳細設定エリア） ---", "gray")
    suffix = "  ※VC Client がデバイスを直接使用" if c["mode"] == "server" else "  ※ブラウザ経由"
    ui.item("AUDIO", f"{c['mode']}{suffix}")
    ui.item("input", c["input_device"])
    ui.item("output", c["output_device"])
    if c["monitor_device"]:
        ui.item("monitor", c["monitor_device"])
    ui.note("  --- 変換設定 ---", "gray")
    ui.item("TUNE（ピッチ）", c["pitch"])
    ui.item("INDEX（比率）", c["index_ratio"])
    ui.item("F0 Det", c["f0"])
    ui.item("CHUNK", c["chunk"])
    ui.item("EXTRA", c["extra"])
    ui.item("S.Thresh（ノイズゲート）", c["noise_gate"])
    ui.item("GAIN in / out", f"{c['gain_in']} / {c['gain_out']}")
    ui.item("vol", c["volume"])
    ui.item("GPU", f"gpu{c['gpu']}")
    ui.note("  --- 起動後の確認 ---", "gray")
    for line in (
        "[start] を押し、マイクに向かって発話する",
        "変換後の声が output デバイスから聞こえることを確認する",
        "buf > res になっているか確認する（buf が短いと音切れしやすい）",
        "二重に聞こえる場合はダイレクトモニターを切る（docs/02-operation.md）",
        "調整した値は config/profiles の JSON に反映する",
    ):
        print(f"       {line}")


# ------------------------------------------------------------------ 適用
def _resolve_device_index(devices: list[dict[str, Any]], query: str, label: str) -> int:
    """部分一致でデバイス index を決める。見つからなければ -1（変更しない）。"""
    if not query:
        return -1
    hits = [d for d in devices if query.lower() in str(d.get("name", "")).lower()]
    if not hits:
        ui.warn(f"{label} に一致するデバイスがありません: '{query}'（変更しません）")
        return -1
    if len(hits) > 1:
        ui.warn(f"{label} の候補が {len(hits)} 件あります。先頭を使用します:")
        for device in hits[:5]:
            ui.bullet(f"index {device.get('index'):>3} [{device.get('host_api', '-')}] {device.get('name')}", "yellow")
    try:
        return int(hits[0]["index"])
    except (KeyError, TypeError, ValueError):
        ui.warn(f"{label} の index を読めませんでした（変更しません）")
        return -1


def build_changes(
    settings: Settings,
    profile: dict[str, Any],
    current: dict[str, Any],
    slots: list[dict[str, Any]],
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """適用する差分と、GUI で設定する必要がある項目を返す。"""
    changes: dict[str, Any] = {}

    model_pth = str(dot_get(profile, "model.pth", "") or "")
    if model_pth:
        model_name = Path(model_pth).name
        matches = [s for s in slots if str(s.get("model_file", "")) == model_name]
        if not matches:
            ui.warn(f"{model_name} は未登録です。`vlvc register-model` で登録してください（スロットは変更しません）")
        else:
            if len(matches) > 1:
                ui.warn(f"{model_name} が複数スロットに登録されています。先頭を使用します")
            changes["current_slot_index"] = int(matches[0].get("slot_index", 0))

    mode = str(dot_get(profile, "audio.mode", "") or "")
    if mode in {"server", "client"}:
        changes["voice_changer_input_mode"] = mode

    in_idx = _resolve_device_index(inputs, str(dot_get(profile, "audio.inputDevice", "") or ""), "入力")
    out_idx = _resolve_device_index(outputs, str(dot_get(profile, "audio.outputDevice", "") or ""), "出力")
    if in_idx >= 0:
        changes["audio_input_device_index"] = in_idx
    if out_idx >= 0:
        changes["audio_output_device_index"] = out_idx

    gpu_id = dot_get(profile, "gpu.deviceId", None)
    if gpu_id is not None:
        changes["gpu_device_id_int"] = int(gpu_id)

    noise_gate = dot_get(profile, "convert.noiseGateDb", None)
    if noise_gate is not None:
        changes["noise_gate"] = float(noise_gate)

    extra = dot_get(profile, "convert.extra", None)
    if extra is not None and float(extra) > 0:
        changes["extra_frame_sec"] = round(float(extra) / 48000.0, 4)

    manual: list[str] = []
    for key, label in MANUAL_KEYS:
        value = dot_get(profile, key, None)
        if value is not None:
            manual.append(f"{label} = {value}")
    return changes, manual


def apply(settings: Settings, name_or_path: str, dry_run: bool = False) -> int:
    path, profile = load_profile(name_or_path, settings)
    ui.header("VL_VoiceChanger / プロファイル適用")

    ui.step("1. プロファイルの読み込み")
    ui.ok(f"{profile_name(profile, path)} ({profile_id(profile, path)})")
    ui.info(str(path))

    if not vcclient.port_open(settings.port, settings.host, timeout=1.0):
        ui.err(f"ポート {settings.port} が開いていません。先に `vlvc launch` を実行してください。")
        return 1

    ui.step("2. 現在の設定を取得")
    try:
        current = vcclient.api(settings, "/api/configuration-manager/configuration", timeout=20)
        slots = vcclient.api(settings, "/api/slot-manager/slots", timeout=30) or []
        inputs = vcclient.api(settings, "/api/audio-device-manager/input_devices", timeout=20) or []
        outputs = vcclient.api(settings, "/api/audio-device-manager/output_devices", timeout=20) or []
    except vcclient.VcApiError as exc:
        ui.err(f"設定を取得できません: {exc}")
        if exc.body:
            print(exc.body[:1000])
        return 1

    current = current if isinstance(current, dict) else {}
    ui.info(
        f"current_slot_index = {current.get('current_slot_index')} / "
        f"mode = {current.get('voice_changer_input_mode')}"
    )

    ui.step("3. 適用内容")
    changes, manual = build_changes(settings, profile, current, slots, inputs, outputs)
    if not changes:
        ui.warn("適用できる項目がありません（audio.inputDevice / outputDevice を確認してください）")
        return 0

    for key, value in changes.items():
        before = current.get(key, "-")
        ui.item(key, f"{before} -> {value}", width=30)

    if manual:
        ui.warn("次は REST API では変更できません（GUI で設定してください）:")
        for line in manual:
            ui.bullet(line, "yellow")

    if dry_run:
        ui.info("--dry-run のため適用していません")
        return 0

    ui.step("4. 適用")
    payload = dict(current)
    payload.update(changes)
    try:
        vcclient.api(settings, "/api/configuration-manager/configuration", method="PUT", body=payload, timeout=120)
    except vcclient.VcApiError as exc:
        ui.err(f"適用に失敗しました: {exc}")
        if exc.body:
            print(exc.body[:2000])
        return 1

    time.sleep(0.5)
    try:
        after = vcclient.api(settings, "/api/configuration-manager/configuration", timeout=20) or {}
    except vcclient.VcApiError as exc:
        ui.err(f"適用結果を取得できません: {exc}")
        return 1

    ui.step("5. 適用結果")
    ok_all = True
    for key, value in changes.items():
        if after.get(key) == value:
            ui.ok(f"{key} = {after.get(key)}")
        else:
            ok_all = False
            ui.err(f"{key} = {after.get(key)}（期待 {value}）")

    ui.note("GUI で [start] を押すと変換が始まります（REST では開始できません）。")
    return 0 if ok_all else 1
