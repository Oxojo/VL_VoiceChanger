"""VC Client へのモデル登録（旧 ``Register-Model.ps1``）。

GUI の「モデルをアップロード」と同じ処理を REST API で行う。
実体は ``vcClient.root/upload_dir`` にコピーされ、サーバーがスロットへ移して
``params.json`` を生成する。RVC / ONNX の判別は拡張子で行われるため、
一時ファイル名にも拡張子を残す。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from . import ui, vcclient
from .config import Settings, format_size, resolve_path

VALID_EMBEDDERS = (
    "",
    "hubert_base_l9fp",
    "hubert_base_l12",
    "contentvec",
    "hubert_base_japanese_l9fp",
    "hubert_base_japanese_l12",
    "whisper",
    "applio_japanese_hubert_base_l12",
    "applio_chinese_hubert_base_l12",
    "applio_korean_hubert_base_l12",
)

VALID_EXTENSIONS = {".pth", ".onnx"}


def _free_slot(slots: list[dict]) -> int:
    free = [s for s in slots if not str(s.get("voice_changer_type") or "")]
    if free:
        return int(min(int(s.get("slot_index", 0)) for s in free))
    indices = [int(s.get("slot_index", 0)) for s in slots]
    return (max(indices) + 1) if indices else 0


def register(
    settings: Settings,
    model_path: str,
    name: str = "",
    slot_index: int = -1,
    index_file: str = "",
    terms_url: str = "",
    embedder: str = "",
    force: bool = False,
) -> int:
    ui.header("VL_VoiceChanger / モデル登録")
    embedder = embedder or ""
    if embedder not in VALID_EMBEDDERS:
        ui.err(f"embedder の値が不正です: {embedder}")
        ui.info("指定できる値: " + ", ".join(v for v in VALID_EMBEDDERS if v))
        return 2

    ui.step("1. VC Client の稼働確認")
    if not vcclient.port_open(settings.port, settings.host, timeout=1.0):
        ui.err(f"ポート {settings.port} が開いていません。先に `vlvc launch` を実行してください。")
        return 1
    try:
        hello = vcclient.api(settings, "/api/hello", timeout=10)
        message = (hello or {}).get("message") if isinstance(hello, dict) else None
        ui.ok(f"API 応答: {message or 'OK'}")
    except vcclient.VcApiError as exc:
        ui.err(f"API が応答しません: {exc}")
        return 1

    vc_root = settings.vc_root
    if not str(vc_root) or str(vc_root) == ".":
        ui.err("vcClient.root が未設定です（config/settings.json）。")
        return 1
    if not vc_root.is_dir():
        ui.err(f"vcClient.root が見つかりません: {vc_root}")
        return 1

    ui.step("2. モデルファイルの確認")
    source = resolve_path(model_path)
    if not source.is_file():
        ui.err(f"モデルが見つかりません: {source}")
        return 1
    extension = source.suffix.lower()
    if extension not in VALID_EXTENSIONS:
        ui.err(f"拡張子が .pth / .onnx ではありません: {source.name}")
        ui.info("VC Client は拡張子でモデル種別を判定するため、拡張子は必須です。")
        return 1
    model_name = name or source.stem
    ui.ok(f"{source.name} ({format_size(source.stat().st_size)})")

    ui.step("3. 登録済みスロットの確認")
    try:
        slots = vcclient.api(settings, "/api/slot-manager/slots", timeout=30) or []
    except vcclient.VcApiError as exc:
        ui.err(f"スロットを取得できません: {exc}")
        return 1
    slots = slots if isinstance(slots, list) else []
    ui.info(f"スロット数: {len(slots)}")

    same = [s for s in slots if str(s.get("model_file") or "") == source.name]
    if same and not force:
        ui.warn("同じモデルが既に登録されています（--force で二重登録できます）")
        for slot in same:
            ui.bullet(
                f"slot {slot.get('slot_index')}: {slot.get('name')} / "
                f"{slot.get('inferencer_type')} / {slot.get('sample_rate')} Hz",
                "yellow",
            )
        return 0

    if slot_index < 0:
        slot_index = _free_slot(slots)
        ui.info(f"空きスロットを自動選択: {slot_index}")
    else:
        ui.info(f"登録先スロット: {slot_index}（指定）")

    ui.step("4. モデルの取り込み")
    upload_dir = vc_root / "upload_dir"
    upload_dir.mkdir(parents=True, exist_ok=True)
    staged = upload_dir / source.name
    shutil.copy2(source, staged)
    ui.info(f"upload_dir へコピー: {source.name}")

    payload = {
        "voice_changer_type": "RVC",
        "name": model_name,
        "terms_of_use_url": terms_url,
        "slot_index": slot_index,
        "model_file": source.name,  # サーバーが upload_dir を前置する
        "index_file": index_file or None,
        "icon_file": None,
        "embedder": embedder or None,
    }

    import time

    started = time.monotonic()
    try:
        vcclient.api(settings, "/api/slot-manager/slots", method="POST", body=payload, timeout=600)
    except vcclient.VcApiError as exc:
        ui.err(f"取り込みに失敗しました: {exc}")
        if exc.body:
            print(exc.body[:2000])
        return 1
    ui.ok(f"取り込み完了（{time.monotonic() - started:.1f} 秒）")

    ui.step("5. 登録結果")
    try:
        after = vcclient.api(settings, "/api/slot-manager/slots", timeout=30) or []
    except vcclient.VcApiError as exc:
        ui.err(f"登録結果を取得できません: {exc}")
        return 1
    registered = [
        s
        for s in after
        if int(s.get("slot_index", -1)) == slot_index and str(s.get("model_file") or "") == source.name
    ]
    if not registered:
        ui.err("登録後のスロットが見つかりません。vcclient.log を確認してください。")
        return 1

    slot = registered[0]
    ui.item("slot_index", slot.get("slot_index"))
    ui.item("name", slot.get("name"))
    ui.item("model_file", slot.get("model_file"))
    ui.item("inferencer", f"{slot.get('inferencer_type')} / {slot.get('sample_rate')} Hz / f0={slot.get('is_f0')}")
    ui.item("embedder", slot.get("embedder"))
    ui.item("pitch_estimator", slot.get("pitch_estimator"))
    ui.ok(f"スロット {slot_index} に登録されました")
    ui.note(f"GUI（http://localhost:{settings.port}/）でこのモデルを選択し、[start] で変換を開始してください。")
    ui.info("REST API では変換スレッドを開始できません（GUI の [start] が必要。docs/02-operation.md）。")
    return 0
