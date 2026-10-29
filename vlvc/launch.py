"""VC Client の起動補助（旧 ``Start-VoiceChanger.ps1``）。

プロファイルの検証 → モデル／デバイスの突き合わせ → 起動 → GUI 設定項目の表示。
VC Client v2 系は変換設定を外部から流し込む手段が限られるため、
実際の設定は GUI で行い、REST で変えられる項目は ``vlvc profile apply`` で反映する。
"""

from __future__ import annotations

from . import profiles, ui, vcclient
from .config import Settings


def run(settings: Settings, name_or_path: str = "character-a", dry_run: bool = False, timeout: int | None = None) -> int:
    try:
        result = profiles.validate(settings, name_or_path)
    except Exception as exc:  # noqa: BLE001 - 設定不備はメッセージで返す
        ui.err(str(exc))
        ui.info("config/settings.json とプロファイルの JSON を確認してください。")
        return 1

    ui.header("VL_VoiceChanger / 起動補助")
    ui.info(f"プロファイル: {result['name']} ({result['id']})")
    ui.info(f"ファイル    : {result['path']}")

    ui.step("1. 構成の検証")
    if str(settings.vc_root) != "." and settings.vc_root.is_dir():
        ui.ok(f"VC Client: {settings.vc_root}")
    if result["launcher"].is_file():
        ui.ok(f"起動スクリプト: {settings.launcher}")
    for label, value in result["model_status"]:
        ui.ok(f"{label}: {value}")
    for device in result["devices"]:
        if device["state"] == "ok":
            ui.ok(f"{device['label']}デバイス: {device['value']}")

    for warning in result["warnings"]:
        ui.warn(warning)
    for error in result["errors"]:
        ui.err(error)
    if result["errors"]:
        ui.note(" 起動条件を満たしていません。上記を解消してください。", "red")
        return 1

    ui.step("2. VC Client 起動")
    port = settings.port
    already_open = vcclient.port_open(port, settings.host, timeout=0.8)
    existing = vcclient.find_processes(settings.vc_root, "")

    if already_open:
        ui.ok(f"ポート {port} は既に応答しています（起動済みと判断し、二重起動しません）")
    elif dry_run:
        ui.info("--dry-run のため起動しません")
        ui.info(f"実行時のコマンド: \"{settings.vc_root / settings.launcher}\"（作業ディレクトリ: {settings.vc_root}）")
    elif existing:
        pids = ", ".join(str(p.pid) for p in existing)
        ui.warn(f"VC Client のプロセスが既に動作しています（PID: {pids}）。二重起動せず、GUI の応答を待ちます。")
        if not vcclient.wait_gui(settings, timeout or settings.startup_timeout, settings.vc_root / "vcclient.log"):
            ui.err(f"時間内にポート {port} が開きませんでした（プロセスは動作中）")
            ui.info("初回起動は必須モジュール（約 3.3 GB）の取得で数十分かかります。")
            ui.info("プロセスを終了せず、時間をおいて再度実行してください（--timeout 7200 も可）。")
            return 1
        ui.ok(f"GUI が応答しました: http://localhost:{port}/")
    else:
        ui.info(f"起動: {settings.launcher}")
        try:
            vcclient.launch(settings)
        except (OSError, FileNotFoundError) as exc:
            ui.err(f"起動できませんでした: {exc}")
            return 1
        wait_seconds = timeout or settings.startup_timeout
        ui.info(f"GUI の応答を待機中（最大 {wait_seconds} 秒）")
        ui.info("初回起動は必須モジュール（合計約 3.3 GB）の取得で数十分かかることがあります。")
        if not vcclient.wait_gui(settings, wait_seconds, settings.vc_root / "vcclient.log"):
            ui.err(f"時間内にポート {port} が開きませんでした")
            ui.info("docs/03-troubleshooting.md の「GUI が開かない」を確認してください。")
            return 1
        ui.ok(f"GUI が応答しました: http://localhost:{port}/")

    profiles.print_checklist(result)

    if dry_run:
        ui.note(" [--dry-run] ここまで。実際に起動するときは --dry-run を外してください。", "yellow")
    else:
        ui.note(f" GUI: http://localhost:{port}/", "green")
    return 0
