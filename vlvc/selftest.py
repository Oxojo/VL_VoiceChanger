"""自前エンジンの動作確認（旧 ``Test-OwnEngine.ps1``）。

変換 → F0/音量の解析 → 品質の数値比較 → 分割変換の速度計測、を順に実行し、
``reports/own-engine-latest.md`` にまとめる。どの OS でも動く。
"""

from __future__ import annotations

import datetime as dt
import platform
import subprocess
from pathlib import Path

from . import pyproc, quality, ui, vcclient
from .config import REPO_ROOT, Settings, resolve_path


def _git(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (proc.stdout or "").strip()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _python_version(python: str | None) -> str:
    exe = pyproc.python_exe(python)
    proc = pyproc.run(
        [exe, "-c", "import platform, torch; print(platform.python_version(), 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"]
    )
    return (proc.stdout or proc.stderr or "").strip()


def run(
    settings: Settings,
    model: str = "models/tsukuyomi/tsukuyomi-normal1.pth",
    input_path: str = "logs/verify/tts-input-ja.wav",
    python: str | None = None,
    device: str = "cpu",
    pitch: int = 0,
    hop: float = 0.5,
    context: float = 0.15,
    crossfade: float = 0.05,
    report: str | None = None,
) -> int:
    model_path = resolve_path(model)
    source = resolve_path(input_path)
    report_path = resolve_path(report) if report else settings.reports_dir / "own-engine-latest.md"

    ui.header("VL_VoiceChanger / 自前エンジン動作確認")
    if not model_path.is_file():
        ui.err(f"モデルが見つかりません: {model_path}")
        ui.info("docs/05-models.md を参照して models/ に配置してください。")
        return 1
    if not source.is_file():
        ui.err(f"入力 WAV が見つかりません: {source}")
        ui.info("先に検証用の音声を logs/verify/ に用意してください。")
        return 1

    verify_dir = settings.logs_dir / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    out_wav = verify_dir / f"own-{stem}-{device}.wav"
    out_json = verify_dir / f"own-{stem}-{device}.json"
    chunk_wav = verify_dir / f"own-{stem}-{device}-chunk.wav"

    common = ["--model", str(model_path), "--input", str(source)]

    ui.step("1/4 変換（一括）")
    code, convert_log = pyproc.capture_engine(
        ["convert", *common, "--output", str(out_wav), "--json", str(out_json), "--pitch", str(pitch), "--device", device],
        python=python,
    )
    print(convert_log)
    if code != 0:
        ui.err(f"変換に失敗しました（終了コード {code}）")
        return 1

    ui.step("2/4 解析（入力 / 出力）")
    _, analyze_in = pyproc.capture_engine(["analyze", "--input", str(source)], python=python)
    print(analyze_in)
    _, analyze_out = pyproc.capture_engine(["analyze", "--input", str(out_wav)], python=python)
    print(analyze_out)

    ui.step("3/4 品質の数値比較（一括変換）")
    try:
        quality_text = quality.measure_text(source, out_wav)
    except Exception as exc:  # noqa: BLE001 - レポート作成は続行する
        quality_text = f"（品質比較に失敗: {exc}）"
    print(quality_text)

    ui.step("4/4 分割変換の速度（ライブ成立性）")
    code, bench_log = pyproc.capture_engine(
        [
            "bench",
            *common,
            "--output",
            str(chunk_wav),
            "--hop",
            str(hop),
            "--context",
            str(context),
            "--crossfade",
            str(crossfade),
            "--pitch",
            str(pitch),
            "--device",
            device,
        ],
        python=python,
    )
    print(bench_log)
    if code != 0:
        ui.err(f"分割変換の計測に失敗しました（終了コード {code}）")
        return 1

    try:
        chunk_quality = quality.measure_text(source, chunk_wav, hop=hop)
    except Exception as exc:  # noqa: BLE001
        chunk_quality = f"（品質比較に失敗: {exc}）"
    print(chunk_quality)

    git_branch = _git("branch", "--show-current")
    git_head = _git("rev-parse", "--short", "HEAD")
    py_version = _python_version(python)

    lines = [
        "# 自前エンジン 動作確認レポート",
        "",
        f"生成日時: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"ブランチ: {git_branch}（{git_head}）",
        f"実行環境: {py_version}",
        f"OS: {platform.platform()}",
        f"モデル: {_relative(model_path)}",
        f"入力: {_relative(source)}",
        f"デバイス: {device} / ピッチシフト: {pitch} 半音 / 分割: hop {hop} 秒・context {context} 秒・crossfade {crossfade} 秒",
        "",
        "## 1. 変換",
        "",
        "```text",
        convert_log.strip(),
        "```",
        "",
        "## 2. 入力の解析",
        "",
        "```text",
        analyze_in.strip(),
        "```",
        "",
        "## 3. 出力の解析",
        "",
        "```text",
        analyze_out.strip(),
        "```",
        "",
        "## 4. 品質の数値比較（一括変換）",
        "",
        quality_text.strip(),
        "",
        "## 5. 分割変換（ライブ成立性）",
        "",
        "```text",
        bench_log.strip(),
        "```",
        "",
        "## 6. 分割変換の品質",
        "",
        chunk_quality.strip(),
        "",
        "## 7. 出力ファイル",
        "",
        f"- 一括変換: `{_relative(out_wav)}`（Git 管理外）",
        f"- 分割変換: `{_relative(chunk_wav)}`（Git 管理外）",
        f"- 計測 JSON: `{_relative(out_json)}`（Git 管理外）",
        "",
        "## 8. 判定",
        "",
        "| 項目 | 判定 |",
        "| :--- | :--- |",
        "| 変換が最後まで通る | OK（上記ログ参照） |",
        "| 出力が無音でない | 出力の RMS が 0 でないこと |",
        "| ピッチが入力に追従 | F0 追従（log F0 の相関）が 0.9 以上 |",
        "| 継ぎ目が不連続でない | 継ぎ目の跳びが全体平均と同程度 |",
        "| ライブ成立 | 分割変換の「最大遅延 < hop」 |",
        "",
        "聴感（声質）の最終判定は人による試聴が必要（`docs/04-verification.md`）。",
        "",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    ui.step("出力")
    ui.ok(f"レポート: {report_path}")
    ui.info("聴感の最終判定は docs/04-verification.md の評価シートで行ってください。")
    return 0
