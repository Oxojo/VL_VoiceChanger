"""変換前後の音声を数値で比較する（旧 ``scripts/measure_quality.py``）。

F0（RMVPE）・有声率・エネルギー包絡・調波性・クリップの有無を見る。
分割変換の結果を渡すときは ``--hop 0.5`` を付けると、継ぎ目のサンプル跳びも表示する。
Web UI と同じ指標（``engine/metrics.py``）を使う。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from engine.audio import SAMPLE_RATE, load_mono, resample
from engine.config import resolve_asset
from engine.features import RMVPE
from engine.metrics import boundary_jumps, compare


def measure_text(
    input_path: str | Path,
    output_path: str | Path,
    hop: float | None = None,
    rmvpe_path: str | Path | None = None,
) -> str:
    """比較結果を Markdown（表）で返す。"""
    rmvpe = RMVPE(rmvpe_path or resolve_asset("rmvpe"))

    x_in = load_mono(str(input_path), SAMPLE_RATE)
    f_in = rmvpe.infer(x_in)

    out, sr_out = sf.read(str(output_path), dtype="float32", always_2d=False)
    if out.ndim > 1:
        out = out.mean(axis=1)
    x_out = resample(out, sr_out, SAMPLE_RATE)
    f_out = rmvpe.infer(x_out)

    metrics = compare(x_in, x_out, f_in, f_out)
    m = metrics

    lines = [
        f"入力 : {Path(input_path).name}  {x_in.shape[0] / SAMPLE_RATE:.2f} 秒 (16 kHz 換算)",
        f"出力 : {Path(output_path).name}  {out.shape[0] / sr_out:.2f} 秒 / {sr_out} Hz",
        "",
        "| 項目 | 入力 | 出力 |",
        "| --- | --- | --- |",
        f"| F0 中央値 | {m['f0_median_in']:.1f} Hz | {m['f0_median_out']:.1f} Hz |",
        f"| 有声率 | {m['voiced_ratio_in']:.1f} % | {m['voiced_ratio_out']:.1f} % |",
        f"| RMS | {m['rms_in']:.4f} | {m['rms_out']:.4f} |",
        f"| ピーク | {np.abs(x_in).max():.3f} | {m['peak_out']:.3f} |",
        f"| 調波性 | {m['harmonicity_in']:.3f} | {m['harmonicity_out']:.3f} |",
        "",
        f"F0 追従（log F0 の相関）    : {m['f0_correlation']:.3f}",
        f"エネルギー包絡の相関        : {m['envelope_correlation']:.3f}",
        f"クリップ（|値| >= 1.0）     : {'あり' if m['clipped'] else 'なし'}",
    ]

    if hop:
        mean_jump, max_jump = boundary_jumps(out, sr_out, hop)
        lines.append(f"継ぎ目の跳び（{hop}s ごと）  : 平均 {mean_jump:.4f} / 最大 {max_jump:.4f}")
        lines.append(f"全体の平均 |Δ|              : {float(np.abs(np.diff(out)).mean()):.4f}")
    return "\n".join(lines)


def run(
    input_path: str,
    output_path: str,
    hop: float | None = None,
    rmvpe_path: str | None = None,
) -> int:
    print(measure_text(input_path, output_path, hop, rmvpe_path))
    return 0
