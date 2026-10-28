"""変換前後の音声を数値で比較する（聴取前の自動チェック用）。

    python scripts/measure_quality.py 入力.wav 出力.wav

F0（RMVPE）・有声率・エネルギー包絡・調波性・クリップの有無を見る。
分割変換の結果を渡すときは ``--hop 0.5`` を付けると、継ぎ目のサンプル跳びも表示する。
Web UI と同じ指標（engine/metrics.py）を使う。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.audio import SAMPLE_RATE, load_mono, resample  # noqa: E402
from engine.config import resolve_asset  # noqa: E402
from engine.features import RMVPE  # noqa: E402
from engine.metrics import boundary_jumps, compare  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="変換前後の音声を比較する")
    parser.add_argument("input", help="入力 WAV")
    parser.add_argument("output", help="出力 WAV")
    parser.add_argument("--hop", type=float, default=None, help="分割変換の hop（秒）")
    parser.add_argument("--rmvpe", default=None, help="RMVPE の ONNX パス")
    args = parser.parse_args()

    rmvpe = RMVPE(args.rmvpe or resolve_asset("rmvpe"))

    x_in = load_mono(args.input, SAMPLE_RATE)
    f_in = rmvpe.infer(x_in)

    out, sr_out = sf.read(args.output, dtype="float32", always_2d=False)
    if out.ndim > 1:
        out = out.mean(axis=1)
    x_out = resample(out, sr_out, SAMPLE_RATE)
    f_out = rmvpe.infer(x_out)

    m = compare(x_in, x_out, f_in, f_out)

    print(f"入力 : {Path(args.input).name}  {x_in.shape[0] / SAMPLE_RATE:.2f} 秒 (16 kHz 換算)")
    print(f"出力 : {Path(args.output).name}  {out.shape[0] / sr_out:.2f} 秒 / {sr_out} Hz")
    print()
    print("| 項目 | 入力 | 出力 |")
    print("| --- | --- | --- |")
    print(f"| F0 中央値 | {m['f0_median_in']:.1f} Hz | {m['f0_median_out']:.1f} Hz |")
    print(f"| 有声率 | {m['voiced_ratio_in']:.1f} % | {m['voiced_ratio_out']:.1f} % |")
    print(f"| RMS | {m['rms_in']:.4f} | {m['rms_out']:.4f} |")
    print(f"| ピーク | {np.abs(x_in).max():.3f} | {m['peak_out']:.3f} |")
    print(f"| 調波性 | {m['harmonicity_in']:.3f} | {m['harmonicity_out']:.3f} |")
    print()
    print(f"F0 追従（log F0 の相関）    : {m['f0_correlation']:.3f}")
    print(f"エネルギー包絡の相関        : {m['envelope_correlation']:.3f}")
    print(f"クリップ（|値| >= 1.0）     : {'あり' if m['clipped'] else 'なし'}")

    if args.hop:
        mean_jump, max_jump = boundary_jumps(out, sr_out, args.hop)
        print(f"継ぎ目の跳び（{args.hop}s ごと）  : 平均 {mean_jump:.4f} / 最大 {max_jump:.4f}")
        print(f"全体の平均 |Δ|              : {float(np.abs(np.diff(out)).mean()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
