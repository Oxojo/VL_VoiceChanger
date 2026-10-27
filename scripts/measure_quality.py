"""変換前後の音声を数値で比較する（聴取前の自動チェック用）。

    python scripts/measure_quality.py 入力.wav 出力.wav

F0（RMVPE）・有声率・エネルギー包絡・スペクトル平坦度・調波性・クリップの有無を見る。
分割変換の結果を渡すときは ``--hop 0.5`` を付けると、継ぎ目のサンプル跳びも表示する。
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


def envelope(x: np.ndarray, win: int = 1600) -> np.ndarray:
    m = x.shape[0] // win
    if m == 0:
        return np.zeros(0)
    return np.sqrt((x[: m * win].reshape(m, win).astype(np.float64) ** 2).mean(axis=1))


def harmonicity(x: np.ndarray, sr: int) -> float:
    """自己相関のピーク（高いほど周期的 = 有声的）。"""
    if x.shape[0] < sr // 10:
        return float("nan")
    seg = x[: sr // 2].astype(np.float64)
    seg = seg - seg.mean()
    corr = np.correlate(seg, seg, mode="full")[seg.size - 1 :]
    lo, hi = int(sr / 500), int(sr / 60)
    if corr[0] <= 0 or hi >= corr.size:
        return float("nan")
    return float(corr[lo:hi].max() / corr[0])


def boundary_jumps(x: np.ndarray, sr: int, hop_seconds: float) -> tuple[float, float]:
    step = int(round(hop_seconds * sr))
    idx = np.arange(step, x.shape[0], step) - 1
    if idx.size == 0:
        return float("nan"), float("nan")
    jump = np.abs(x[idx] - x[idx + 1])
    return float(jump.mean()), float(jump.max())


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

    n = min(f_in.shape[0], f_out.shape[0])
    both = (f_in[:n] > 0) & (f_out[:n] > 0)
    f0_corr = float(np.corrcoef(np.log(f_in[:n][both]), np.log(f_out[:n][both]))[0, 1]) if both.sum() > 10 else float("nan")

    env_in, env_out = envelope(x_in), envelope(x_out)
    m = min(env_in.shape[0], env_out.shape[0])
    env_corr = float(np.corrcoef(env_in[:m], env_out[:m])[0, 1]) if m > 1 else float("nan")

    def median(f0: np.ndarray) -> float:
        voiced = f0[f0 > 0]
        return float(np.median(voiced)) if voiced.size else float("nan")

    def voiced_ratio(f0: np.ndarray) -> float:
        return float((f0 > 0).mean() * 100)

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))

    print(f"入力 : {Path(args.input).name}  {x_in.shape[0] / SAMPLE_RATE:.2f} 秒 (16 kHz 換算)")
    print(f"出力 : {Path(args.output).name}  {out.shape[0] / sr_out:.2f} 秒 / {sr_out} Hz")
    print()
    print("| 項目 | 入力 | 出力 |")
    print("| --- | --- | --- |")
    print(f"| F0 中央値 | {median(f_in):.1f} Hz | {median(f_out):.1f} Hz |")
    print(f"| 有声率 | {voiced_ratio(f_in):.1f} % | {voiced_ratio(f_out):.1f} % |")
    print(f"| RMS | {rms(x_in):.4f} | {rms(x_out):.4f} |")
    print(f"| ピーク | {float(np.abs(x_in).max()):.3f} | {float(np.abs(x_out).max()):.3f} |")
    print(f"| 調波性 | {harmonicity(x_in, SAMPLE_RATE):.3f} | {harmonicity(x_out, SAMPLE_RATE):.3f} |")
    print()
    print(f"F0 追従（log F0 の相関）    : {f0_corr:.3f}")
    print(f"エネルギー包絡の相関        : {env_corr:.3f}")
    print(f"クリップ（|値| >= 1.0）     : {'あり' if float(np.abs(out).max()) >= 1.0 else 'なし'}")

    if args.hop:
        mean_jump, max_jump = boundary_jumps(out, sr_out, args.hop)
        print(f"継ぎ目の跳び（{args.hop}s ごと）  : 平均 {mean_jump:.4f} / 最大 {max_jump:.4f}")
        print(f"全体の平均 |Δ|              : {float(np.abs(np.diff(out)).mean()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
