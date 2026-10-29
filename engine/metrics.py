"""変換結果を数値で比べるための指標。

CLI（scripts/measure_quality.py）と Web UI の両方から使う。
すべて 16 kHz モノラル波形を前提にしている。
"""

from __future__ import annotations

import numpy as np

from .audio import SAMPLE_RATE


def rms(x: np.ndarray) -> float:
    """二乗平均平方根。"""
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def peak(x: np.ndarray) -> float:
    """絶対値の最大。"""
    return float(np.abs(x).max()) if x.size else 0.0


def envelope(x: np.ndarray, win: int = 1600) -> np.ndarray:
    """短時間 RMS の包絡（既定 100 ms 窓）。"""
    m = x.shape[0] // win
    if m == 0:
        return np.zeros(0)
    return np.sqrt((x[: m * win].reshape(m, win).astype(np.float64) ** 2).mean(axis=1))


def f0_stats(f0: np.ndarray) -> tuple[float, float]:
    """(中央値 [Hz], 有声率 [%]) を返す。"""
    voiced = f0[f0 > 0]
    median = float(np.median(voiced)) if voiced.size else 0.0
    ratio = float(voiced.size / f0.size * 100) if f0.size else 0.0
    return median, ratio


def f0_correlation(f_in: np.ndarray, f_out: np.ndarray) -> float:
    """両方が有声のフレームだけで log F0 の相関を取る。"""
    n = min(f_in.shape[0], f_out.shape[0])
    both = (f_in[:n] > 0) & (f_out[:n] > 0)
    if both.sum() < 10:
        return float("nan")
    return float(np.corrcoef(np.log(f_in[:n][both]), np.log(f_out[:n][both]))[0, 1])


def envelope_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """エネルギー包絡の相関。"""
    ea, eb = envelope(a), envelope(b)
    m = min(ea.shape[0], eb.shape[0])
    if m < 2:
        return float("nan")
    return float(np.corrcoef(ea[:m], eb[:m])[0, 1])


def harmonicity(x: np.ndarray, sr: int = SAMPLE_RATE) -> float:
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


def compare(x_in: np.ndarray, x_out: np.ndarray, f_in: np.ndarray, f_out: np.ndarray) -> dict[str, float]:
    """入力と出力の比較指標をまとめて返す（すべて JSON にできる形）。"""
    f0_in, voiced_in = f0_stats(f_in)
    f0_out, voiced_out = f0_stats(f_out)
    return {
        "f0_median_in": f0_in,
        "f0_median_out": f0_out,
        "voiced_ratio_in": voiced_in,
        "voiced_ratio_out": voiced_out,
        "f0_correlation": f0_correlation(f_in, f_out),
        "envelope_correlation": envelope_correlation(x_in, x_out),
        "harmonicity_in": harmonicity(x_in),
        "harmonicity_out": harmonicity(x_out),
        "rms_in": rms(x_in),
        "rms_out": rms(x_out),
        "peak_out": peak(x_out),
        "clipped": bool(peak(x_out) >= 1.0),
    }


def boundary_jumps(x: np.ndarray, sr: int, hop_seconds: float) -> tuple[float, float]:
    """継ぎ目のサンプル跳び（平均, 最大）。分割変換のクリック検出に使う。"""
    step = int(round(hop_seconds * sr))
    if step <= 0:
        return float("nan"), float("nan")
    idx = np.arange(step, x.shape[0], step) - 1
    if idx.size == 0:
        return float("nan"), float("nan")
    jump = np.abs(x[idx] - x[idx + 1])
    return float(jump.mean()), float(jump.max())
