"""音声 I/O と前処理。

RVC 系の変換は 16 kHz モノラルを入力に取るため、読み込み時に必ず
16 kHz へリサンプルし、DC 成分を落とすハイパスフィルタを掛ける。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, resample_poly

#: 変換パイプラインの内部サンプリングレート
SAMPLE_RATE = 16000

#: ハイパスフィルタ（RVC 系の前処理と同じ定数）
_HPF_ORDER = 5
_HPF_CUTOFF = 48.0

_b, _a = butter(_HPF_ORDER, _HPF_CUTOFF, btype="high", fs=SAMPLE_RATE)


def load_mono(path: str | Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """WAV をモノラル float32 で読み込み、target_sr へリサンプルする。"""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]

    if sr != target_sr:
        g = math.gcd(int(sr), int(target_sr))
        mono = resample_poly(mono, target_sr // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(mono, dtype=np.float32)


def highpass(x: np.ndarray) -> np.ndarray:
    """48 Hz のハイパスフィルタ（ゼロ位相）。"""
    if x.shape[0] <= 3 * max(len(_a), len(_b)):
        # filtfilt の padlen を満たさない極端に短い入力は素通しする
        return x.astype(np.float32, copy=False)
    return filtfilt(_b, _a, x).astype(np.float32, copy=False)


def save_wav(path: str | Path, audio: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    """float 波形を WAV として保存する（クリップ処理込み）。"""
    out = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out = out * (0.99 / peak)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), out, int(sr), subtype=subtype)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """有理数リサンプリング。"""
    if sr_from == sr_to:
        return x.astype(np.float32, copy=False)
    g = math.gcd(int(sr_from), int(sr_to))
    return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32, copy=False)


def rms_envelope(x: np.ndarray, sr: int, win_s: float = 1.0, hop_s: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """短時間 RMS と、その中心時刻（秒）を返す。"""
    win = max(1, int(round(sr * win_s)))
    hop = max(1, int(round(sr * hop_s)))
    if x.shape[0] < win:
        x = np.pad(x, (0, win - x.shape[0]))
    n = 1 + (x.shape[0] - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    centers = (np.arange(n) * hop + win / 2.0) / sr
    return rms, centers


def match_rms(source: np.ndarray, sr_src: int, target: np.ndarray, sr_tgt: int, rate: float) -> np.ndarray:
    """音量感を入力側へ寄せる（RVC の rms_mix_rate 相当）。

    rate=0 で入力の音量に完全一致、rate=1 で変換結果そのまま。
    """
    rate = float(np.clip(rate, 0.0, 1.0))
    if rate >= 0.999:
        return target

    rms_s, c_s = rms_envelope(source, sr_src)
    rms_t, c_t = rms_envelope(target, sr_tgt)

    t = np.arange(target.shape[0], dtype=np.float64) / sr_tgt
    r_s = np.interp(t, c_s, rms_s, left=rms_s[0], right=rms_s[-1])
    r_t = np.interp(t, c_t, rms_t, left=rms_t[0], right=rms_t[-1])

    r_s = np.maximum(r_s, 1e-6)
    r_t = np.maximum(r_t, 1e-6)
    gain = np.power(r_s, 1.0 - rate) * np.power(r_t, rate - 1.0)
    return (target.astype(np.float64) * gain).astype(np.float32)
