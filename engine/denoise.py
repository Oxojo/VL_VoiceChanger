"""入力音声のノイズ除去（自前実装）。

学習済みモデルを使わず、スペクトルサブトラクションで定常ノイズを抑える。
マイクの空調・ファン・ハム（50/60 Hz）のような「ずっと鳴っている音」に効く。
話者の声と同時に鳴る非定常ノイズ（紙ずれ、笑い声など）は落としきれない。

処理の流れ:

    16 kHz 波形
      → STFT（Hann 窓 / 1024 点 / hop 256）
      → 時間方向の下位パーセンタイルでノイズ成分を推定（周波数ごと）
      → 推定ノイズを差し引いたマスクを作る（強さで過減算量を変える）
      → マスクを時間方向に平滑化（「音楽的ノイズ」の抑制）
      → ISTFT で波形に戻す

CLI の ``engine denoise`` と、変換前処理（``ConvertOptions.denoise``）の両方から使う。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import filtfilt, iirnotch, istft, stft

from .audio import SAMPLE_RATE

#: STFT の既定値（16 kHz で 64 ms 窓 / 16 ms ホップ）
N_FFT = 1024
HOP = 256


def noise_floor_db(x: np.ndarray, sr: int = SAMPLE_RATE, frame_ms: float = 20.0, percentile: float = 10.0) -> float:
    """静かな部分の RMS（dBFS）を返す。ノイズ除去の効果を測るのに使う。"""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    win = max(1, int(sr * frame_ms / 1000.0))
    frames = x[: x.shape[0] // win * win].reshape(-1, win)
    if frames.size == 0:
        return -120.0
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return float(20.0 * np.log10(max(float(np.percentile(rms, percentile)), 1e-10)))


def spectral_gate(
    x: np.ndarray,
    sr: int = SAMPLE_RATE,
    strength: float = 0.6,
    n_fft: int = N_FFT,
    hop: int = HOP,
    noise_percentile: float = 15.0,
) -> np.ndarray:
    """スペクトルサブトラクションで定常ノイズを抑える。

    Args:
        x: 入力波形（モノラル）
        sr: サンプリングレート
        strength: 0（無効）〜1（強め）。0.5 前後が声を壊しにくい
        n_fft: FFT 長
        hop: ホップ長
        noise_percentile: ノイズ推定に使う時間方向のパーセンタイル
    Returns:
        同じ長さの波形
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0 or x.shape[0] < n_fft * 2:
        return x.copy()

    stft_kw = dict(
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        boundary="zeros",
        padded=True,
        scaling="spectrum",
    )
    istft_kw = dict(
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        boundary=True,
        scaling="spectrum",
    )

    _, _, spec = stft(x, **stft_kw)
    mag = np.abs(spec)

    # 周波数ごとに「時間方向で静かな方」をノイズとみなす
    noise = np.percentile(mag, noise_percentile, axis=1)
    noise = np.convolve(noise, np.ones(3) / 3.0, mode="same")

    # 過減算（strength が大きいほど強く引く）
    over = 1.0 + 2.0 * strength
    ratio = mag / (over * noise[:, None] + 1e-12)
    mask = np.clip(1.0 - 1.0 / np.maximum(ratio, 1e-6), 0.0, 1.0)

    # 完全に 0 にすると不自然なので下限を残す（弱い設定ほど多く残す）
    mask = np.maximum(mask, 0.02 + 0.06 * (1.0 - strength))

    # 時間方向の平滑化（フレーム間の急変によるプツッという音を防ぐ）
    smooth = np.empty_like(mask)
    prev = mask[:, 0].copy()
    for i in range(mask.shape[1]):
        prev = 0.65 * prev + 0.35 * mask[:, i]
        smooth[:, i] = prev

    _, y = istft(spec * smooth, **istft_kw)
    y = np.asarray(y, dtype=np.float32)
    if y.shape[0] < x.shape[0]:
        y = np.pad(y, (0, x.shape[0] - y.shape[0]))
    return y[: x.shape[0]]


def dehum(x: np.ndarray, sr: int = SAMPLE_RATE, mains: float = 50.0, harmonics: int = 3, q: float = 30.0) -> np.ndarray:
    """電源ハム（50/60 Hz とその倍音）をノッチフィルタで落とす。"""
    if mains <= 0:
        return np.asarray(x, dtype=np.float32).copy()
    y = np.asarray(x, dtype=np.float64).reshape(-1)
    for k in range(1, max(1, int(harmonics)) + 1):
        freq = mains * k
        if freq >= sr / 2.0 - 1.0:
            break
        b, a = iirnotch(w0=freq, Q=q, fs=sr)
        y = filtfilt(b, a, y)
    return y.astype(np.float32)


def reduce_noise(x: np.ndarray, sr: int = SAMPLE_RATE, strength: float = 0.0, dehum_hz: float = 0.0) -> np.ndarray:
    """ハム除去 → スペクトルゲートの順に適用する（両方 0 ならそのまま返す）。"""
    y = np.asarray(x, dtype=np.float32).reshape(-1)
    if dehum_hz and dehum_hz > 0:
        y = dehum(y, sr, mains=dehum_hz)
    if strength and strength > 0:
        y = spectral_gate(y, sr, strength=strength)
    return y
