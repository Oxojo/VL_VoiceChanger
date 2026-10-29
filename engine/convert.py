"""WAV → WAV の変換パイプライン。

流れ（すべて自前コード。VC Client のサーバーは使わない）:

    入力 WAV
      → 16 kHz モノラル + 48 Hz ハイパス
      → 前後 80 サンプルを反射パディング
      → RMVPE で F0（100 fps）
      → ContentVec layer12 で特徴量（50 fps）→ 2 倍に展開（100 fps）
      → ピッチシフト → メル尺度で 1-255 に量子化
      → RVC v2 合成器（自前実装）で 40 kHz 波形を生成
      → パディング除去 → 音量を入力に寄せる → 出力 WAV
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from .audio import SAMPLE_RATE, highpass, load_mono, match_rms, resample, save_wav
from .config import resolve_asset
from .denoise import noise_floor_db, reduce_noise
from .features import ContentVec, RMVPE
from .model import ModelInfo, build_model, load_checkpoint

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

#: F0 の量子化レンジ（RVC 共通の定数）
F0_MIN = 50.0
F0_MAX = 1100.0

#: 波形の前後パディング（16 kHz 換算サンプル数 = 5 ms）
PAD_SAMPLES = 80


def f0_to_coarse(f0: np.ndarray, f0_min: float = F0_MIN, f0_max: float = F0_MAX) -> np.ndarray:
    """F0 [Hz] をメル尺度の 1-255 の離散値へ変換する。"""
    f0_mel_min = 1127.0 * np.log(1.0 + f0_min / 700.0)
    f0_mel_max = 1127.0 * np.log(1.0 + f0_max / 700.0)

    f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
    positive = f0_mel > 0
    f0_mel[positive] = (f0_mel[positive] - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    return np.rint(f0_mel).astype(np.int64)


def interpolate_unvoiced(f0: np.ndarray) -> np.ndarray:
    """無声区間（0 Hz）を前後の有声区間で線形補間する。"""
    f0 = f0.copy()
    uv = f0 <= 0
    if not uv.any():
        return f0
    if uv.all():
        f0[:] = 0.0
        return f0
    idx = np.arange(f0.shape[0])
    f0[uv] = np.interp(idx[uv], idx[~uv], f0[~uv])
    return f0


@dataclass
class ConvertOptions:
    """変換時の調整パラメータ。

    子音保護（protect）と検索インデックス（index）は未実装のため持たない。
    """

    pitch_shift: int = 0
    speaker: int = 0
    rms_mix: float = 0.25
    seed: int | None = 1234
    output_sr: int | None = None
    pad_samples: int = PAD_SAMPLES
    #: ノイズ除去の強さ（0 で無効、0.5 前後が声を壊しにくい）
    denoise: float = 0.0
    #: 電源ハムの周波数（0 で無効、50 / 60 Hz）
    dehum: float = 0.0


@dataclass
class ConvertReport:
    """変換結果のメタ情報（検証記録用）。"""

    input_seconds: float
    output_seconds: float
    output_sr: int
    frames: int
    f0_median: float
    f0_voiced_ratio: float
    rms_in: float
    rms_out: float
    timings: dict[str, float] = field(default_factory=dict)
    #: ノイズ除去の前後における静音部の RMS（dBFS）
    noise_floor_in_db: float = 0.0
    noise_floor_clean_db: float = 0.0


class VoiceConverter:
    """RVC v2 + ContentVec + RMVPE を組み合わせた変換器。"""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        contentvec_path: str | Path | None = None,
        rmvpe_path: str | Path | None = None,
        providers: list[str] | None = None,
    ):
        self.device = device
        t0 = time.perf_counter()

        ckpt = load_checkpoint(model_path)
        self.model, self.info = build_model(ckpt, path=model_path, device=device)

        cv = Path(contentvec_path) if contentvec_path else resolve_asset("contentvec")
        rm = Path(rmvpe_path) if rmvpe_path else resolve_asset("rmvpe")
        self.contentvec = ContentVec(cv, providers=providers)
        self.rmvpe = RMVPE(rm, providers=providers)

        self.load_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------ 変換
    @staticmethod
    def _fit_frames(tensor: torch.Tensor, frames: int) -> torch.Tensor:
        """[1, T, C] を [1, frames, C] に合わせる（不足は末尾を複製）。"""
        cur = tensor.shape[1]
        if cur == frames:
            return tensor
        if cur > frames:
            return tensor[:, :frames, :]
        pad = frames - cur
        return F.pad(tensor.transpose(1, 2), (0, pad), mode="replicate").transpose(1, 2)

    @staticmethod
    def _fit_frames_1d(array: np.ndarray, frames: int) -> np.ndarray:
        """[T] を [frames] に合わせる（不足は末尾を複製）。"""
        cur = array.shape[0]
        if cur == frames:
            return array
        if cur > frames:
            return array[:frames]
        return np.pad(array, (0, frames - cur), mode="edge")

    def convert_array(
        self,
        waveform16k: np.ndarray,
        options: ConvertOptions | None = None,
        on_stage: "Callable[[str, float], None] | None" = None,
    ) -> tuple[np.ndarray, ConvertReport]:
        """16 kHz モノラル波形を変換し、(波形, レポート) を返す。

        ``on_stage`` を渡すと、段階（f0 / contentvec / synth / post）と
        おおよその進捗（0-1）が通知される。Web UI の進捗表示に使う。
        """
        opt = options or ConvertOptions()
        timings: dict[str, float] = {}

        def stage(name: str, progress: float) -> None:
            if on_stage is not None:
                on_stage(name, progress)

        stage("start", 0.0)

        audio_in = np.asarray(waveform16k, dtype=np.float32)
        if audio_in.shape[0] < int(0.1 * SAMPLE_RATE):
            raise ValueError("入力が短すぎます（0.1 秒以上にしてください）")

        audio = highpass(audio_in)

        # --- ノイズ除去（任意）--------------------------------------------
        denoise_seconds = 0.0
        if opt.denoise > 0 or opt.dehum > 0:
            t = time.perf_counter()
            audio = reduce_noise(audio, SAMPLE_RATE, strength=opt.denoise, dehum_hz=opt.dehum)
            denoise_seconds = time.perf_counter() - t
            timings["denoise"] = denoise_seconds
            stage("denoise", 0.05)

        audio_pad = np.pad(audio, (opt.pad_samples, opt.pad_samples), mode="reflect")
        p_len = audio_pad.shape[0] // 160

        # --- F0（100 fps）------------------------------------------------
        t = time.perf_counter()
        f0 = self.rmvpe.infer(audio_pad)
        if f0.shape[0] < p_len:
            f0 = np.pad(f0, (0, p_len - f0.shape[0]), mode="edge")
        f0 = f0[:p_len]
        voiced_ratio = float(np.mean(f0 > 0)) if p_len else 0.0
        f0 = interpolate_unvoiced(f0)
        f0 = f0 * (2.0 ** (opt.pitch_shift / 12.0))
        pitchf = f0.astype(np.float32)
        pitch = f0_to_coarse(pitchf)
        timings["f0"] = time.perf_counter() - t
        stage("f0", 0.15)

        # --- 特徴量（50 fps → 100 fps）-----------------------------------
        t = time.perf_counter()
        feats = self.contentvec.extract(audio_pad, layer=12)
        feats_t = torch.from_numpy(feats)
        feats_t = F.interpolate(feats_t.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        timings["contentvec"] = time.perf_counter() - t
        stage("contentvec", 0.35)

        # ContentVec のフレーム数は 20 ms 単位で切り捨てられるため、
        # p_len（10 ms 単位）に合わせて端を複製・切り詰める。
        # ここを揃えないと区間ごとに最大 20 ms ずれ、分割変換で時間ずれが累積する。
        feats_t = self._fit_frames(feats_t, p_len)
        pitch = self._fit_frames_1d(pitch, p_len)
        pitchf = self._fit_frames_1d(pitchf, p_len)

        # --- 合成 --------------------------------------------------------
        t = time.perf_counter()
        generator = None
        if opt.seed is not None:
            torch.manual_seed(int(opt.seed))
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(opt.seed))
                generator = torch.Generator(device=self.device).manual_seed(int(opt.seed))
            else:
                generator = torch.Generator(device="cpu").manual_seed(int(opt.seed))

        phone = feats_t.to(self.device)
        lengths = torch.tensor([p_len], dtype=torch.long, device=self.device)
        pitch_t = torch.from_numpy(pitch).long().unsqueeze(0).to(self.device)
        pitchf_t = torch.from_numpy(pitchf).float().unsqueeze(0).to(self.device)
        sid = torch.tensor([int(opt.speaker)], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.model.infer(phone, lengths, pitch_t, pitchf_t, sid, generator=generator)
        audio_out = out[0, 0].detach().cpu().float().numpy()
        timings["synth"] = time.perf_counter() - t
        stage("synth", 0.9)

        # --- 後処理 ------------------------------------------------------
        t = time.perf_counter()
        trim = int(round(opt.pad_samples * self.info.sample_rate / SAMPLE_RATE))
        if trim > 0:
            audio_out = audio_out[trim:-trim] if audio_out.shape[0] > 2 * trim else audio_out

        audio_out = match_rms(audio, SAMPLE_RATE, audio_out, self.info.sample_rate, opt.rms_mix)

        out_sr = int(opt.output_sr or self.info.sample_rate)
        if out_sr != self.info.sample_rate:
            audio_out = resample(audio_out, self.info.sample_rate, out_sr)
        timings["post"] = time.perf_counter() - t
        stage("post", 1.0)

        report = ConvertReport(
            input_seconds=float(audio.shape[0] / SAMPLE_RATE),
            output_seconds=float(audio_out.shape[0] / out_sr),
            output_sr=out_sr,
            frames=int(p_len),
            f0_median=float(np.median(pitchf[pitchf > 0])) if np.any(pitchf > 0) else 0.0,
            f0_voiced_ratio=voiced_ratio,
            rms_in=float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0,
            rms_out=float(np.sqrt(np.mean(np.square(audio_out, dtype=np.float64)))) if audio_out.size else 0.0,
            noise_floor_in_db=noise_floor_db(audio_in, SAMPLE_RATE),
            noise_floor_clean_db=noise_floor_db(audio, SAMPLE_RATE),
            timings=timings,
        )
        return audio_out, report

    def convert_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        options: ConvertOptions | None = None,
        subtype: str = "PCM_16",
    ) -> ConvertReport:
        """ファイルを読み、変換して WAV として書き出す。"""
        t = time.perf_counter()
        wav = load_mono(input_path, SAMPLE_RATE)
        audio_out, report = self.convert_array(wav, options)
        report.timings["total"] = time.perf_counter() - t
        save_wav(output_path, audio_out, report.output_sr, subtype=subtype)
        return report
