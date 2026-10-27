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

import numpy as np
import torch
import torch.nn.functional as F

from .audio import SAMPLE_RATE, highpass, load_mono, match_rms, resample, save_wav
from .config import resolve_asset
from .features import ContentVec, RMVPE
from .model import ModelInfo, build_model, load_checkpoint

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
    def convert_array(self, waveform16k: np.ndarray, options: ConvertOptions | None = None) -> tuple[np.ndarray, ConvertReport]:
        """16 kHz モノラル波形を変換し、(波形, レポート) を返す。"""
        opt = options or ConvertOptions()
        timings: dict[str, float] = {}

        audio_in = np.asarray(waveform16k, dtype=np.float32)
        if audio_in.shape[0] < int(0.1 * SAMPLE_RATE):
            raise ValueError("入力が短すぎます（0.1 秒以上にしてください）")

        audio = highpass(audio_in)
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

        # --- 特徴量（50 fps → 100 fps）-----------------------------------
        t = time.perf_counter()
        feats = self.contentvec.extract(audio_pad, layer=12)
        feats_t = torch.from_numpy(feats)
        feats_t = F.interpolate(feats_t.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        timings["contentvec"] = time.perf_counter() - t

        if feats_t.shape[1] < p_len:
            p_len = int(feats_t.shape[1])
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
        else:
            feats_t = feats_t[:, :p_len, :]

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

        report = ConvertReport(
            input_seconds=float(audio.shape[0] / SAMPLE_RATE),
            output_seconds=float(audio_out.shape[0] / out_sr),
            output_sr=out_sr,
            frames=int(p_len),
            f0_median=float(np.median(pitchf[pitchf > 0])) if np.any(pitchf > 0) else 0.0,
            f0_voiced_ratio=voiced_ratio,
            rms_in=float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0,
            rms_out=float(np.sqrt(np.mean(np.square(audio_out, dtype=np.float64)))) if audio_out.size else 0.0,
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
