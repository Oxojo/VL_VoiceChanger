"""分割（チャンク）変換と速度計測。

ライブ用途では「一定時間ごとに区切って変換し、遅延なく次の区間へ進む」必要がある。
ここでは 1 回の変換に前後の文脈（context）を付けて品質を保ちながら、
区間ごとの処理時間・実時間比（RTF）を測る。

区間の切り出しは 10 ms（16 kHz で 160 サンプル）単位に量子化する。
変換器の出力は ``フレーム数 × (サンプリングレート / 100)`` サンプルに揃うため、
切り出し位置を時間ではなくフレームで数えることで、区間をつなぐときにずれが累積しない。

音声デバイスとの接続（マイク → 変換 → スピーカー）は、この ``ChunkedConverter`` を
そのまま呼ぶだけで実装できる。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .audio import SAMPLE_RATE
from .convert import ConvertOptions, VoiceConverter

#: 変換器の内部フレーム長（16 kHz / 100 fps = 10 ms）
FRAME = 160


@dataclass
class ChunkConfig:
    """分割変換の設定。"""

    hop_seconds: float = 0.5  # 1 回で進める長さ
    context_seconds: float = 0.15  # 前後に付ける文脈
    crossfade_seconds: float = 0.05  # 継ぎ目のクロスフェード長
    seed: int | None = 1234  # 区間ごとに同じシードを使うと境界が安定する


@dataclass
class ChunkStats:
    """分割変換の実測値。"""

    chunks: int = 0
    audio_seconds: float = 0.0
    process_seconds: float = 0.0
    latencies: list[float] = field(default_factory=list)

    @property
    def rtf(self) -> float:
        """実時間比（1.0 未満なら実時間より速い）。"""
        return self.process_seconds / self.audio_seconds if self.audio_seconds else float("nan")

    @property
    def mean_latency(self) -> float:
        return float(np.mean(self.latencies)) if self.latencies else float("nan")

    @property
    def max_latency(self) -> float:
        return float(np.max(self.latencies)) if self.latencies else float("nan")

    def realtime_ok(self, hop_seconds: float) -> bool:
        """各区間の処理が区間長に収まっているか（= ライブで破綻しないか）。"""
        return bool(self.latencies) and self.max_latency < hop_seconds


def _frames(seconds: float, minimum: int = 1) -> int:
    """秒をフレーム数（10 ms 単位）へ丸める。"""
    return max(minimum, int(round(seconds * SAMPLE_RATE / FRAME)))


class ChunkedConverter:
    """長い音声を区間に分けて変換する。"""

    def __init__(self, converter: VoiceConverter, config: ChunkConfig | None = None):
        self.converter = converter
        self.config = config or ChunkConfig()

    def process(self, audio16k: np.ndarray, options: ConvertOptions | None = None) -> tuple[np.ndarray, ChunkStats]:
        cfg = self.config
        opt = options or ConvertOptions(seed=cfg.seed)

        sr_out = int(opt.output_sr or self.converter.info.sample_rate)
        samples_per_frame = sr_out // 100  # 40 kHz なら 400

        hop_frames = _frames(cfg.hop_seconds)
        ctx_frames = _frames(cfg.context_seconds, minimum=0)
        fade_frames = min(_frames(cfg.crossfade_seconds, minimum=0), max(hop_frames - 1, 0))

        audio = np.asarray(audio16k, dtype=np.float32)
        total_frames = audio.shape[0] // FRAME
        stats = ChunkStats(audio_seconds=float(total_frames * FRAME / SAMPLE_RATE))
        pieces: list[np.ndarray] = []

        for start in range(0, total_frames, hop_frames):
            end = min(start + hop_frames, total_frames)
            ctx_start = max(0, start - ctx_frames)
            # 継ぎ目をクロスフェードするため、区間の末尾を少し先まで変換しておく
            ctx_end = min(total_frames, end + ctx_frames + fade_frames)

            segment = audio[ctx_start * FRAME : ctx_end * FRAME]

            t0 = time.perf_counter()
            out, _ = self.converter.convert_array(segment, opt)
            stats.latencies.append(time.perf_counter() - t0)

            skip = (start - ctx_start) * samples_per_frame
            take_frames = min(end + fade_frames, total_frames) - start
            chunk = out[skip : skip + take_frames * samples_per_frame]
            if chunk.shape[0] < take_frames * samples_per_frame:
                chunk = np.pad(chunk, (0, take_frames * samples_per_frame - chunk.shape[0]))

            if pieces and fade_frames > 0:
                n = min(fade_frames * samples_per_frame, pieces[-1].shape[0], chunk.shape[0])
                if n > 0:
                    fade = np.linspace(0.0, 1.0, n, dtype=np.float32)
                    pieces[-1][-n:] = pieces[-1][-n:] * (1.0 - fade) + chunk[:n] * fade
                    chunk = chunk[n:]

            pieces.append(chunk)
            stats.chunks += 1

        stats.process_seconds = float(sum(stats.latencies))
        if not pieces:
            return np.zeros(0, dtype=np.float32), stats

        joined = np.concatenate(pieces)
        expected = total_frames * samples_per_frame
        if joined.shape[0] < expected:
            joined = np.pad(joined, (0, expected - joined.shape[0]))
        elif joined.shape[0] > expected:
            joined = joined[:expected]
        return joined.astype(np.float32), stats
