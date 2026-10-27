"""分割（チャンク）変換と速度計測。

ライブ用途では「一定時間ごとに区切って変換し、遅延なく次の区間へ進む」必要がある。
ここでは 1 回の変換に前後の文脈（context）を付けて品質を保ちながら、
区間ごとの処理時間・実時間比（RTF）を測る。

音声デバイスとの接続（マイク → 変換 → スピーカー）はこのモジュールの
``ChunkedConverter`` をそのまま呼ぶだけで実装できる。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .audio import SAMPLE_RATE
from .convert import ConvertOptions, VoiceConverter


@dataclass
class ChunkConfig:
    """分割変換の設定。"""

    hop_seconds: float = 0.5  # 1 回で進める長さ
    context_seconds: float = 0.15  # 前後に付ける文脈
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


class ChunkedConverter:
    """長い音声を区間に分けて変換する。"""

    def __init__(self, converter: VoiceConverter, config: ChunkConfig | None = None):
        self.converter = converter
        self.config = config or ChunkConfig()

    def process(self, audio16k: np.ndarray, options: ConvertOptions | None = None) -> tuple[np.ndarray, ChunkStats]:
        cfg = self.config
        opt = options or ConvertOptions(seed=cfg.seed)
        sr_out = int(opt.output_sr or self.converter.info.sample_rate)
        sr_model = self.converter.info.sample_rate

        hop = max(1, int(round(cfg.hop_seconds * SAMPLE_RATE)))
        ctx = max(0, int(round(cfg.context_seconds * SAMPLE_RATE)))
        audio = np.asarray(audio16k, dtype=np.float32)

        stats = ChunkStats(audio_seconds=float(audio.shape[0] / SAMPLE_RATE))
        pieces: list[np.ndarray] = []

        for start in range(0, audio.shape[0], hop):
            end = min(start + hop, audio.shape[0])
            ctx_start = max(0, start - ctx)
            ctx_end = min(audio.shape[0], end + ctx)

            t0 = time.perf_counter()
            out, _ = self.converter.convert_array(audio[ctx_start:ctx_end], opt)
            stats.latencies.append(time.perf_counter() - t0)

            # 文脈部分を切り落として、区間 [start, end) に相当する出力だけを残す
            skip_head = int(round((start - ctx_start) / SAMPLE_RATE * sr_out))
            skip_tail = int(round((ctx_end - end) / SAMPLE_RATE * sr_out))
            if skip_head:
                out = out[skip_head:]
            if skip_tail:
                out = out[: out.shape[0] - skip_tail]
            pieces.append(out)
            stats.chunks += 1

        stats.process_seconds = float(sum(stats.latencies))
        if not pieces:
            return np.zeros(0, dtype=np.float32), stats

        joined = np.concatenate(pieces)
        expect = int(round(audio.shape[0] / SAMPLE_RATE * sr_out))
        if joined.shape[0] < expect:
            joined = np.pad(joined, (0, expect - joined.shape[0]))
        elif joined.shape[0] > expect:
            joined = joined[:expect]
        return joined.astype(np.float32), stats
