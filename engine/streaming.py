"""Incremental WAV-to-WAV RVC adapter (not a causal model).

This accepts arbitrary pieces of 16 kHz mono PCM, retains only recent input
context, and emits *committed* samples.  It still invokes the existing
non-causal VoiceConverter on overlapping windows; it is an I/O foundation
and a measurement target, NOT a low-latency / state-cached RVC inference
implementation.  Call push() from a worker, never from an audio callback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .audio import SAMPLE_RATE
from .convert import ConvertOptions, VoiceConverter
from .realtime import ChunkConfig, FRAME, _frames


@dataclass
class StreamStats:
    """Statistics for completed 10 ms input frames and inference calls."""

    input_frames: int = 0
    emitted_frames: int = 0
    chunks: int = 0
    process_seconds: float = 0.0
    latencies: list[float] = field(default_factory=list)

    @property
    def input_seconds(self) -> float:
        return self.input_frames / 100.0

    @property
    def rtf(self) -> float:
        return self.process_seconds / self.input_seconds if self.input_frames else float("nan")

    @property
    def max_inference_ms(self) -> float:
        return max(self.latencies, default=0.0) * 1000.0


class StreamingChunkedConverter:
    """Process incrementally received audio using the *existing* RVC backend.

    push() returns zero or more newly committed output samples.  flush()
    processes the remaining input (dropping <10 ms of trailing source
    samples, as ChunkedConverter does).  Neither method waits for a
    microphone or speaker: the caller owns device I/O and worker queues.

    After each conversion only the necessary left context is retained.
    The right lookahead is `context_seconds + crossfade_seconds`, in
    addition to the hop needed to assemble one output block.
    """

    def __init__(
        self,
        converter: VoiceConverter,
        config: ChunkConfig | None = None,
        options: ConvertOptions | None = None,
    ) -> None:
        self.converter = converter
        self.config = config or ChunkConfig()
        self.options = options or ConvertOptions(seed=self.config.seed)

        if self.config.hop_seconds <= 0 or self.config.context_seconds < 0 or self.config.crossfade_seconds < 0:
            raise ValueError("hop must be positive; context and crossfade must be nonnegative")
        if self.options.denoise > 0 or self.options.dehum > 0:
            raise ValueError("offline denoise/dehum uses whole-signal statistics or zero-phase filters; disable for streaming")

        self.output_sr = int(self.options.output_sr or self.converter.info.sample_rate)
        if self.output_sr <= 0 or self.output_sr % 100:
            raise ValueError("output sampling rate must be positive and divisible by 100")

        self.samples_per_frame = self.output_sr // 100
        self.hop_frames = _frames(self.config.hop_seconds)
        self.context_frames = _frames(self.config.context_seconds, minimum=0)
        self.fade_frames = min(
            _frames(self.config.crossfade_seconds, minimum=0),
            max(self.hop_frames - 1, 0),
        )
        self.reset()

    def reset(self) -> None:
        """Reset input/framing state without reloading model weights."""
        self._buffer = np.zeros(0, dtype=np.float32)
        self._unframed = np.zeros(0, dtype=np.float32)
        self._buffer_start = 0  # absolute 10 ms frame index
        self._total_frames = 0
        self._next_frame = 0
        self._pending_fade = np.zeros(0, dtype=np.float32)
        self._closed = False
        self.stats = StreamStats()

    def push(self, samples: np.ndarray) -> np.ndarray:
        """Append arbitrary 16 kHz mono PCM samples and emit ready output."""
        if self._closed:
            raise RuntimeError("stream already flushed; call reset() to reuse it")
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("expected a one-dimensional mono audio array")
        if x.size:
            pending = np.concatenate((self._unframed, x))
            n = pending.size // FRAME * FRAME
            if n:
                self._buffer = np.concatenate((self._buffer, pending[:n]))
                self._total_frames += n // FRAME
            self._unframed = pending[n:].copy()
            self.stats.input_frames = self._total_frames

        output = []
        required_future = self.context_frames + self.fade_frames
        while self._total_frames >= self._next_frame + self.hop_frames + required_future:
            output.append(self._convert_chunk(self._next_frame + self.hop_frames))
        return self._join(output)

    def flush(self) -> np.ndarray:
        """Complete all whole frames; discard a final incomplete 10 ms frame."""
        if self._closed:
            return self._empty()
        output = []
        while self._next_frame < self._total_frames:
            output.append(self._convert_chunk(min(self._next_frame + self.hop_frames, self._total_frames)))
        self._closed = True
        self._unframed = self._empty()
        self._pending_fade = self._empty()
        return self._join(output)

    def _convert_chunk(self, end: int) -> np.ndarray:
        start = self._next_frame
        left = max(0, start - self.context_frames)
        right = min(self._total_frames, end + self.context_frames + self.fade_frames)
        start_sample = (left - self._buffer_start) * FRAME
        end_sample = (right - self._buffer_start) * FRAME
        segment = self._buffer[start_sample:end_sample]

        # The current RVC converter refuses source clips shorter than 0.1 s.
        # Only short final tails need padding; trim to the real frame count below.
        if segment.size < int(0.1 * SAMPLE_RATE):
            segment = np.pad(segment, (0, int(0.1 * SAMPLE_RATE) - segment.size))

        t0 = time.perf_counter()
        out, _ = self.converter.convert_array(segment, self.options)
        dt = time.perf_counter() - t0
        self.stats.chunks += 1
        self.stats.latencies.append(dt)
        self.stats.process_seconds += dt

        # Align with the existing 100 fps frame contract even for truncated
        # or resampled backend output.
        wanted = (right - left) * self.samples_per_frame
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        if out.size < wanted:
            out = np.pad(out, (0, wanted - out.size))
        else:
            out = out[:wanted]

        skip = (start - left) * self.samples_per_frame
        take = (min(end + self.fade_frames, self._total_frames) - start) * self.samples_per_frame
        chunk = out[skip : skip + take].copy()

        # The previous chunk's extra fade overlaps this chunk's opening;
        # emit *exactly* (end-start) new frames, keeping this chunk's tail.
        if self._pending_fade.size:
            n = min(self._pending_fade.size, chunk.size)
            if n:
                fade = np.linspace(0.0, 1.0, n, dtype=np.float32)
                chunk[:n] = self._pending_fade[:n] * (1.0 - fade) + chunk[:n] * fade

        committed = (end - start) * self.samples_per_frame
        emitted = chunk[:committed].copy()
        self._pending_fade = chunk[committed:].copy()
        self._next_frame = end
        self.stats.emitted_frames += end - start

        # Reclaim history; only the previous left context is needed again.
        keep_from = max(0, self._next_frame - self.context_frames)
        discard = (keep_from - self._buffer_start) * FRAME
        if discard:
            self._buffer = self._buffer[discard:].copy()
            self._buffer_start = keep_from
        return emitted

    def _empty(self) -> np.ndarray:
        return np.empty(0, dtype=np.float32)

    def _join(self, pieces: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(pieces) if pieces else self._empty()
