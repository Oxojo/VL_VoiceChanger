"""Arbitrary-packet adapter for the official LLVC infer_stream convention.

It deliberately retains the upstream 1*L input shift and 2*L preceding context
(including convnet-prenet overlap). No RVC model or feature extractor is used.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from ..streaming import StreamStats
from . import assets
from .model import Net


class LLVCStream:
    output_sr = 16000
    feature_cache = None
    fade_frames = 0

    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.chunk_samples = model.L * model.dec_chunk_size
        self.hop_frames = self.chunk_samples / 160
        self.context_frames = model.L / 160
        self.warmup_seconds = 0.0
        self.restore_threads = None
        if not model.lookahead or self.chunk_samples < 2 * model.L:
            raise ValueError('このアダプタは公式LLVCのlookahead付き設定専用です')
        self.reset()

    @torch.inference_mode()
    def reset(self):
        self.buffers = self.model.init_buffers(1, self.device)
        self.pre_context = (self.model.convnet_pre.init_ctx_buf(1, self.device)
                            if hasattr(self.model, 'convnet_pre') else None)
        self.previous = np.zeros(2 * self.model.L, dtype=np.float32)
        self.pending = np.empty(0, dtype=np.float32)
        self.skip = self.model.L
        self.received = self.emitted = 0
        self.stats = StreamStats()
        self.closed = self.failed = False

    def close(self):
        self.buffers = self.pre_context = None
        self.previous = self.pending = np.empty(0, dtype=np.float32)
        self.closed = True
        if self.restore_threads is not None:
            torch.set_num_threads(self.restore_threads)
            self.restore_threads = None

    @torch.inference_mode()
    def _step(self, body):
        packet = torch.from_numpy(np.concatenate((self.previous, body))).to(self.device)[None, None, :]
        start = time.perf_counter()
        output, *states = self.model(packet, *self.buffers, self.pre_context, pad=False)
        audio = output.detach().cpu().numpy().reshape(-1).copy()
        elapsed = time.perf_counter() - start  # CPU copy synchronizes CUDA output.
        if audio.size != self.chunk_samples or not np.isfinite(audio).all():
            self.failed = True
            raise ValueError('LLVCが不正な長さ/非有限値の音声を返しました。再接続してください')
        self.buffers = tuple(states[:3])
        self.pre_context = states[3]
        self.previous = body[-2*self.model.L:].copy()
        self.stats.chunks += 1
        self.stats.process_seconds += elapsed
        self.stats.latencies.append(elapsed)
        self.stats.stage_seconds['llvc'] = self.stats.process_seconds
        return audio.astype(np.float32, copy=False)

    def push(self, samples):
        if self.closed or self.failed:
            raise RuntimeError('LLVCセッションが終了/失敗しています。再接続してください')
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError('LLVC入力は有限値の16kHz mono PCMで送ってください')
        self.received += samples.size
        self.stats.input_frames = self.received / 160
        skipped = min(self.skip, samples.size)
        self.skip -= skipped
        self.pending = np.concatenate((self.pending, samples[skipped:]))
        outputs = []
        try:
            while self.pending.size >= self.chunk_samples:
                outputs.append(self._step(self.pending[:self.chunk_samples]))
                self.pending = self.pending[self.chunk_samples:].copy()
        except Exception:
            self.failed = True  # upstream mutates buffers: never retry a failed state
            raise
        result = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
        self.emitted += result.size
        self.stats.emitted_frames = self.emitted / 160
        return result

    def flush(self):
        if self.closed:
            return np.empty(0, dtype=np.float32)
        if self.failed:
            raise RuntimeError('LLVC推論が失敗しています。再接続してください')
        outputs = []
        remaining = self.received - self.emitted
        try:
            while remaining > 0:
                body = np.pad(self.pending, (0, self.chunk_samples-self.pending.size))
                audio = self._step(body)
                count = min(remaining, audio.size)
                outputs.append(audio[:count])
                remaining -= count
                self.pending = np.empty(0, dtype=np.float32)
        except Exception:
            self.failed = True
            raise
        result = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
        self.emitted += result.size
        self.stats.emitted_frames = self.emitted / 160
        self.close()
        return result


def load_stream(device, cache, warmup=True):
    checkpoint = assets.CHECKPOINT
    if not checkpoint.is_file():
        raise ValueError('LLVCモデルがありません。「LLVCモデルを取得」を実行してください')
    stat = checkpoint.stat()
    key = (device, stat.st_mtime_ns, stat.st_size)
    if key not in cache:
        config = json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))
        with torch.random.fork_rng(devices=[]):
            model = Net(**config['model_params'])
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(payload['model'], strict=True)
        del payload
        model.eval().to(device)
        # Keep only the current checkpoint, at most one copy per device.
        for old in list(cache):
            if old[1:] != key[1:]:
                del cache[old]
        cache[key] = model
    stream = LLVCStream(cache[key])
    # Tiny CPU chunks suffer from large thread pools. The server reserves all
    # conversion work for this session, then closes it on the same executor.
    if device == 'cpu':
        stream.restore_threads = torch.get_num_threads()
        torch.set_num_threads(1)
    try:
        if warmup:
            start = time.perf_counter()
            stream.push(np.zeros(stream.chunk_samples * 3 + stream.model.L, dtype=np.float32))
            stream.reset()
            stream.warmup_seconds = time.perf_counter() - start
        return stream
    except Exception:
        stream.close()
        raise
