"""Bounded, mono float32 WebSocket transport for the incremental converter."""

from __future__ import annotations

import asyncio
import json
import time
from functools import partial
from urllib.parse import urlsplit
from typing import Literal

import numpy as np
import torch
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from ..convert import ConvertOptions
from ..features.providers import feature_providers
from ..realtime import ChunkConfig
from ..streaming import StreamingChunkedConverter

MAX_PACKET_SAMPLES = 1600  # at most 100 ms / message
MAX_STREAM_FRAMES = 18000  # three minutes per session
IDLE_SECONDS = 30


class StreamRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    type: str = "start"
    backend: Literal["rvc", "llvc"] = "rvc"
    model: str = Field(min_length=1, max_length=200)
    device: str = "auto"
    ort_cuda: bool = False
    debug: bool = False
    warmup: bool = True
    feature_cache: bool = False
    pitch: int = Field(default=0, ge=-24, le=24)
    speaker: int = Field(default=0, ge=0, le=255)
    rms_mix: float = Field(default=0.25, ge=0, le=1)
    seed: int = 1234
    output_sr: int = Field(default=0, ge=0, le=96000)
    hop: float = Field(default=0.5, ge=0.1, le=2)
    context: float = Field(default=0.15, ge=0, le=1)
    crossfade: float = Field(default=0.05, ge=0, le=0.2)
    denoise: float = 0
    dehum: float = 0

    def build(self, manager):
        # Called only on the same executor as offline conversion. Model cache,
        # torch seed and model inference are never used concurrently.
        from .server import Job

        if self.type != "start":
            raise ValueError("最初に start を送ってください")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device は auto / cpu / cuda を選んでください")
        device = self.device
        if device == "auto":
            device = "cpu" if self.backend == "llvc" else ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA を利用できません。CPU を選んでください")
        if self.backend == "llvc":
            if (self.feature_cache or self.ort_cuda
                    or self.pitch or self.speaker or self.denoise or self.dehum
                    or self.output_sr not in (0, 16000)):
                raise ValueError("LLVCはLLVC用に学習した声を使います。RVCモデル・ピッチ・話者・ONNX・特徴量キャッシュ等の設定は直接使えません")
            from ..llvc.streaming import load_stream
            return load_stream(device, manager._llvc_models, warmup=self.warmup, model_id=self.model)
        if self.denoise != 0 or self.dehum != 0:
            raise ValueError("ストリーミングではノイズ除去とハム除去を 0 にしてください")
        if self.output_sr and self.output_sr % 100:
            raise ValueError("出力サンプリングレートは 100 の倍数にしてください")
        conv = manager._converter(self.model, device, Job(id="stream", request={}), ort_cuda=self.ort_cuda)
        stream = StreamingChunkedConverter(
            conv,
            ChunkConfig(self.hop, self.context, self.crossfade, None if self.seed < 0 else self.seed),
            ConvertOptions(
                pitch_shift=self.pitch, speaker=self.speaker, rms_mix=self.rms_mix,
                seed=None if self.seed < 0 else self.seed, output_sr=self.output_sr or None,
            ),
            feature_cache=self.feature_cache,
        )
        stream.warmup_seconds = 0.0
        if self.warmup:
            # Exercise both the first window (no left context) and the next
            # steady-state window before the browser opens its microphone.
            # Reset only adapter state; keep the loaded model/CUDA kernels.
            started = time.perf_counter()
            lengths = dict.fromkeys((
                stream.hop_frames + stream.context_frames + stream.fade_frames,
                stream.hop_frames + stream.context_frames * 2 + stream.fade_frames,
            ))
            for frames in lengths:
                # A disposable adapter validates the model output while avoiding
                # the many intermediate history lengths when context > hop.
                probe = StreamingChunkedConverter(
                    conv, ChunkConfig(frames / 100, 0, 0), stream.options,
                )
                probe.push(np.zeros(frames * 160, dtype=np.float32))
            if self.feature_cache:
                try:
                    frames = stream.hop_frames * 3 + stream.context_frames + stream.fade_frames
                    stream.push(np.zeros(frames * 160, dtype=np.float32))
                finally:
                    stream.reset()
            stream.warmup_seconds = time.perf_counter() - started
        return stream


def register_stream_routes(app, manager) -> None:
    @app.websocket("/api/stream")
    async def stream_audio(ws: WebSocket) -> None:
        origin = ws.headers.get("origin")
        if origin and urlsplit(origin).netloc != ws.headers.get("host"):
            await ws.close(code=1008)
            return
        await ws.accept()
        reserved = False
        stream = None
        build_task = None
        loop = asyncio.get_running_loop()

        async def work(func, *args, trace=None):
            if trace is None:
                return await loop.run_in_executor(manager._executor, partial(func, *args))
            submitted = time.perf_counter()

            def measured():
                started = time.perf_counter()
                trace["executor_wait_ms"] = (started - submitted) * 1000
                try:
                    return func(*args)
                finally:
                    trace["worker_ms"] = (time.perf_counter() - started) * 1000

            return await loop.run_in_executor(manager._executor, measured)

        async def status(stream, kind, debug=None):
            s = stream.stats
            payload = {
                "type": kind, "input_frames": s.input_frames,
                "emitted_frames": s.emitted_frames, "chunks": s.chunks,
                "rtf": s.rtf if s.input_frames else None,
                "max_inference_ms": s.max_inference_ms,
                "stage_ms": {name: seconds * 1000 / s.chunks
                             for name, seconds in s.stage_seconds.items()} if s.chunks else {},
                "feature_cache": stream.feature_cache.summary() if stream.feature_cache is not None else {"enabled": False},
            }
            if debug is not None:
                payload["debug"] = debug
            await ws.send_json(payload)

        try:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=10)
            if len(raw) > 4096:
                raise ValueError("開始メッセージが大きすぎます")
            req = StreamRequest.model_validate_json(raw)
            manager.reserve_stream()
            reserved = True
            build_task = asyncio.create_task(work(req.build, manager))
            stream = await asyncio.shield(build_task)
            ready = {
                "type": "ready", "input_sr": 16000, "output_sr": stream.output_sr,
                "packet_samples": 320, "max_packet_samples": MAX_PACKET_SAMPLES,
                "max_seconds": MAX_STREAM_FRAMES / 100,
                "buffer_ms": 20 if req.backend == "llvc" else (stream.hop_frames + stream.context_frames + stream.fade_frames) * 10,
                "backend": req.backend,
                "runtime": getattr(stream, "runtime_name", None),
                "runtime_validation_max_error": getattr(stream, "validation_max_error", None),
                "preflight": getattr(stream, "preflight", None),
                "feature_providers": feature_providers(stream.converter) if req.backend == "rvc" else {},
                "playback_lead_ms": 15 if req.backend == "llvc" else 50,
                "warmup_ms": stream.warmup_seconds * 1000,
                "feature_cache": {"enabled": req.feature_cache,
                                  "guard_ms": stream.feature_cache.guard_frames * 10 if req.feature_cache else 0},
            }
            if req.debug:
                ready["debug"] = {
                    "hop_ms": stream.hop_frames * 10,
                    "context_ms": stream.context_frames * 10,
                    "crossfade_ms": stream.fade_frames * 10,
                }
            await ws.send_json(ready)
            received_samples = 0
            request_seq = 0

            async def convert(func, samples=None):
                nonlocal request_seq
                request_seq += 1
                if not req.debug:
                    out = await work(func, samples) if samples is not None else await work(func)
                    if out.size:
                        await ws.send_bytes(out.astype("<f4", copy=False).tobytes())
                    return None
                trace = {}
                first_frame = stream.stats.emitted_frames
                inference_before = stream.stats.process_seconds
                stages_before = stream.stats.stage_seconds.copy()
                out = (await work(func, samples, trace=trace) if samples is not None
                       else await work(func, trace=trace))
                if out.size:
                    await ws.send_bytes(out.astype("<f4", copy=False).tobytes())
                trace.update({
                    "request_seq": request_seq,
                    "server_ms": (time.perf_counter() - received_at) * 1000,
                    "inference_ms": (stream.stats.process_seconds - inference_before) * 1000,
                    "output_start_frame": first_frame,
                    "output_frames": stream.stats.emitted_frames - first_frame,
                    "feature_cache": stream.feature_cache.summary() if stream.feature_cache is not None else {"enabled": False},
                    "stage_ms": {name: (seconds - stages_before.get(name, 0)) * 1000
                                 for name, seconds in stream.stats.stage_seconds.items()
                                 if seconds != stages_before.get(name, 0)},
                })
                return trace

            while True:
                message = await asyncio.wait_for(ws.receive(), timeout=IDLE_SECONDS)
                received_at = time.perf_counter() if req.debug else None
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    data = message["bytes"]
                    if not data or len(data) % 4 or len(data) > MAX_PACKET_SAMPLES * 4:
                        raise ValueError("音声は 1〜1600 サンプルの float32 mono で送ってください")
                    samples = np.frombuffer(data, dtype="<f4")
                    if not np.isfinite(samples).all():
                        raise ValueError("音声に NaN / Infinity が含まれています")
                    received_samples += samples.size
                    if received_samples > MAX_STREAM_FRAMES * 160:
                        raise ValueError("試験セッションは 180 秒までです。停止して再開してください")
                    debug = await convert(stream.push, samples)
                    # Exactly one ack per input packet, even before first output.
                    # Clients wait for this before sending the next packet.
                    await status(stream, "ack", debug)
                elif message.get("text") is not None:
                    if len(message["text"]) > 128 or json.loads(message["text"]) != {"type": "stop"}:
                        raise ValueError("停止メッセージは {\"type\":\"stop\"} です")
                    debug = await convert(stream.flush)
                    await status(stream, "done", debug)
                    await ws.close(code=1000)
                    break
                else:
                    raise ValueError("音声または停止メッセージを送ってください")
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            # Report model/asset/validation failures to the UI; never substitute
            # a mock model or silently drop audio in the production path.
            try:
                detail = getattr(exc, "detail", None) or str(exc)
                if isinstance(exc, asyncio.TimeoutError):
                    detail = "音声の受信がタイムアウトしました。再接続してください"
                await ws.send_json({"type": "error", "message": detail})
                await ws.close(code=1011)
            except (WebSocketDisconnect, RuntimeError, OSError):
                pass
        finally:
            if reserved and req.backend == "rvc":
                manager.release_stream()
            elif reserved:
                async def cleanup():
                    nonlocal stream
                    # ASGI may cancel the connection during model loading or
                    # disconnect. Keep the reservation until in-flight work and
                    # same-worker cleanup finish, even after that cancellation.
                    if stream is None and build_task is not None:
                        try:
                            stream = await build_task
                        except Exception:
                            pass
                    def finish():
                        try:
                            close = getattr(stream, "close", None)
                            if close is not None:
                                close()
                        finally:
                            manager.release_stream()
                    await work(finish)
                await asyncio.shield(cleanup())
