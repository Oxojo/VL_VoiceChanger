"""Bounded, mono float32 WebSocket transport for the incremental converter."""

from __future__ import annotations

import asyncio
import json
from functools import partial
from urllib.parse import urlsplit

import numpy as np
import torch
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from ..convert import ConvertOptions
from ..realtime import ChunkConfig
from ..streaming import StreamingChunkedConverter

MAX_PACKET_SAMPLES = 1600  # at most 100 ms / message
MAX_STREAM_FRAMES = 18000  # three minutes per session
IDLE_SECONDS = 30


class StreamRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    type: str = "start"
    model: str = Field(min_length=1, max_length=200)
    device: str = "auto"
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
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA を利用できません。CPU を選んでください")
        if self.denoise != 0 or self.dehum != 0:
            raise ValueError("ストリーミングではノイズ除去とハム除去を 0 にしてください")
        if self.output_sr and self.output_sr % 100:
            raise ValueError("出力サンプリングレートは 100 の倍数にしてください")
        conv = manager._converter(self.model, device, Job(id="stream", request={}))
        return StreamingChunkedConverter(
            conv,
            ChunkConfig(self.hop, self.context, self.crossfade, None if self.seed < 0 else self.seed),
            ConvertOptions(
                pitch_shift=self.pitch, speaker=self.speaker, rms_mix=self.rms_mix,
                seed=None if self.seed < 0 else self.seed, output_sr=self.output_sr or None,
            ),
        )


def register_stream_routes(app, manager) -> None:
    @app.websocket("/api/stream")
    async def stream_audio(ws: WebSocket) -> None:
        origin = ws.headers.get("origin")
        if origin and urlsplit(origin).netloc != ws.headers.get("host"):
            await ws.close(code=1008)
            return
        await ws.accept()
        reserved = False
        loop = asyncio.get_running_loop()

        async def work(func, *args):
            return await loop.run_in_executor(manager._executor, partial(func, *args))

        async def status(stream, kind):
            s = stream.stats
            await ws.send_json({
                "type": kind, "input_frames": s.input_frames,
                "emitted_frames": s.emitted_frames, "chunks": s.chunks,
                "rtf": s.rtf if s.input_frames else None,
                "max_inference_ms": s.max_inference_ms,
            })

        try:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=10)
            if len(raw) > 4096:
                raise ValueError("開始メッセージが大きすぎます")
            req = StreamRequest.model_validate_json(raw)
            manager.reserve_stream()
            reserved = True
            stream = await work(req.build, manager)
            await ws.send_json({
                "type": "ready", "input_sr": 16000, "output_sr": stream.output_sr,
                "packet_samples": 320, "max_packet_samples": MAX_PACKET_SAMPLES,
                "max_seconds": MAX_STREAM_FRAMES / 100,
                "buffer_ms": (stream.hop_frames + stream.context_frames + stream.fade_frames) * 10,
            })
            received_samples = 0
            while True:
                message = await asyncio.wait_for(ws.receive(), timeout=IDLE_SECONDS)
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
                    out = await work(stream.push, samples)
                    if out.size:
                        await ws.send_bytes(out.astype("<f4", copy=False).tobytes())
                    # Exactly one ack per input packet, even before first output.
                    # Clients wait for this before sending the next packet.
                    await status(stream, "ack")
                elif message.get("text") is not None:
                    if len(message["text"]) > 128 or json.loads(message["text"]) != {"type": "stop"}:
                        raise ValueError("停止メッセージは {\"type\":\"stop\"} です")
                    out = await work(stream.flush)
                    if out.size:
                        await ws.send_bytes(out.astype("<f4", copy=False).tobytes())
                    await status(stream, "done")
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
            if reserved:
                manager.release_stream()
