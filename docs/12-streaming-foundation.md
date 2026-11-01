# Incremental audio input: first streaming foundation

This feature is **not** StreamVC and does **not** make RVC causal. It adds
`engine.streaming.StreamingChunkedConverter`: a bounded-history adapter
around the existing `VoiceConverter.convert_array()`. This is the first
step toward a microphone → conversion worker → speaker pipeline while
preserving all existing offline and VC Client commands.

## What it does

- Receives arbitrary 16 kHz mono `float32` blocks via `push(samples)`.
- Buffers incomplete 10 ms / 160-sample frames until a whole frame arrives.
- Waits for `hop + context + crossfade` audio before processing a full hop.
- Reuses the existing RVC converter on overlapping source windows.
- Blends adjacent 50 ms tails, **emitting only committed audio** (no
  previously returned samples are changed).
- Discards source history older than the needed left context.
- `flush()` handles the final short block and drops a final incomplete
  10 ms frame, matching the current offline chunking convention.
- Records per-inference timings and RTF (not end-to-end latency).

The existing RVC model still recomputes RMVPE, ContentVec and the synth
for each overlapping window. Feature caching, cached causal convolutions,
sound-device I/O, low-latency scheduling and StreamVC weights are **not
implemented**. The ONNX feature extractors continue using CPU execution
providers by default, even if the RVC synthesizer uses CUDA.

**Do not call `push()` from a real-time audio callback.** The method runs
blocking model inference; use an audio ring buffer and a dedicated worker
thread/process, followed by a nonblocking playback queue.

### Example (array chunks, no audio device)

```python
import numpy as np
from engine.audio import load_mono
from engine.convert import VoiceConverter
from engine.realtime import ChunkConfig
from engine.streaming import StreamingChunkedConverter

converter = VoiceConverter("models/tsukuyomi/tsukuyomi-normal1.pth", device="cuda")
stream = StreamingChunkedConverter(converter, ChunkConfig(
    hop_seconds=0.5, context_seconds=0.15, crossfade_seconds=0.05,
))
wav = load_mono("logs/verify/tts-input-ja.wav")  # float32 mono / 16 kHz

outputs = []
for pos in range(0, wav.size, 320):  # mock 20 ms microphone arrivals
    outputs.append(stream.push(wav[pos:pos + 320]))
outputs.append(stream.flush())
result = np.concatenate(outputs)
print(stream.stats.rtf, stream.stats.max_inference_ms)
```

### Run the benchmark with existing model assets

```bash
python -m engine stream-bench \
  -m models/tsukuyomi/tsukuyomi-normal1.pth \
  -i logs/verify/tts-input-ja.wav \
  -o logs/verify/stream-sim.wav \
  --input-frame-ms 20 \
  --hop 0.5 --context 0.15 --crossfade 0.05 \
  --device cuda
```

The CLI **simulates incremental arrival** from a recorded WAV as quickly
as it can; it is not a live mic-to-speaker loop. The printed
`first-output input` is the minimum buffering/peek-ahead before
inference. It does not include model compute time, scheduling, input
device buffering or playback buffering.

For these defaults the first hop requires 700 ms of captured audio before
it can be converted. Even if its inference becomes faster than real time,
this configuration is not suitable for interactive low-latency live use.

## Current limitations / next implementation steps

1. **Profile per stage**: RMVPE vs ContentVec vs synth, and compare
   `engine bench` with `engine stream-bench` for identical parameters.
2. **GPU feature inference**: explicitly configure ONNX Runtime GPU
   providers and benchmark; check actual `session.get_providers()`,
   rather than assuming CUDA torch implies ONNX on GPU.
3. **Causal input processing**: the current 48 Hz high-pass uses a
   zero-phase (noncausal) filter; optional spectral denoise uses whole
   recording statistics, and dehum is also zero-phase. This adapter
   rejects denoise/dehum until stateful alternatives exist.
4. **Model streaming**: ContentVec and RVC synth need bounded lookahead
   / cache investigation. Merely decreasing `hop` often *increases*
   redundant inference overhead. A causal StreamVC backend should
   implement explicit encoder, F0 and decoder state.
5. **Audio device and queues**: dedicated input/output callbacks,
   worker, timestamped PCM blocks and underrun/overrun metrics.
6. **Verification**: test actual RTF p50/p95/max, algorithmic latency,
   microphone-to-speaker latency, quality and click-free boundaries.

## Tests

```bash
python -m unittest tests.test_streaming -v
python -m unittest discover -s tests -v
```

The tests use an identity mock; no model weights or GPU are needed.
They check arbitrary input boundaries, sample-exact output length,
correct crossfade, bounded left history, final tails, reset and
validation of offline-only preprocessing.
