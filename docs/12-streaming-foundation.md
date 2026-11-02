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
- Validates the waveform length returned by the backend against the
  **padded model input length**. Differences exceeding one 10 ms output
  frame raise `ValueError` instead of silently filling large gaps with
  zeros. Validation happens before committing chunk statistics or samples.
  Small resampling/rounding differences (up to one frame) remain tolerated.
- If a later inference in one `push()` or `flush()` fails, the adapter
  restores its output position, retained context/fade and completed-chunk
  statistics to before that call. Received input stays buffered. After
  correcting the backend, retry with `push([])` or `flush()` respectively;
  do not resend the same input. `reset()` discards it instead. Backend
  internal state is not restored, so stateful backends may require reset.

The existing RVC model still recomputes RMVPE, ContentVec and the synth
for each overlapping window. Feature caching, cached causal convolutions,
sound-device I/O, low-latency scheduling and StreamVC weights are **not
implemented in the adapter**. The Web UI now connects browser microphone
capture and incremental playback through `/api/stream`; see
[Web UI instructions](08-web-ui.md#マイクのストリーミング変換試験機能).
The ONNX feature extractors continue using CPU execution
providers by default, even if the RVC synthesizer uses CUDA.
The Web UI also exposes explicit CUDA selection for feature extraction,
verifies both sessions, and displays actual providers and per-stage mean timings.
See [the CUDA trial](13-web-onnx-cuda.md) for installation and A/B comparison.

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
  --device cuda --ort-cuda
```

To enable ONNX Runtime CUDA for ContentVec and RMVPE, install a
matching `onnxruntime-gpu` build first and pass `--ort-cuda`. This is
opt-in: plain `--device cuda` accelerates PyTorch synthesis but does not
switch the ONNX extractors. The CLI validates that the ONNX sessions
actually have CUDA first in their provider list, and errors otherwise.
The same `--ort-cuda` option is available for `engine convert` and
`engine bench` to allow fair comparisons.

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
5. **Audio device and queues**: the Web UI now provides AudioWorklet capture,
   a serial inference worker, bounded input/playback queues and playback
   underrun counts. Native audio I/O, timestamped end-to-end latency and
   low-latency model scheduling remain future work.
6. **Verification**: test actual RTF p50/p95/max, algorithmic latency,
   microphone-to-speaker latency, quality and click-free boundaries.

## Tests

```bash
python -m unittest discover -s tests -p 'test_streaming.py' -v
python -m unittest discover -s tests -v
```

The unit tests use model-free fakes, so they can run without model
weights or GPU. In addition to the identity mock, the regression suite
uses a converter producing **different waveforms per chunk** and
compares the incremental assembly to the pre-existing offline
`ChunkedConverter.process()` for varied input lengths, fade widths and
arbitrary input block boundaries. Further cases cover:

- 1/7/160/320-sample input arrivals and a final partial (<10ms) frame.
- One-frame (<100ms) input padded for model inference, but emitted at
  exactly its original length.
- 48 kHz synthesis output length and audio-sample alignment.
- Fade endpoints (1→2 and 2→3), avoiding silent double/missing mixing.
- A faulty backend returning a waveform 50ms shorter than expected:
  the streaming adapter rejects it **without marking the chunk
  complete**.

**Verification note:** These are deterministic unit-level cases.
They do not validate the actual ContentVec, RMVPE or RVC model
weights, quality of live output or end-to-end latency. Run the full
repository tests and model-based benchmarks in the environment that
has its ONNX assets and PyTorch dependencies before merging.

For a short final chunk, `StreamingChunkedConverter` pads the input
to 100 ms for the backend's minimum-length requirement and trims the
output back to the original number of 10ms frames. The offline
`ChunkedConverter` is left unchanged and may reject a standalone
final segment shorter than 100ms with the real converter. This is
an intentional documented behavioral difference, not strict
model-output equivalence for those very short inputs.
