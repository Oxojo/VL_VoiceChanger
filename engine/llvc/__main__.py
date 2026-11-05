"""Download or verify real LLVC inference on a WAV. No synthetic fallback."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from .assets import install_checkpoint
from .streaming import load_stream
from ..audio import load_mono


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model', default='llvc-default', help='LLVC voice ID, e.g. llvc:my-voice')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if not args.download and args.input is None:
        parser.error('specify --download and/or --input')
    if args.input is not None and args.output is None:
        parser.error('--input requires --output')
    if args.download:
        print(json.dumps(install_checkpoint(), indent=2))
    if args.input is None:
        return
    audio = load_mono(args.input, 16000)
    if not audio.size:
        parser.error("入力音声が空です")
    stream = load_stream(args.device, {}, model_id=args.model)
    outputs = []
    started = time.perf_counter()
    try:
        for i in range(0, len(audio), 320):
            outputs.append(stream.push(audio[i:i+320]))
        outputs.append(stream.flush())
    finally:
        stream.close()
    elapsed = time.perf_counter() - started
    output = np.concatenate(outputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, output, 16000)
    durations = np.asarray(stream.stats.latencies) * 1000
    print(json.dumps({'input_samples': len(audio), 'output_samples': len(output),
                      'wall_seconds': elapsed, 'inference_rtf': stream.stats.rtf,
                      'chunk_ms': stream.chunk_samples/16,
                      'inference_ms_mean': float(durations.mean()) if durations.size else None,
                      'inference_ms_p95': float(np.percentile(durations,95)) if durations.size else None,
                      'note': 'file streaming simulation; not microphone-to-ear latency'}, indent=2))


if __name__ == '__main__':
    main()
