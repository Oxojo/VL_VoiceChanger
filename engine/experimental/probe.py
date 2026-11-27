#!/usr/bin/env python3
"""Run an UNTRAINED StreamVC-style neural-core experiment; never emit audio."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=100, help='measured 20ms frames (default: 100)')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0 or args.threads < 1:
        parser.error('frames/threads must be positive, warmup nonnegative')
    import torch
    from engine.experimental.streamvc import StreamVCNeuralCore

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available')

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    print('UNTRAINED neural-core timing/state probe. No voice conversion or audio-quality result.', flush=True)
    model = StreamVCNeuralCore().eval().to(device)
    with torch.inference_mode():
        # Synthetic diagnostics only: zero side channels do NOT implement YIN.
        source = torch.randn(1, 1, 320 * args.frames, device=device) * 0.05
        side = torch.zeros(1, 10, args.frames, device=device)
        reference = torch.randn(1, 1, 16000, device=device) * 0.05
        sync()
        started = time.perf_counter()
        speaker = model.encode_speaker(reference)
        sync()
        speaker_ms = (time.perf_counter() - started) * 1000
        session = model.new_session()
        for _ in range(args.warmup):
            session.push(source[..., :320], side[..., :1], speaker)
        session.reset()
        durations, outputs, sizes = [], [], []
        for i in range(args.frames):
            sync()
            started = time.perf_counter()
            output = session.push(source[..., i*320:(i+1)*320], side[..., i:i+1], speaker)
            sync()
            durations.append((time.perf_counter() - started) * 1000)
            outputs.append(output)
            sizes.append(session.state_bytes)
        streamed = torch.cat(outputs, dim=-1)
        # Check against an independent full-pass execution, not self-consistency.
        full = model(source, side, speaker)
        torch.testing.assert_close(streamed, full, atol=2e-5, rtol=2e-4)
        error = (streamed - full).abs().max().item()
        session.flush()
    ordered = sorted(durations)
    report = {
        'experiment': 'untrained_streamvc_neural_core', 'trained': False,
        'paper': 'https://arxiv.org/abs/2401.03078',
        'torch': torch.__version__, 'device': str(device), 'threads': args.threads,
        'cpu': __import__('platform').processor(),
        'scales': {'content': 64, 'speaker': 32, 'decoder': 40, 'embedding': 64},
        'parameters': sum(p.numel() for p in model.parameters()),
        'frame_ms': 20, 'frames': args.frames, 'warmup_frames': args.warmup,
        'neural_ms': {'mean': statistics.mean(durations), 'p50': statistics.median(durations),
                      'p95': ordered[max(0, __import__('math').ceil(len(ordered)*.95)-1)], 'max': max(durations)},
        'real_time_factor': statistics.mean(durations) / 20,
        'frames_over_20ms': sum(d > 20 for d in durations),
        'speaker_reference_once_ms': speaker_ms,
        'state_bytes': max(sizes), 'state_size_constant': len(set(sizes)) == 1,
        'state_bytes_after_flush': session.state_bytes,
        'full_vs_stream_max_abs_error': error,
        'output_samples': streamed.shape[-1],
        'excluded': ['trained weights and voice quality', 'YIN and running F0 whitening',
                     'paper training/output delay alignment', 'audio capture/transport/playback'],
        'paper_reported_ms': {'architecture': 60, 'pixel7_compute': 10.8, 'total': 70.8},
        'is_end_to_end_latency_measurement': False,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
