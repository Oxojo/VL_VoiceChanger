"""Single-session FP32 CUDA Graph runner for fixed 13ms LLVC chunks.

Replay removes per-operator Python/driver launches. Audio and all state remain
explicit inputs/outputs; the caller copies audio to the host only after replay.
Output tensors are graph-owned views valid until the next call. Never share a
runner across sessions. The loaded model weights may still be shared.
"""
from __future__ import annotations

import torch


class CUDAGraphRuntime:
    name = 'PyTorch CUDA Graph / FP32'

    @torch.inference_mode()
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        if self.device.type != 'cuda':
            raise ValueError('CUDA GraphにはCUDA上のモデルが必要です')
        self.validation_max_error = None
        with torch.cuda.device(self.device):
            length = model.L * (model.dec_chunk_size + 2)
            audio = torch.zeros(1, 1, length, device=self.device)
            state = model.init_buffers(1, self.device)
            pre = model.convnet_pre.init_ctx_buf(1, self.device)
            self.inputs = (audio, *state, pre)
            self.graph = torch.cuda.CUDAGraph()
            # Warm cuBLAS/cuDNN and allocators outside capture, on a side stream.
            side = torch.cuda.Stream(device=self.device)
            current = torch.cuda.current_stream(self.device)
            side.wait_stream(current)
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._forward()
            current.wait_stream(side)
            torch.cuda.synchronize(self.device)
            with torch.cuda.graph(self.graph, stream=side):
                self.outputs = self._forward()
            current.wait_stream(side)
        self.validate()

    def _forward(self):
        audio, enc, dec, out, pre = self.inputs
        # Decoder/prenet mutate their input state. Preserve a functional step;
        # allocations/clones are captured and reused on replay, not reallocated.
        return self.model(audio, enc.clone(), dec.clone(), out.clone(), pre.clone(), pad=False)

    @torch.inference_mode()
    def __call__(self, audio, enc, dec, out, pre, *, pad=False):
        if pad:
            raise ValueError('CUDA Graph accepts only fixed, contextualized chunks')
        values = (audio, enc, dec, out, pre)
        for destination, value in zip(self.inputs, values):
            if value.device != self.device or value.dtype != destination.dtype or value.shape != destination.shape:
                raise ValueError('CUDA Graph入力のdevice/dtype/shapeが一致しません')
        # Enqueue all copies before replay on the same current stream. In normal
        # streaming the source states are the previous graph outputs on device.
        with torch.cuda.device(self.device):
            for destination, value in zip(self.inputs, values):
                destination.copy_(value, non_blocking=True)
            self.graph.replay()
        return self.outputs

    @torch.inference_mode()
    def validate(self):
        expected_state = [s.clone().zero_() for s in self.inputs[1:]]
        actual_state = [s.clone().zero_() for s in self.inputs[1:]]
        phase = torch.linspace(-1, 1, self.inputs[0].numel(), device=self.device).reshape_as(self.inputs[0])
        maximum = 0.0
        for i in range(6):
            source = torch.sin(phase * (i+1)) * .1
            expected = self.model(source, *(s.clone() for s in expected_state), pad=False)
            actual = self(source, *actual_state)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)
            maximum = max(maximum, float((actual[0]-expected[0]).abs().max().cpu()))
            expected_state = list(expected[1:])
            actual_state = list(actual[1:])
        self.validation_max_error = maximum


def prepare_cuda_runtime(model):
    try:
        return CUDAGraphRuntime(model)
    except Exception as exc:
        raise RuntimeError('LLVCのCUDA Graph準備/数値検証に失敗しました。'
                           'デバイスをCPUにしてONNX Runtime経路を使用してください。'
                           f' 詳細: {exc}') from exc
