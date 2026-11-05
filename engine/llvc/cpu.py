"""Fixed 13ms, FP32 LLVC ONNX execution; same weights and explicit state I/O.

No quantization, extra lookahead, changed chunk size, or untrained fallback.
CPU graph preparation/validation happen before microphone capture starts.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

INPUT_NAMES = ('audio', 'enc', 'dec', 'out', 'pre')
OUTPUT_NAMES = ('output', 'enc_next', 'dec_next', 'out_next', 'pre_next')


class StepGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio, enc, dec, out, pre):
        # The upstream decoder/prenet update their input state in place. Clone
        # for a functional signature and independent export/parity validation.
        return self.model(audio, enc.clone(), dec.clone(), out.clone(), pre.clone(), pad=False)


def example_inputs(model):
    enc, dec, out = model.init_buffers(1, torch.device('cpu'))
    pre = model.convnet_pre.init_ctx_buf(1, torch.device('cpu'))
    length = model.L * (model.dec_chunk_size + 2)
    audio = torch.linspace(-.1, .1, length).reshape(1, 1, -1)
    return audio, enc, dec, out, pre


class CPURuntime:
    name = 'ONNX Runtime CPU / FP32'

    def __init__(self, graph_bytes):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(graph_bytes, sess_options=options, providers=['CPUExecutionProvider'])
        self.validation_max_error = None

    def __call__(self, audio, enc, dec, out, pre, *, pad=False):
        if pad:
            raise ValueError('CPU runtime accepts only fixed, already-contextualized chunks')
        inputs = {name: value.detach().numpy()
                  for name, value in zip(INPUT_NAMES, (audio, enc, dec, out, pre))}
        return tuple(torch.from_numpy(value) for value in self.session.run(list(OUTPUT_NAMES), inputs))

    @torch.inference_mode()
    def validate(self, model):
        reference = StepGraph(model).eval()
        audio, *initial = example_inputs(model)
        expected_state = [s.clone() for s in initial]
        actual_state = [s.clone() for s in initial]
        maximum = 0.0
        for i in range(6):
            source = torch.sin(audio * (i+1) * 17) * .1
            expected = reference(source, *expected_state)
            actual = self(source, *actual_state)
            for a, b in zip(actual, expected):
                if not torch.isfinite(a).all():
                    raise ValueError('LLVC CPU graph produced non-finite values')
                np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-3, atol=1e-4)
            maximum = max(maximum, float((actual[0]-expected[0]).abs().max()))
            expected_state, actual_state = list(expected[1:]), list(actual[1:])
        self.validation_max_error = maximum


@torch.inference_mode()
def prepare_cpu_runtime(model, checkpoint):
    """Cache by checkpoint identity and exporter/model source/version.

    Validate outputs AND the recurrent state using the actual loaded weights,
    including when reusing a disk cache. Never silently fall back to slow eager
    inference after export or parity failure.
    """
    root = Path(__file__).parent
    digest = hashlib.sha256(('llvc-fixed-step-v1:' + torch.__version__).encode())
    for name in ('cpu.py', 'model.py', 'cached_convnet.py', 'position.py'):
        digest.update((root / name).read_bytes())
    stat = checkpoint.stat()
    identity = f'{stat.st_mtime_ns:x}-{stat.st_size:x}-{digest.hexdigest()[:16]}'
    folder = checkpoint.parent / 'onnx'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{identity}.onnx'
    if path.is_file():
        try:
            runtime = CPURuntime(path.read_bytes())
            runtime.validate(model)
            return runtime
        except Exception:
            # A corrupt/incompatible generated cache is rebuilt once. Source
            # checkpoints are never modified by this path.
            pass
    temporary = None
    try:
        import onnx  # noqa: F401 - legacy export requires it; installed by engine extra
        with tempfile.NamedTemporaryFile(dir=folder, suffix='.onnx', delete=False) as output:
            temporary = Path(output.name)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
            torch.onnx.export(
                StepGraph(model).eval(), example_inputs(model), str(temporary),
                input_names=list(INPUT_NAMES), output_names=list(OUTPUT_NAMES),
                opset_version=17, dynamo=False, external_data=False,
            )
        runtime = CPURuntime(temporary.read_bytes())
        runtime.validate(model)
        os.replace(temporary, path)
        return runtime
    except ImportError as exc:
        raise RuntimeError('LLVC CPUの準備にはonnxが必要です。npm run setupを実行して再起動してください') from exc
    except Exception as exc:
        raise RuntimeError(f'LLVC CPUのONNX準備/数値検証に失敗しました: {exc}') from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
