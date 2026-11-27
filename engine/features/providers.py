"""Explicit ONNX CUDA selection shared by CLI and Web UI."""

from __future__ import annotations

import onnxruntime as ort


def ort_providers(enable_cuda: bool) -> list[str] | None:
    if not enable_cuda:
        return None
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "ONNX CUDA provider is not installed. Install a compatible onnxruntime-gpu "
            "and verify its CUDA/cuDNN runtime before enabling ONNX CUDA."
        )
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def feature_providers(converter) -> dict[str, list[str]]:
    return {
        name: list(getattr(converter, name).session.get_providers())
        for name in ("contentvec", "rmvpe")
    }


def verify_ort_cuda(converter, enable_cuda: bool) -> None:
    if not enable_cuda:
        return
    for name, providers in feature_providers(converter).items():
        if not providers or providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"{name}: expected CUDAExecutionProvider, got {providers}")
