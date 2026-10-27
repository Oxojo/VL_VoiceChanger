"""マスク生成・畳み込みパディング・重み正規化などの小さな部品。"""

from __future__ import annotations

import torch
from torch import nn


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Conv 系の重みを初期化する（学習はしないが、構築時の挙動を揃える）。"""
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    """[B] の長さから [B, max_length] の bool マスクを作る。"""
    if max_length is None:
        max_length = int(length.max())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def fused_add_tanh_sigmoid_multiply(input_a: torch.Tensor, input_b: torch.Tensor, n_channels: int) -> torch.Tensor:
    """WaveNet 系のゲート（tanh × sigmoid）。"""
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    return t_act * s_act


def weight_norm(module: nn.Module, name: str = "weight", dim: int = 0) -> nn.Module:
    """RVC のチェックポイント形式（weight_g / weight_v）に合わせる重み正規化。

    PyTorch の旧 ``torch.nn.utils.weight_norm`` と同じパラメータ名・同じ式
    （w = g * v / ||v||）を、前方フックで自前に実装したもの。
    ``state_dict`` には ``<name>_g`` と ``<name>_v`` だけが現れる。
    """
    weight = getattr(module, name)
    if weight is None:
        raise ValueError(f"{type(module).__name__} に {name} がありません")

    del module._parameters[name]

    reduce_dims = tuple(d for d in range(weight.dim()) if d != dim)
    g = torch.linalg.vector_norm(weight.detach(), dim=reduce_dims, keepdim=True).clamp_min(1e-12)
    module.register_parameter(name + "_g", nn.Parameter(g.clone()))
    module.register_parameter(name + "_v", nn.Parameter(weight.detach().clone()))

    def _normalized(m: nn.Module) -> torch.Tensor:
        v = getattr(m, name + "_v")
        gg = getattr(m, name + "_g")
        n = torch.linalg.vector_norm(v, dim=reduce_dims, keepdim=True).clamp_min(1e-12)
        return v * (gg / n)

    # 構築直後にも ``module.weight`` を参照できるようにしておく（初期化処理のため）。
    # 実際の値は毎回の forward 前にフックで貼り直すので、重みを読み込んでもずれない。
    setattr(module, name, _normalized(module))

    def _pre_hook(m: nn.Module, _inputs):
        setattr(m, name, _normalized(m))

    module.register_forward_pre_hook(_pre_hook)
    return module


def remove_weight_norm(module: nn.Module) -> nn.Module:
    """学習しないため何もしない（API 互換のためのダミー）。"""
    return module
