"""RVC 系チェックポイント（.pth）の読み込み。

チェックポイントは次の構造を持つ:
    {
      "weight": state_dict,
      "config": [spec_channels, segment_size, inter_channels, hidden_channels,
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout,
                 resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel,
                 upsample_kernel_sizes, spk_embed_dim, gin_channels, sr],
      "sr": "40k", "f0": 1, "version": "v2", "info": "..."
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from .synthesizer import SynthesizerTrnMs768NSFsid

SR_TABLE = {"32k": 32000, "40k": 40000, "48k": 48000}


@dataclass
class ModelInfo:
    """チェックポイントのメタ情報。"""

    path: Path
    sample_rate: int
    version: str
    has_f0: bool
    speakers: int
    info: str = ""
    config: list = field(default_factory=list)


def _parse_sr(value) -> int:
    if isinstance(value, str):
        if value in SR_TABLE:
            return SR_TABLE[value]
        return int(value.rstrip("kK")) * 1000
    return int(value)


def load_checkpoint(path: str | Path) -> dict:
    """ローカルの .pth を読み、中身の辞書をそのまま返す。"""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "weight" not in ckpt:
        raise ValueError(f"RVC のチェックポイント形式ではありません: {path}")
    return ckpt


def build_model(ckpt: dict, path: str | Path = "", device: str = "cpu") -> tuple[SynthesizerTrnMs768NSFsid, ModelInfo]:
    """チェックポイントからモデルを構築し、重みを読み込む。"""
    version = str(ckpt.get("version", "v1"))
    if version != "v2":
        raise ValueError(f"自前エンジンは RVC v2 のみ対応です（version={version}）")

    cfg = list(ckpt["config"])
    if len(cfg) != 18:
        raise ValueError(f"config の要素数が想定外です: {len(cfg)}")

    (
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels,
        sr,
    ) = cfg

    has_posterior = any(k.startswith("enc_q.") for k in ckpt["weight"])

    model = SynthesizerTrnMs768NSFsid(
        spec_channels=spec_channels,
        segment_size=segment_size,
        inter_channels=inter_channels,
        hidden_channels=hidden_channels,
        filter_channels=filter_channels,
        n_heads=n_heads,
        n_layers=n_layers,
        kernel_size=kernel_size,
        p_dropout=p_dropout,
        resblock=resblock,
        resblock_kernel_sizes=list(resblock_kernel_sizes),
        resblock_dilation_sizes=[list(d) for d in resblock_dilation_sizes],
        upsample_rates=list(upsample_rates),
        upsample_initial_channel=upsample_initial_channel,
        upsample_kernel_sizes=list(upsample_kernel_sizes),
        spk_embed_dim=spk_embed_dim,
        gin_channels=gin_channels,
        sr=_parse_sr(sr),
        with_posterior=has_posterior,
    )

    missing, unexpected = model.load_state_dict(ckpt["weight"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "重みの読み込みに失敗しました。\n"
            f"  未初期化: {sorted(missing)[:8]}{' ...' if len(missing) > 8 else ''}\n"
            f"  余剰キー: {sorted(unexpected)[:8]}{' ...' if len(unexpected) > 8 else ''}"
        )

    model.eval().to(device)

    info = ModelInfo(
        path=Path(path),
        sample_rate=model.sr,
        version=version,
        has_f0=bool(int(ckpt.get("f0", 1))),
        speakers=int(spk_embed_dim),
        info=str(ckpt.get("info", "")),
        config=cfg,
    )
    return model, info
