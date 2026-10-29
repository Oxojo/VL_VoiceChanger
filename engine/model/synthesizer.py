"""RVC v2（40k / 48k 系）の音声合成器。

``SynthesizerTrnMs768NSFsid`` 相当の推論経路だけを実装する。
学習用の損失計算・事後分布サンプリングは行わない。
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .attentions import Encoder
from .commons import sequence_mask
from .modules import PosteriorEncoder, ResBlock1, ResidualCouplingBlock, weight_norm


class TextEncoder(nn.Module):
    """ContentVec 特徴量 + 粗いピッチ → 潜在変数の分布パラメータ。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb_phone = nn.Linear(in_channels, hidden_channels)
        self.emb_pitch = nn.Embedding(256, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self, phone: torch.Tensor, pitch: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = self.lrelu(x * math.sqrt(self.hidden_channels))  # [B, T, H]
        x = torch.transpose(x, 1, -1)  # [B, H, T]

        x_mask = sequence_mask(lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class SineGen(nn.Module):
    """F0 から正弦波の励振信号を作る（NSF の音源）。"""

    def __init__(
        self,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
    ):
        super().__init__()
        self.sampling_rate = samp_rate
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

    def _f02sine(self, f0: torch.Tensor, upp: int) -> torch.Tensor:
        """f0 [B, T, 1] を upp 倍に展開した正弦波 [B, T*upp, dim]。"""
        rad = f0 / self.sampling_rate * torch.arange(1, upp + 1, device=f0.device, dtype=f0.dtype)

        # フレーム境界で位相が跳ばないよう、直前フレームの位相を累積して補正する
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0.dtype)
        rad = rad + F.pad(rad_acc, (0, 0, 1, -1))

        rad = rad.reshape(f0.shape[0], -1, 1)
        rad = rad * torch.arange(1, self.dim + 1, device=f0.device, dtype=f0.dtype).reshape(1, 1, -1)
        return torch.sin(2 * math.pi * rad)

    def forward(self, f0: torch.Tensor, upp: int):
        """f0 [B, T] → (sine [B, T*upp, dim], uv [B, T*upp, 1], noise 同形状)"""
        f0 = f0[:, None].transpose(1, 2)  # [B, T, 1]
        sine_waves = self._f02sine(f0, upp) * self.sine_amp

        uv = (f0 > self.voiced_threshold).to(f0.dtype)
        uv = F.interpolate(uv.transpose(2, 1), scale_factor=float(upp), mode="nearest").transpose(2, 1)

        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        return sine_waves * uv + noise, uv, noise


class SourceModuleHnNSF(nn.Module):
    """正弦波を 1 本の励振信号へ合成するモジュール。"""

    def __init__(self, sampling_rate: int, harmonic_num: int = 0, sine_amp: float = 0.1, add_noise_std: float = 0.003, voiced_threshold: float = 0.0):
        super().__init__()
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, upp: int = 1) -> torch.Tensor:
        sine_waves, _, _ = self.l_sin_gen(x, upp)
        return self.l_tanh(self.l_linear(sine_waves))


class GeneratorNSF(nn.Module):
    """HiFi-GAN + NSF の波形生成器。"""

    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gin_channels: int,
        sr: int,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upp = math.prod(upsample_rates)

        self.f0_upsamp = nn.Upsample(scale_factor=self.upp)
        self.m_source = SourceModuleHnNSF(sampling_rate=sr, harmonic_num=0)

        self.noise_convs = nn.ModuleList()
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        if resblock != "1":
            raise NotImplementedError("ResBlock2 は未実装です（RVC v2 の配布モデルは '1'）")

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        c_cur,
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    nn.Conv1d(1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2)
                )
            else:
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1) if gin_channels != 0 else None

    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        har_source = self.m_source(f0, self.upp).transpose(1, 2)  # [B, 1, T*upp]

        x = self.conv_pre(x)
        if g is not None and self.cond is not None:
            x = x + self.cond(g)

        for i, (ups, noise_conv) in enumerate(zip(self.ups, self.noise_convs)):
            x = F.leaky_relu(x, 0.1)
            x = ups(x)
            x = x + noise_conv(har_source)

            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j]
                xs = rb(x) if xs is None else xs + rb(x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        return torch.tanh(x)


class SynthesizerTrnMs768NSFsid(nn.Module):
    """RVC v2 の推論用モデル（768 次元特徴 / NSF 付き）。"""

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        spk_embed_dim: int,
        gin_channels: int,
        sr: int,
        with_posterior: bool = True,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.segment_size = segment_size
        self.spk_embed_dim = spk_embed_dim
        self.gin_channels = gin_channels
        self.sr = sr

        self.enc_p = TextEncoder(
            768,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )
        self.dec = GeneratorNSF(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
        )
        self.enc_q = (
            PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
            if with_posterior
            else None
        )
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    @property
    def hop_length(self) -> int:
        """潜在 1 フレームあたりの出力サンプル数（100 fps → 400）。"""
        return self.dec.upp

    @torch.no_grad()
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        nsff0: torch.Tensor,
        sid: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """ContentVec 特徴量と F0 から波形を生成する。

        Args:
            phone: [1, T, 768] ContentVec 特徴量（100 fps に展開済み）
            phone_lengths: [1] 有効フレーム数
            pitch: [1, T] 粗いピッチ（1-255 の離散値）
            nsff0: [1, T] 連続 F0 [Hz]
            sid: [1] 話者 ID
        Returns:
            [1, 1, T * 400] の波形（-1..1）
        """
        g = self.emb_g(sid).unsqueeze(-1)

        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        noise = torch.randn(m_p.shape, generator=generator, dtype=m_p.dtype, device=m_p.device)
        z_p = (m_p + torch.exp(logs_p) * noise * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        return self.dec(z * x_mask, nsff0, g=g)
