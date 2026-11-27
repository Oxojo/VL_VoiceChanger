"""Paper-inspired StreamVC neural-core feasibility experiment (arXiv:2401.03078).

This is NOT a trained voice converter or a reproduction of the paper's results.
No weights, YIN/whitening, training alignment or browser transport are included.
Side features and a speaker embedding are supplied by the caller. State belongs
exclusively to a session, never to shared model weights. Inference only.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

State = dict[int, torch.Tensor]


class CausalConv(nn.Conv1d):
    """Left-padded convolution, retaining both history and stride phase."""

    def __init__(self, incoming, outgoing, kernel, stride=1, dilation=1):
        super().__init__(incoming, outgoing, kernel, stride=stride, dilation=dilation)
        self.history = dilation * (kernel - 1) + 1 - stride
        if self.history < 0:
            raise ValueError('effective kernel must be at least stride')

    def forward(self, x):
        if x.shape[-1] < self.stride[0]:
            return x.new_empty(x.shape[0], self.out_channels, 0)
        return super().forward(F.pad(x, (self.history, 0)))

    def step(self, x, state: State):
        if not x.shape[-1]:
            return x.new_empty(x.shape[0], self.out_channels, 0)
        previous = state.get(id(self))
        if previous is None:
            previous = x.new_zeros(x.shape[0], self.in_channels, self.history)
        joined = torch.cat((previous, x), dim=-1)
        receptive = self.dilation[0] * (self.kernel_size[0] - 1) + 1
        count = max(0, (joined.shape[-1] - receptive) // self.stride[0] + 1)
        # clone avoids retaining the entire input packet's storage through a view.
        state[id(self)] = joined[..., count * self.stride[0]:].detach().clone()
        if count == 0:
            return x.new_empty(x.shape[0], self.out_channels, 0)
        return super().forward(joined[..., :receptive + (count - 1) * self.stride[0]])


class CausalTranspose(nn.ConvTranspose1d):
    """Overlap-add across packets; apply bias once per emitted sample.

    Trimming each packet independently loses the overlap at every boundary.
    The final tail is intentionally dropped, matching the full-pass right trim.
    """

    def __init__(self, incoming, outgoing, stride):
        super().__init__(incoming, outgoing, 2 * stride, stride=stride)
        self.tail_size = stride

    def forward(self, x):
        if not x.shape[-1]:
            return x.new_empty(x.shape[0], self.out_channels, 0)
        return super().forward(x)[..., :-self.tail_size]

    def step(self, x, state: State):
        if not x.shape[-1]:
            return x.new_empty(x.shape[0], self.out_channels, 0)
        result = F.conv_transpose1d(x, self.weight, bias=None, stride=self.stride)
        previous = state.get(id(self))
        if previous is not None:
            result[..., :self.tail_size] += previous
        count = x.shape[-1] * self.stride[0]
        state[id(self)] = result[..., count:].detach().clone()
        result = result[..., :count]
        if self.bias is not None:
            result = result + self.bias[None, :, None]
        return result


class Residual(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = CausalConv(channels, channels, 7, dilation=dilation)
        self.projection = CausalConv(channels, channels, 1)

    def forward(self, x, state=None):
        y = F.elu(_run(self.conv, x, state))
        return x + F.elu(_run(self.projection, y, state))


class FiLM(nn.Module):
    def __init__(self, channels, embedding):
        super().__init__()
        self.scale = nn.Linear(embedding, channels)
        self.bias = nn.Linear(embedding, channels)

    def forward(self, x, speaker):
        return x * self.scale(speaker)[..., None] + self.bias(speaker)[..., None]


def _run(layer, x, state):
    if state is None:
        return layer(x)
    return layer.step(x, state)


class Encoder(nn.Module):
    def __init__(self, scale, embedding):
        super().__init__()
        self.first = CausalConv(1, scale, 7)
        self.blocks = nn.ModuleList()
        channels = scale
        for stride in (2, 4, 5, 8):
            self.blocks.append(nn.ModuleList([
                *(Residual(channels, d) for d in (1, 3, 9)),
                CausalConv(channels, channels * 2, 2 * stride, stride=stride),
            ]))
            channels *= 2
        self.last = CausalConv(channels, embedding, 3)

    def forward(self, x, state=None):
        x = F.elu(_run(self.first, x, state))
        for block in self.blocks:
            for residual in block[:-1]:
                x = residual(x, state)
            x = F.elu(_run(block[-1], x, state))
        return F.elu(_run(self.last, x, state))


class Decoder(nn.Module):
    def __init__(self, scale, embedding):
        super().__init__()
        channels = 16 * scale
        self.first = CausalConv(embedding + 10, channels, 7)
        self.blocks = nn.ModuleList()
        for stride in (8, 5, 4, 2):
            outgoing = channels // 2
            # Paper §2.2.4: speaker FiLM between residual units.
            self.blocks.append(nn.ModuleList([
                CausalTranspose(channels, outgoing, stride),
                *(nn.ModuleList([Residual(outgoing, d), FiLM(outgoing, embedding)])
                  for d in (1, 3, 9)),
            ]))
            channels = outgoing
        self.last = CausalConv(channels, 1, 7)

    def forward(self, x, speaker, state=None):
        x = F.elu(_run(self.first, x, state))
        for block in self.blocks:
            x = _run(block[0], x, state)
            for residual, film in block[1:]:
                x = film(residual(x, state), speaker)
        return _run(self.last, x, state)


class StreamVCNeuralCore(nn.Module):
    """SoundStream-style C=64/32/40, D=64, 320:1 encoder/decoder.

    The 10 side channels stand for 9 YIN outputs + energy. Supplying zeros is
    suitable only for timing/state tests. This class does not load RVC weights.
    """

    frame_samples = 320

    def __init__(self, content_scale=64, speaker_scale=32, decoder_scale=40, embedding=64):
        super().__init__()
        self.embedding = embedding
        self.content = Encoder(content_scale, embedding)
        self.speaker = Encoder(speaker_scale, embedding)
        self.pool_query = nn.Parameter(torch.zeros(embedding))
        self.decoder = Decoder(decoder_scale, embedding)

    def encode_speaker(self, reference):
        latent = self.speaker(reference)
        if not latent.shape[-1]:
            raise ValueError('speaker reference must contain at least 320 samples')
        attention = torch.softmax(torch.einsum('bct,c->bt', latent, self.pool_query), dim=-1)
        return torch.einsum('bct,bt->bc', latent, attention)

    def forward(self, source, side_features, speaker, state=None):
        if source.ndim != 3 or source.shape[1] != 1 or source.shape[-1] % 320:
            raise ValueError('source must have shape (batch, 1, multiples of 320)')
        frames = source.shape[-1] // 320
        if side_features.shape != (source.shape[0], 10, frames):
            raise ValueError('side features must have shape (batch, 10, source frames)')
        if speaker.shape != (source.shape[0], self.embedding):
            raise ValueError('invalid speaker embedding shape')
        if not frames:
            return source.new_empty(source.shape[0], 1, 0)
        return self.decoder(torch.cat((self.content(source, state), side_features), dim=1), speaker, state)

    def new_session(self):
        return NeuralCoreSession(self)


class NeuralCoreSession:
    """One inference stream with explicit reset/close; frames are already 20ms."""

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.state: State = {}
        self.closed = False

    @torch.inference_mode()
    def push(self, source, side_features, speaker):
        if self.closed:
            raise RuntimeError('session is closed; reset before reuse')
        return self.model(source, side_features, speaker, self.state)

    def flush(self):
        # This experiment has no YIN lookahead or training-alignment queue.
        # Transpose tails are the same right trim used by the full-pass model.
        self.state.clear()
        self.closed = True

    @property
    def state_bytes(self):
        return sum(x.numel() * x.element_size() for x in self.state.values())
