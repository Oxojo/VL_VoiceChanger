"""Opt-in approximate feature reuse for overlapping, noncausal windows.

Only an older prefix is reused. Both sides of the refresh boundary receive
guards (at least the configured context in the streaming adapter). This is
not equivalent to full-window bidirectional inference.
"""

from dataclasses import dataclass, field, replace

import numpy as np
import torch


@dataclass
class FeatureBatch:
    phone: torch.Tensor  # CPU [1, frames, channels], expanded to 100 fps
    f0: np.ndarray       # raw, unshifted F0 at 100 fps
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class CacheStats:
    calls: int = 0
    requested_frames: int = 0
    computed_frames: int = 0
    reused_frames: int = 0
    alignment_misses: int = 0


class StreamingFeatureCache:
    def __init__(self, guard_frames: int = 10):
        if not isinstance(guard_frames, int) or guard_frames < 0:
            raise ValueError("feature cache guard must be a nonnegative frame count")
        self.guard_frames = guard_frames
        self.reset()

    def reset(self):
        self.start = 0
        self.phase = None
        self.f0 = np.empty(0, dtype=np.float32)
        self.phone = None
        self.stats = CacheStats()

    def checkpoint(self):
        # Stored tensors/arrays are replaced, never modified in place.
        return self.start, self.phase, self.f0, self.phone, replace(self.stats)

    def restore(self, checkpoint):
        self.start, self.phase, self.f0, self.phone, self.stats = checkpoint

    @staticmethod
    def _validate(batch, frames):
        if (batch.f0.ndim != 1 or batch.f0.size != frames
                or batch.phone.ndim != 3 or batch.phone.shape[:2] != (1, frames)):
            raise ValueError("feature extractor returned inconsistent frame counts")

    def extract(self, audio_pad, frames, *, start_frame, real_frames, extract):
        phase = start_frame % 2  # preserve the 20 ms ContentVec lattice
        end = self.start + self.f0.size
        prefix = (min(end - start_frame, frames) if self.phone is not None
                  and phase == self.phase and self.start <= start_frame < end else 0)
        # Refresh left guard too, but discard its output. Round down to a
        # whole 20 ms step so upsampled ContentVec pairs never shift by 10 ms.
        skip = max(0, prefix - self.guard_frames)
        skip -= skip % 2
        fresh = extract(audio_pad[skip * 160:], frames - skip)
        self._validate(fresh, frames - skip)
        if skip:
            offset = start_frame - self.start
            discard = prefix - skip
            result = FeatureBatch(
                torch.cat((self.phone[:, offset:offset + prefix], fresh.phone[:, discard:]), dim=1),
                np.concatenate((self.f0[offset:offset + prefix], fresh.f0[discard:])),
                fresh.timings,
            )
            reused = prefix
        else:
            # No extraction work saved: use the full fresh result, not stale
            # features merely to claim a cache hit.
            result, reused = fresh, 0
        self._validate(result, frames)
        self.stats.calls += 1
        self.stats.requested_frames += frames
        self.stats.computed_frames += frames - skip
        self.stats.reused_frames += reused
        if self.phase is not None and phase != self.phase:
            self.stats.alignment_misses += 1

        # Exclude the right guard and reflected end padding from committed
        # features. Retain at most the current source window, not stream history.
        keep = max(0, min(frames, real_frames - self.guard_frames))
        self.start, self.phase = start_frame, phase
        self.f0 = result.f0[:keep].copy()
        self.phone = result.phone[:, :keep].clone()
        return result

    def summary(self):
        s = self.stats
        return {
            "enabled": True, "guard_ms": self.guard_frames * 10,
            "requested_frames": s.requested_frames, "computed_frames": s.computed_frames,
            "reused_frames": s.reused_frames, "alignment_misses": s.alignment_misses,
            "avoided_frames": s.requested_frames - s.computed_frames,
            "retained_frames": self.f0.size,
        }
