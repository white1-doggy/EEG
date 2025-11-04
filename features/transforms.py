from __future__ import annotations

import random
from typing import Callable, Sequence

import torch


class Compose:
    def __init__(self, transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class TimeShift:
    def __init__(self, max_shift: int) -> None:
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_shift <= 0:
            return x
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=-1)


class ChannelDrop:
    def __init__(self, drop_prob: float) -> None:
        self.drop_prob = drop_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0:
            return x
        mask = torch.rand(x.shape[0], device=x.device) > self.drop_prob
        return x * mask.unsqueeze(-1)


class RandomBandStop:
    def __init__(self, fs: float, width: float = 2.0, min_freq: float = 1.0, max_freq: float | None = None) -> None:
        self.fs = fs
        self.width = width
        self.min_freq = min_freq
        self.max_freq = max_freq if max_freq is not None else fs / 2 - width

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        freq = random.uniform(self.min_freq, self.max_freq)
        w0 = freq / (self.fs / 2)
        bw = self.width / (self.fs / 2)
        if bw <= 0:
            return x
        # simple notch using FFT mask
        fft = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.fs)
        mask = (freqs < freq - self.width / 2) | (freqs > freq + self.width / 2)
        fft = fft * mask.to(x.device)
        return torch.fft.irfft(fft, n=x.shape[-1], dim=-1)


class SpecAugment:
    def __init__(self, max_time_mask: int, max_freq_mask: int) -> None:
        self.max_time_mask = max_time_mask
        self.max_freq_mask = max_freq_mask

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: [C, F, T]
        c, f, t = spec.shape
        if self.max_time_mask > 0:
            len_t = random.randint(0, self.max_time_mask)
            start = random.randint(0, max(t - len_t, 0))
            spec[:, :, start : start + len_t] = 0
        if self.max_freq_mask > 0:
            len_f = random.randint(0, self.max_freq_mask)
            start = random.randint(0, max(f - len_f, 0))
            spec[:, start : start + len_f, :] = 0
        return spec


__all__ = [
    "Compose",
    "TimeShift",
    "ChannelDrop",
    "RandomBandStop",
    "SpecAugment",
]
