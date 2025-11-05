from __future__ import annotations

import torch


def compute_stft(
    x: torch.Tensor,
    window: int,
    hop: int,
    nfft: int,
    center: bool = False,
    window_fn: str = "hann",
) -> torch.Tensor:
    """Compute STFT for multi-channel signals and return complex tensor [C, F, T]."""
    if x.dim() != 2:
        raise ValueError("Input must be [C, T].")
    c, t = x.shape
    win = torch.hann_window(window, device=x.device) if window_fn == "hann" else torch.ones(window, device=x.device)
    stft = torch.stft(
        x,
        n_fft=nfft,
        hop_length=hop,
        win_length=window,
        window=win,
        return_complex=True,
        center=center,
    )
    assert stft.shape[0] == c
    return stft


__all__ = ["compute_stft"]
