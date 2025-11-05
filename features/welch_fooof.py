from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from scipy import signal


def compute_welch(x: np.ndarray, fs: float, bands: Iterable[Tuple[float, float]]) -> torch.Tensor:
    """Compute relative Welch band power for each channel."""
    c, t = x.shape
    freqs, pxx = signal.welch(x, fs=fs, nperseg=int(fs), axis=-1)
    total_power = pxx.sum(-1, keepdims=True) + 1e-12
    band_power = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        band_power.append(pxx[:, mask].sum(-1))
    band_power = np.stack(band_power, axis=-1)
    rel_power = band_power / total_power
    return torch.from_numpy(rel_power.astype(np.float32))


def compute_fooof_aperiodic(x: np.ndarray, fs: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Approximate FOOOF slope and offset via linear regression in log-log space."""
    c, t = x.shape
    freqs, pxx = signal.welch(x, fs=fs, nperseg=int(fs), axis=-1)
    freqs = freqs[1:]
    log_freqs = np.log(freqs + 1e-6)
    log_power = np.log(pxx[:, 1:] + 1e-12)
    slopes = []
    offsets = []
    ones = np.ones_like(log_freqs)
    denom = np.linalg.pinv(np.stack([log_freqs, ones], axis=0).T)
    for ch in range(c):
        coeff = denom @ log_power[ch]
        slope, offset = coeff[0], coeff[1]
        slopes.append(slope)
        offsets.append(offset)
    return (
        torch.tensor(slopes, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(offsets, dtype=torch.float32).unsqueeze(-1),
    )


def precompute_teachers(
    data: Dict[str, np.ndarray],
    fs: float,
    bands: Iterable[Tuple[float, float]],
    output_dir: Path,
) -> None:
    """Pre-compute teacher signals (Welch, FOOOF approximations)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, x in data.items():
        welch = compute_welch(x, fs, bands)
        slope, offset = compute_fooof_aperiodic(x, fs)
        torch.save(
            {
                "welch_rel": welch,
                "fooof_slope": slope,
                "fooof_offset": offset,
            },
            output_dir / f"{key}.pt",
        )


__all__ = ["compute_welch", "compute_fooof_aperiodic", "precompute_teachers"]
