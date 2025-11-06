from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BandGateHead(nn.Module):
    def __init__(self, cfg: dict, num_bands: int = 5) -> None:
        super().__init__()
        model_cfg = cfg.get("model", {})
        e_spec = int(model_cfg.get("E_spec", 128))
        init_centers = torch.tensor([2.5, 6.0, 10.0, 20.0, 37.5]).log()
        self.mu = nn.Parameter(init_centers[:num_bands])
        self.sigma = nn.Parameter(torch.full((num_bands,), 0.3))
        self.gamma = nn.Linear(num_bands, e_spec)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("gate_scale", torch.tensor(1.0))

    def forward(self, log_power: torch.Tensor, h_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, f = log_power.shape
        b2, tf, c2, e_spec = h_spec.shape
        assert b == b2 and c == c2
        freqs = torch.linspace(1.0, f, f, device=log_power.device)
        centers = self.mu.exp()
        widths = torch.clamp(self.sigma, 0.1, 1.0)
        freq_mat = freqs.unsqueeze(-1)
        gaussian = torch.exp(-0.5 * ((torch.log(freq_mat + 1e-6) - centers) / widths) ** 2)
        gaussian = gaussian / (gaussian.sum(0, keepdim=True) + 1e-6)
        band_pow = torch.matmul(log_power, gaussian)
        band_weights = F.softmax(band_pow, dim=-1)
        g = self.gamma(band_weights)
        gate = 1 + self.alpha.sigmoid() * self.gate_scale * torch.sigmoid(g).unsqueeze(1)
        h_tilde = h_spec * gate
        return band_weights, h_tilde


class AperiodicHead(nn.Module):
    def __init__(self, cfg: dict, hidden: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(128, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, log_power: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, f = log_power.shape
        pooled = log_power.reshape(b * c, f)
        if pooled.shape[-1] != 128:
            pooled = F.interpolate(pooled.unsqueeze(1), size=128, mode="linear", align_corners=False).squeeze(1)
        slope_offset = self.mlp(pooled)
        slope_offset = slope_offset.reshape(b, c, 2)
        slope = slope_offset[..., :1]
        offset = slope_offset[..., 1:]
        return slope, offset


class CFCHead(nn.Module):
    def __init__(self, cfg: dict, low_bands: Optional[Tuple[int, ...]] = None, high_bands: Optional[Tuple[int, ...]] = None) -> None:
        super().__init__()
        model_cfg = cfg.get("model", {})
        self.e_cfc = int(model_cfg.get("E_cfc", 32))
        self.low_bands = low_bands or (1, 2)
        self.high_bands = high_bands or (3, 4)
        self.linear = nn.Linear(len(self.low_bands) * len(self.high_bands), self.e_cfc)

    def forward(self, s_power: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, f, tf = s_power.shape
        bands = torch.chunk(s_power, chunks=5, dim=2)
        low = torch.stack([bands[i].mean(2) for i in self.low_bands], dim=-1)
        high = torch.stack([bands[i].mean(2) for i in self.high_bands], dim=-1)
        low_phase = torch.angle(torch.complex(low.cos(), low.sin()))
        high_amp = high.exp()
        cfc_values = []
        for i in range(len(self.low_bands)):
            for j in range(len(self.high_bands)):
                value = (high_amp[..., j] * torch.cos(low_phase[..., i])).mean(-1)
                cfc_values.append(value)
        cfc_map = torch.stack(cfc_values, dim=-1).reshape(b, c, len(self.low_bands), len(self.high_bands))
        features = self.linear(cfc_map.flatten(2))
        h = features.unsqueeze(1).repeat(1, tf, 1, 1)
        return cfc_map, h


class MicrostateHead(nn.Module):
    def __init__(self, cfg: dict, k: int = 4) -> None:
        super().__init__()
        model_cfg = cfg.get("model", {})
        e_total = model_cfg.get("E_spec", 128) + model_cfg.get("E_graph", 64) + model_cfg.get("E_cfc", 32)
        self.k = k
        self.linear = nn.Linear(e_total, k)
        self.tau_min = 0.5
        self.tau_max = 1.0
        self.register_buffer("step", torch.tensor(0.0))

    def forward(self, z: torch.Tensor, anneal_steps: int = 1000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: [B, Tf, E]
        b, tf, e = z.shape
        tau = self._temperature(anneal_steps)
        logits = self.linear(z)
        probs = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        probs_norm = probs / (probs.sum(1, keepdim=True) + 1e-6)
        cls_init = torch.einsum("bte,btk->be", z, probs_norm)
        transition = torch.einsum("btk,btj->bkj", probs[:, 1:], probs[:, :-1])
        transition = transition / (transition.sum(-1, keepdim=True) + 1e-6)
        dwell = probs.mean(1)
        self.step.data = torch.clamp(self.step.data + 1.0, max=float(anneal_steps))
        return probs, transition, dwell, cls_init

    def _temperature(self, anneal_steps: int) -> float:
        step = min(self.step.item(), float(anneal_steps))
        ratio = 1 - step / max(anneal_steps, 1)
        return self.tau_min + (self.tau_max - self.tau_min) * ratio


__all__ = ["BandGateHead", "AperiodicHead", "CFCHead", "MicrostateHead"]
