from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN2D(nn.Module):
    """2D CNN backbone producing per-time spectral embeddings."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        model_cfg = cfg.get("model", {})
        e_spec = int(model_cfg.get("E_spec", 128))
        self.proj_dim = e_spec
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, e_spec, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(e_spec),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, Tf]
        if x.dim() != 4:
            raise ValueError("Input must be [B, C, F, Tf].")
        b, c, f, tf = x.shape
        x = x.reshape(b * c, 1, f, tf)
        x = self.net(x)
        x = F.adaptive_avg_pool2d(x, (1, tf))
        x = x.squeeze(2).permute(0, 2, 1)  # [B*C, Tf, E_spec]
        x = x.reshape(b, c, tf, self.proj_dim).permute(0, 2, 1, 3)
        assert x.shape == (b, tf, c, self.proj_dim)
        return x


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        return torch.einsum("ij,bcj->bci", adj, h)


class GNN(nn.Module):
    """Time-shared GCN backbone."""

    def __init__(self, cfg: dict, in_dim: int = 8) -> None:
        super().__init__()
        model_cfg = cfg.get("model", {})
        e_graph = int(model_cfg.get("E_graph", 64))
        hidden = max(e_graph // 2, 32)
        self.layers = nn.ModuleList(
            [
                GCNLayer(in_dim, hidden),
                GCNLayer(hidden, e_graph),
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden), nn.LayerNorm(e_graph)])

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: [B, Tf, C, d0]
        if x.dim() != 4:
            raise ValueError("Node features must be [B, Tf, C, d].")
        b, tf, c, d = x.shape
        adj = adjacency.to(x.device).float()
        if adj.shape[0] != c:
            raise ValueError("Adjacency channel mismatch.")
        adj = adj + torch.eye(c, device=adj.device)
        deg = adj.sum(-1)
        deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
        norm_adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        h = x
        for idx, layer in enumerate(self.layers):
            h_flat = h.reshape(b * tf, c, -1)
            h_out = layer(h_flat, norm_adj)
            h_out = torch.relu(h_out)
            if idx < len(self.layers) - 1:
                h = self.norms[idx](h_out)
                h = h.reshape(b, tf, c, -1)
            else:
                h = h_out.reshape(b, tf, c, -1)
        h = self.norms[-1](h)
        assert h.shape[0] == b and h.shape[1] == tf and h.shape[2] == c
        return h


class TemporalTransformer(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        model_cfg = cfg.get("model", {})
        transformer_cfg = model_cfg.get("transformer", {})
        self.dim = int(transformer_cfg.get("dim", 256))
        depth = int(transformer_cfg.get("depth", 3))
        heads = int(transformer_cfg.get("heads", 6))
        self.input_proj = nn.Linear(model_cfg.get("E_spec", 128) + model_cfg.get("E_graph", 64) + model_cfg.get("E_cfc", 32), self.dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls_proj = nn.Linear(model_cfg.get("E_spec", 128) + model_cfg.get("E_graph", 64) + model_cfg.get("E_cfc", 32), self.dim)

    def forward(self, x: torch.Tensor, cls_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, Tf, E]
        if x.dim() != 3:
            raise ValueError("Temporal input must be [B, Tf, E].")
        b, tf, e = x.shape
        h = self.input_proj(x)
        pos = self._positional_encoding(tf, self.dim, device=x.device)
        h = h + pos
        if cls_init is None:
            cls_token = torch.zeros(b, 1, self.dim, device=x.device)
        else:
            cls_token = self.cls_proj(cls_init).unsqueeze(1)
        tokens = torch.cat([cls_token, h], dim=1)
        encoded = self.encoder(tokens)
        cls = encoded[:, 0]
        return cls

    def _positional_encoding(self, length: int, dim: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(length, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


__all__ = ["CNN2D", "GNN", "TemporalTransformer"]
