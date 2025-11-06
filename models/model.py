from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.backbones import CNN2D, GNN, TemporalTransformer
from models.heads import AperiodicHead, BandGateHead, CFCHead, MicrostateHead


class FullModel(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.cnn = CNN2D(cfg)
        self.gnn = GNN(cfg)
        self.band = BandGateHead(cfg)
        self.aper = AperiodicHead(cfg)
        self.cfc = CFCHead(cfg)
        self.micro = MicrostateHead(cfg)
        self.temporal = TemporalTransformer(cfg)
        model_cfg = cfg.get("model", {})
        e_total = model_cfg.get("E_spec", 128) + model_cfg.get("E_graph", 64) + model_cfg.get("E_cfc", 32)
        self.classifier = nn.Linear(self.temporal.dim, 4)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        s_power = batch["S_power"]
        x_node = batch["X_node"]
        adjacency = batch.get("A")
        if adjacency is None or adjacency.numel() == 0:
            c = s_power.shape[1]
            adjacency = torch.eye(c, device=s_power.device)
        else:
            adjacency = adjacency.to(s_power.device)
        h_spec = self.cnn(s_power)
        h_graph = self.gnn(x_node, adjacency)
        logp = s_power.mean(-1)
        slope, offset = self.aper(logp)
        band_w, h_spec_tilde = self.band(logp, h_spec)
        cfc_map, h_cfc = self.cfc(s_power)
        h = torch.cat([h_spec_tilde, h_graph, h_cfc], dim=-1)
        b, tf, c, e = h.shape
        assert h_graph.shape[:3] == (b, tf, c)
        z = h.mean(2)
        probs, transition, dwell, cls_init = self.micro(z)
        temporal_feat = self.temporal(z, cls_init=cls_init)
        logits = self.classifier(temporal_feat)
        return {
            "logits": logits,
            "band_w": band_w,
            "slope": slope,
            "offset": offset,
            "cfc_map": cfc_map,
            "micro_probs": probs,
            "trans": transition,
            "dwell": dwell,
        }


__all__ = ["FullModel"]
