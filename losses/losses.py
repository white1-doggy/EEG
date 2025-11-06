from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def compute_losses(output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    loss_cfg = cfg.get("loss_weights", {})
    y = batch["y"]
    logits = output["logits"]
    class_weights = cfg.get("class_weights")
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, device=logits.device, dtype=logits.dtype)
    l_cls = F.cross_entropy(logits, y, weight=weight_tensor)

    if "welch_rel" in batch:
        l_band = F.mse_loss(output["band_w"], batch["welch_rel"].detach())
    else:
        l_band = torch.tensor(0.0, device=logits.device)

    if "fooof_slope" in batch and "fooof_offset" in batch:
        l_aper = F.mse_loss(output["slope"], batch["fooof_slope"].detach()) + F.mse_loss(
            output["offset"], batch["fooof_offset"].detach()
        )
    else:
        l_aper = torch.tensor(0.0, device=logits.device)

    if "pac_ref" in batch:
        l_cfc = F.mse_loss(output["cfc_map"], batch["pac_ref"].detach())
    else:
        l_cfc = torch.tensor(0.0, device=logits.device)

    probs = output["micro_probs"]
    l_sparse = probs.abs().mean()
    l_smooth = (probs[:, 1:] - probs[:, :-1]).pow(2).mean()
    entropy = probs.clamp(1e-6, 1 - 1e-6)
    l_entropy = -(entropy * entropy.log()).mean()

    total = (
        l_cls
        + loss_cfg.get("lb", 0.0) * l_band
        + loss_cfg.get("la", 0.0) * l_aper
        + loss_cfg.get("lc", 0.0) * l_cfc
        + loss_cfg.get("lm1", 0.0) * l_sparse
        + loss_cfg.get("lm2", 0.0) * l_smooth
        + loss_cfg.get("lm3", 0.0) * l_entropy
    )
    logs = {
        "L_cls": l_cls.detach(),
        "L_band": l_band.detach(),
        "L_aper": l_aper.detach(),
        "L_cfc": l_cfc.detach(),
        "L_sparse": l_sparse.detach(),
        "L_smooth": l_smooth.detach(),
        "L_entropy": l_entropy.detach(),
    }
    return total, logs


__all__ = ["compute_losses"]
