from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from utils.stft import compute_stft


@dataclass
class EEGSample:
    x_raw: torch.Tensor
    center_id: int
    subject_id: str
    y: int
    welch_rel: Optional[torch.Tensor] = None
    fooof_slope: Optional[torch.Tensor] = None
    fooof_offset: Optional[torch.Tensor] = None
    pac_ref: Optional[torch.Tensor] = None


class EEGDataset(Dataset):
    """Dataset that performs segmentation, STFT and feature alignment on-the-fly.

    The dataset expects a list of :class:`EEGSample` or dictionaries with matching keys.
    Optional teacher targets are transparently forwarded to the model.
    """

    def __init__(
        self,
        samples: Sequence[EEGSample | Dict],
        stft_cfg: Dict,
        bands: Sequence[Sequence[float]],
        graph_path: Optional[Path] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        node_feature_dim: int = 8,
    ) -> None:
        self.samples: List[EEGSample] = [self._ensure_sample(s) for s in samples]
        self.transform = transform
        self.window = int(stft_cfg["window"])
        self.hop = int(stft_cfg["hop"])
        self.nfft = int(stft_cfg["nfft"])
        self.bands = torch.tensor(bands, dtype=torch.float32)
        self.node_feature_dim = node_feature_dim
        self.graph_path = graph_path
        if graph_path is not None:
            graph_data = torch.load(graph_path, map_location="cpu")
            self.A = graph_data["A"].float()
            self.L = graph_data.get("L")
            if self.L is not None:
                self.L = self.L.float()
        else:
            self.A = None
            self.L = None

    @staticmethod
    def _ensure_sample(sample: EEGSample | Dict) -> EEGSample:
        if isinstance(sample, EEGSample):
            return sample
        return EEGSample(
            x_raw=sample["x_raw"],
            center_id=int(sample.get("center_id", 0)),
            subject_id=str(sample.get("subject_id", "")),
            y=int(sample["y"]),
            welch_rel=sample.get("welch_rel"),
            fooof_slope=sample.get("fooof_slope"),
            fooof_offset=sample.get("fooof_offset"),
            pac_ref=sample.get("pac_ref"),
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def _compute_node_features(self, x: torch.Tensor, tf: int) -> torch.Tensor:
        # x: [C, T]
        c, t = x.shape
        window = self.window
        hop = self.hop
        if t < window:
            raise ValueError("Time dimension smaller than STFT window.")
        # Sliding RMS and mean absolute value
        unfold = x.unfold(-1, window, hop)  # [C, Tf, window]
        assert unfold.shape[1] == tf, (unfold.shape, tf)
        rms = torch.sqrt((unfold.pow(2).mean(-1) + 1e-6))
        mav = unfold.abs().mean(-1)
        feats = [rms, mav]
        if self.node_feature_dim > 2:
            # basic higher order stats
            z = unfold - unfold.mean(-1, keepdim=True)
            kurtosis = (z.pow(4).mean(-1) / (z.pow(2).mean(-1).pow(2) + 1e-6))
            feats.append(kurtosis)
        node_feat = torch.stack(feats, dim=-1)  # [C, Tf, d]
        d = node_feat.shape[-1]
        if d < self.node_feature_dim:
            pad = self.node_feature_dim - d
            node_feat = torch.nn.functional.pad(node_feat, (0, pad))
        elif d > self.node_feature_dim:
            node_feat = node_feat[..., : self.node_feature_dim]
        node_feat = node_feat.permute(1, 0, 2)  # [Tf, C, d]
        return node_feat

    def __getitem__(self, index: int) -> Dict:  # type: ignore[override]
        sample = self.samples[index]
        x_raw = sample.x_raw.clone()
        if self.transform is not None:
            x_raw = self.transform(x_raw)
        if x_raw.dim() != 2:
            raise ValueError("x_raw must be [C, T].")
        s_power, freqs, tf = self._stft_features(x_raw)
        node_feat = self._compute_node_features(x_raw, tf)
        out: Dict[str, torch.Tensor | int | str] = {
            "S_power": s_power,
            "X_node": node_feat,
            "y": torch.tensor(sample.y, dtype=torch.long),
            "center": torch.tensor(sample.center_id, dtype=torch.long),
            "x_raw": x_raw,
        }
        if self.A is not None:
            out["A"] = self.A
        if self.L is not None:
            out["L"] = self.L
        else:
            out["L"] = torch.zeros(0)
        if sample.welch_rel is not None:
            out["welch_rel"] = sample.welch_rel
        if sample.fooof_slope is not None:
            out["fooof_slope"] = sample.fooof_slope
        if sample.fooof_offset is not None:
            out["fooof_offset"] = sample.fooof_offset
        if sample.pac_ref is not None:
            out["pac_ref"] = sample.pac_ref
        out["freqs"] = freqs
        out["subject_id"] = sample.subject_id
        return out

    def _stft_features(self, x_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        stft = compute_stft(x_raw, window=self.window, hop=self.hop, nfft=self.nfft)
        # stft: [C, F, Tf]
        c, f, tf = stft.shape
        power = stft.abs().pow(2)
        log_power = torch.log(power + 1e-6)
        assert log_power.shape == (c, f, tf), log_power.shape
        freqs = torch.linspace(0, math.pi, f)
        return log_power, freqs, tf
