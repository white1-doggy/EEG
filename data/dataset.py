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
    """Container for individual EEG samples.

    ``x_raw`` may be omitted if ``path`` is provided. In that case the dataset will
    lazily load the raw signal from disk when the sample is accessed.
    """

    x_raw: Optional[torch.Tensor] = None
    path: Optional[Path] = None
    center_id: int = 0
    subject_id: str = ""
    y: int = 0
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
        fs: int = 250,
        seg_len_s: float = 8.0,
        seg_stride_s: Optional[float] = None,
    ) -> None:
        self.samples: List[EEGSample] = [self._ensure_sample(s) for s in samples]
        self.transform = transform
        self.window = int(stft_cfg["window"])
        self.hop = int(stft_cfg["hop"])
        self.nfft = int(stft_cfg["nfft"])
        self.bands = torch.tensor(bands, dtype=torch.float32)
        self.node_feature_dim = node_feature_dim
        self.graph_path = graph_path
        self.fs = fs
        self.seg_len = int(round(fs * seg_len_s))
        if self.seg_len <= 0:
            raise ValueError("Segment length must be positive.")
        stride_seconds = seg_stride_s if seg_stride_s is not None else seg_len_s
        self.seg_stride = max(1, int(round(fs * stride_seconds)))
        if graph_path is not None:
            graph_data = torch.load(graph_path, map_location="cpu")
            self.A = graph_data["A"].float()
            self.L = graph_data.get("L")
            if self.L is not None:
                self.L = self.L.float()
        else:
            self.A = None
            self.L = None
        self._segments: List[tuple[int, int]] = self._prepare_segments()

    @staticmethod
    def _ensure_sample(sample: EEGSample | Dict) -> EEGSample:
        if isinstance(sample, EEGSample):
            return sample
        tensor = sample.get("x_raw")
        if tensor is not None and not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        path = sample.get("path") or sample.get("file") or sample.get("source")
        return EEGSample(
            x_raw=tensor,
            path=Path(path) if path is not None else None,
            center_id=int(sample.get("center_id", 0)),
            subject_id=str(sample.get("subject_id", "")),
            y=int(sample.get("y", 0)),
            welch_rel=sample.get("welch_rel"),
            fooof_slope=sample.get("fooof_slope"),
            fooof_offset=sample.get("fooof_offset"),
            pac_ref=sample.get("pac_ref"),
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._segments)

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
        sample_idx, start = self._segments[index]
        sample = self.samples[sample_idx]
        x_raw_full = self._load_x_raw(sample)
        end = start + self.seg_len
        segment = x_raw_full[:, start:end]
        if segment.shape[-1] < self.seg_len:
            pad = self.seg_len - segment.shape[-1]
            segment = torch.nn.functional.pad(segment, (0, pad))
        if self.transform is not None:
            segment = self.transform(segment)
        x_raw = segment
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
        out["segment_start"] = torch.tensor(start, dtype=torch.long)
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

    def _prepare_segments(self) -> List[tuple[int, int]]:
        segments: List[tuple[int, int]] = []
        for idx, sample in enumerate(self.samples):
            length = self._probe_length(sample)
            if length <= self.seg_len:
                segments.append((idx, 0))
                continue
            start = 0
            last_start = None
            while start + self.seg_len <= length:
                segments.append((idx, start))
                last_start = start
                start += self.seg_stride
            tail_start = max(0, length - self.seg_len)
            if last_start is None or tail_start != last_start:
                segments.append((idx, tail_start))
        if not segments:
            raise RuntimeError("No segments could be generated from the provided samples.")
        return segments

    def _probe_length(self, sample: EEGSample) -> int:
        if sample.x_raw is not None:
            return sample.x_raw.shape[-1]
        if sample.path is None:
            raise ValueError("Sample has neither in-memory data nor file path.")
        suffix = sample.path.suffix.lower()
        if suffix == ".set":
            try:
                import mne
            except ImportError as exc:  # pragma: no cover - informative failure path
                raise ImportError(
                    "Reading .set files requires the 'mne' package. Install it via 'pip install mne'."
                ) from exc

            raw = mne.io.read_raw_eeglab(str(sample.path), preload=False, verbose=False)
            length = int(raw.n_times)
            return length
        raise ValueError(f"Unsupported EEG file format: {sample.path}")

    def _stft_features(self, x_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        stft = compute_stft(x_raw, window=self.window, hop=self.hop, nfft=self.nfft)
        # stft: [C, F, Tf]
        c, f, tf = stft.shape
        power = stft.abs().pow(2)
        log_power = torch.log(power + 1e-6)
        assert log_power.shape == (c, f, tf), log_power.shape
        freqs = torch.linspace(0, math.pi, f)
        return log_power, freqs, tf

    def _load_x_raw(self, sample: EEGSample) -> torch.Tensor:
        if sample.x_raw is not None:
            return sample.x_raw.clone()
        if sample.path is None:
            raise ValueError("Sample has neither in-memory data nor file path.")
        suffix = sample.path.suffix.lower()
        if suffix == ".set":
            return self._load_from_set(sample)
        raise ValueError(f"Unsupported EEG file format: {sample.path}")

    def _load_from_set(self, sample: EEGSample) -> torch.Tensor:
        try:
            import mne
        except ImportError as exc:  # pragma: no cover - informative failure path
            raise ImportError(
                "Reading .set files requires the 'mne' package. Install it via 'pip install mne'."
            ) from exc

        if sample.path is None:
            raise ValueError("Cannot load .set file without a valid path")
        raw = mne.io.read_raw_eeglab(str(sample.path), preload=True, verbose=False)
        data = raw.get_data()  # [C, T]
        tensor = torch.from_numpy(data).float()
        # Cache loaded tensor to avoid reloading on subsequent epochs.
        sample.x_raw = tensor
        return tensor.clone()
