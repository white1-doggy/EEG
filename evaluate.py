from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from data.dataset import EEGDataset
from models.model import FullModel
from utils.metrics import confidence_interval, macro_auroc, macro_f1
from utils.stft import compute_stft


def load_model(cfg: Dict, checkpoint: Path, device: torch.device) -> FullModel:
    model = FullModel(cfg)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_dataset(cfg: Dict, data_path: Path) -> EEGDataset:
    stft_cfg = cfg["stft"]
    bands = cfg["bands"]
    samples = torch.load(data_path)
    graph_path = cfg.get("graph_path")
    graph_path = Path(graph_path) if graph_path else None
    return EEGDataset(samples, stft_cfg, bands, graph_path=graph_path)


def run_model(model: FullModel, batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        return model(batch)


def evaluate_loso(model: FullModel, dataset: EEGDataset, device: torch.device) -> Dict[str, float]:
    centers = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        centers[sample.center_id].append(idx)
    scores = []
    for center_id, indices in centers.items():
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=32)
        preds = []
        targets = []
        probs = []
        for batch in loader:
            out = run_model(model, batch, device)
            logits = out["logits"]
            preds.append(logits.argmax(-1).cpu())
            targets.append(batch["y"].cpu())
            probs.append(torch.softmax(logits, dim=-1).cpu())
        y_true = torch.cat(targets).numpy()
        y_pred = torch.cat(preds).numpy()
        y_prob = torch.cat(probs).numpy()
        f1 = macro_f1(y_true, y_pred)
        auroc = macro_auroc(y_true, y_prob)
        scores.append(auroc)
        print(f"Center {center_id} - macro-F1: {f1:.3f}, AUROC: {auroc:.3f}")
    mean_auroc = np.mean(scores)
    ci_low, ci_high = confidence_interval(np.array(scores))
    return {"mean_auroc": mean_auroc, "ci_low": ci_low, "ci_high": ci_high}


def band_mask(s_power: torch.Tensor, active: Sequence[int]) -> torch.Tensor:
    parts = torch.chunk(s_power, chunks=5, dim=2)
    masked = [parts[i] if i in active else torch.zeros_like(parts[i]) for i in range(len(parts))]
    return torch.cat(masked, dim=2)


def fidelity_curves(model: FullModel, dataset: EEGDataset, device: torch.device) -> Dict[str, List[float]]:
    loader = DataLoader(dataset, batch_size=16)
    deletion_scores = []
    insertion_scores = []
    for step in range(6):
        deletion_preds = []
        deletion_targets = []
        insertion_preds = []
        insertion_targets = []
        for batch in loader:
            base_out = run_model(model, batch, device)
            band_w = base_out["band_w"].mean(1)
            order = torch.argsort(band_w, dim=-1, descending=True)
            s_power = batch["S_power"].clone()
            b, c, f, tf = s_power.shape
            if step > 0:
                active_del = [[idx for idx in range(5) if idx not in order[b_idx, :step].tolist()] for b_idx in range(b)]
            else:
                active_del = [list(range(5)) for _ in range(b)]
            del_power = torch.stack([band_mask(s_power[i], active_del[i]) for i in range(b)], dim=0)
            batch_del = dict(batch)
            batch_del["S_power"] = del_power
            del_out = run_model(model, batch_del, device)
            logits = del_out["logits"]
            deletion_preds.append(logits.argmax(-1).cpu())
            deletion_targets.append(batch["y"].cpu())

            if step == 0:
                active_ins = [list() for _ in range(b)]
            else:
                active_ins = [order[b_idx, :step].tolist() for b_idx in range(b)]
            ins_power = torch.stack([band_mask(s_power[i], active_ins[i]) for i in range(b)], dim=0)
            batch_ins = dict(batch)
            batch_ins["S_power"] = ins_power
            ins_out = run_model(model, batch_ins, device)
            logits_ins = ins_out["logits"]
            insertion_preds.append(logits_ins.argmax(-1).cpu())
            insertion_targets.append(batch["y"].cpu())
        y_true = torch.cat(deletion_targets).numpy()
        y_pred = torch.cat(deletion_preds).numpy()
        deletion_scores.append(macro_f1(y_true, y_pred))
        y_true = torch.cat(insertion_targets).numpy()
        y_pred = torch.cat(insertion_preds).numpy()
        insertion_scores.append(macro_f1(y_true, y_pred))
    return {"deletion": deletion_scores, "insertion": insertion_scores}


def phase_randomization(model: FullModel, dataset: EEGDataset, cfg: Dict, device: torch.device) -> float:
    loader = DataLoader(dataset, batch_size=1)
    deltas = []
    for batch in loader:
        baseline = run_model(model, batch, device)
        base_prob = torch.softmax(baseline["logits"], dim=-1)
        x_raw = batch["x_raw"].squeeze(0)
        fft = torch.fft.rfft(x_raw, dim=-1)
        angles = torch.rand_like(fft.real) * 2 * torch.pi
        random_phase = torch.cos(angles) + 1j * torch.sin(angles)
        shuffled = torch.fft.irfft(torch.abs(fft) * random_phase, n=x_raw.shape[-1], dim=-1)
        stft = compute_stft(shuffled, window=cfg["stft"]["window"], hop=cfg["stft"]["hop"], nfft=cfg["stft"]["nfft"])
        s_power = torch.log(stft.abs().pow(2) + 1e-6)
        batch_rand = dict(batch)
        batch_rand["S_power"] = s_power.unsqueeze(0)
        rand_out = run_model(model, batch_rand, device)
        rand_prob = torch.softmax(rand_out["logits"], dim=-1)
        delta = (base_prob - rand_prob).abs().mean().item()
        deltas.append(delta)
    return float(np.mean(deltas))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EEG addiction model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["loso", "fidelity", "phase"], default="loso")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(cfg, Path(args.data))
    model = load_model(cfg, Path(args.checkpoint), device)

    if args.task == "loso":
        result = evaluate_loso(model, dataset, device)
        print(result)
    elif args.task == "fidelity":
        result = fidelity_curves(model, dataset, device)
        print(result)
    else:
        delta = phase_randomization(model, dataset, cfg, device)
        print({"delta_auroc": delta})


if __name__ == "__main__":
    main()
