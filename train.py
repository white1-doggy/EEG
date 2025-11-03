from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import yaml

from data.dataset import EEGDataset
from losses.losses import compute_losses
from models.model import FullModel
from utils.metrics import macro_auroc, macro_f1


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, num_domains: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_samples(path: Path) -> Iterable[Dict]:
    data = torch.load(path)
    if isinstance(data, dict) and "samples" in data:
        return data["samples"]
    return data


def _describe_sample(sample: Any, index: int) -> str:
    if isinstance(sample, dict):
        for key in ("sample_id", "id", "subject_id", "file", "path", "name"):
            value = sample.get(key)
            if value:
                return str(value)
    else:
        for key in ("sample_id", "id", "subject_id", "file", "path", "name"):
            if hasattr(sample, key):
                value = getattr(sample, key)
                if value:
                    return str(value)
    return str(index)


def _compute_split_lengths(num_samples: int, ratios: Tuple[float, float, float]) -> List[int]:
    total = sum(ratios)
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    base = [int(num_samples * r / total) for r in ratios]
    diff = num_samples - sum(base)
    for i in range(diff):
        base[i % len(base)] += 1
    return base


def ensure_split_file(
    samples: List[Any],
    split_path: Path,
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, List[int]]:
    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        split_data = data.get("splits", data)
        splits: Dict[str, List[int]] = {}
        for key in ("train", "val", "test"):
            entries = split_data.get(key, [])
            indices = []
            for entry in entries:
                if isinstance(entry, dict) and "index" in entry:
                    indices.append(int(entry["index"]))
                else:
                    indices.append(int(entry))
            splits[key] = indices
        return splits

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    lengths = _compute_split_lengths(len(samples), ratios)
    names = ("train", "val", "test")
    splits = {}
    cursor = 0
    for name, length in zip(names, lengths):
        splits[name] = indices[cursor : cursor + length]
        cursor += length

    split_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "ratios": {
            "train": ratios[0],
            "val": ratios[1],
            "test": ratios[2],
        },
        "seed": seed,
        "num_samples": len(samples),
    }
    serialisable = {
        "meta": meta,
        "splits": {
            name: [
                {"index": idx, "id": _describe_sample(samples[idx], idx)} for idx in sorted(idxs)
            ]
            for name, idxs in splits.items()
        },
    }
    with open(split_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(serialisable, f)
    print(f"Saved new split file to {split_path} with lengths {[len(splits[n]) for n in names]}")
    return splits


def build_dataloaders(cfg: Dict, args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    stft_cfg = cfg["stft"]
    bands = cfg["bands"]
    graph_path = cfg.get("graph_path")
    graph_path = Path(graph_path) if graph_path else None
    data_file = Path(args.data_file)
    samples = list(load_samples(data_file))
    split_path = Path(args.split_file) if args.split_file else data_file.with_suffix(".splits.yaml")
    ratios = tuple(args.split_ratios)
    if len(ratios) != 3:
        raise ValueError("--split-ratios must contain exactly three values (train, val, test).")
    splits = ensure_split_file(samples, split_path, ratios, args.split_seed)
    train_samples = [samples[i] for i in splits.get("train", [])]
    val_indices = splits.get("val", [])
    val_samples = [samples[i] for i in val_indices]
    train_dataset = EEGDataset(train_samples, stft_cfg, bands, graph_path=graph_path)
    val_dataset = EEGDataset(val_samples, stft_cfg, bands, graph_path=graph_path) if val_samples else None
    batch_size = cfg["train"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def create_optimizer(model: nn.Module, cfg: Dict) -> optim.Optimizer:
    train_cfg = cfg["train"]
    return optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=0.01)


def create_scheduler(optimizer: optim.Optimizer, train_loader: DataLoader, cfg: Dict):
    max_epochs = cfg["train"]["max_epochs"]
    total_steps = max_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds = []
    targets = []
    probs = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_to_device(batch, device)
            out = model(batch)
            logits = out["logits"]
            preds.append(logits.argmax(dim=-1).cpu())
            targets.append(batch["y"].cpu())
            probs.append(torch.softmax(logits, dim=-1).cpu())
    y_true = torch.cat(targets).numpy()
    y_pred = torch.cat(preds).numpy()
    y_prob = torch.cat(probs).numpy()
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "macro_auroc": macro_auroc(y_true, y_prob),
    }


def train(cfg: Dict, args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(cfg, args)
    model = FullModel(cfg).to(device)
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, train_loader, cfg)
    scaler = amp.GradScaler(enabled=args.amp)
    use_dann = cfg["train"].get("use_dann", False)
    dann_lambda = cfg["train"].get("dann_lambda", 0.5)
    domain_classifier = None
    if use_dann:
        num_domains = int(cfg["train"].get("num_centers", 1))
        domain_classifier = DomainDiscriminator(model.temporal.dim, num_domains).to(device)
        domain_optimizer = optim.Adam(domain_classifier.parameters(), lr=cfg["train"]["lr"])
    max_epochs = cfg["train"]["max_epochs"]
    gate_warmup = cfg["train"].get("gate_warmup_epochs", 0)

    global_step = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        if gate_warmup > 0:
            scale = min(1.0, epoch / gate_warmup)
            model.band.gate_scale.fill_(scale)
        for batch in train_loader:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=args.amp):
                out = model(batch)
                loss, logs = compute_losses(out, batch, cfg)
                if use_dann and domain_classifier is not None:
                    feats = out["logits"].detach()
                    rev = grad_reverse(feats, lambd=min(1.0, global_step / (max_epochs * len(train_loader))) * dann_lambda)
                    domain_logits = domain_classifier(rev)
                    domain_loss = F.cross_entropy(domain_logits, batch["center"])
                    loss = loss + domain_loss
            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if use_dann and domain_classifier is not None:
                domain_optimizer.zero_grad(set_to_none=True)
                with amp.autocast(enabled=args.amp):
                    feats = out["logits"].detach()
                    domain_logits = domain_classifier(feats)
                    domain_loss = F.cross_entropy(domain_logits, batch["center"])
                scaler.scale(domain_loss).backward()
                scaler.step(domain_optimizer)
                scaler.update()
            global_step += 1
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device)
            print(f"Epoch {epoch}: {metrics}")

    save_path = Path(args.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "model.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEG addiction model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-file", type=str, required=True, help="Path to the full dataset manifest (.pt/.pth)")
    parser.add_argument("--split-file", type=str, default="", help="Optional path to dataset split YAML file")
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=(7.0, 1.5, 1.5),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Ratios for train/val/test split when generating a new split file",
    )
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for split generation")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train(cfg, args)


if __name__ == "__main__":
    main()
