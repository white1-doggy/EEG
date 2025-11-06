from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_band_curve(band_scores: Sequence[float], band_labels: Sequence[str], title: str, save_path: Path | None = None) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(band_scores)), band_scores, marker="o")
    plt.xticks(range(len(band_labels)), band_labels, rotation=45)
    plt.ylabel("Macro-F1")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_phase_randomization(delta_scores: Sequence[float], save_path: Path | None = None) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(delta_scores)), delta_scores)
    plt.ylabel("ΔAUROC")
    plt.title("Phase randomization impact")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_cfc_heatmap(cfc_map: np.ndarray, low_bands: Sequence[str], high_bands: Sequence[str], save_path: Path | None = None) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cfc_map, cmap="viridis", aspect="auto")
    plt.colorbar(label="CFC strength")
    plt.xticks(range(len(high_bands)), high_bands)
    plt.yticks(range(len(low_bands)), low_bands)
    plt.title("CFC heatmap")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_microstate_transition(transition: np.ndarray, save_path: Path | None = None) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(transition, cmap="magma", aspect="equal")
    plt.colorbar(label="Transition prob")
    plt.xlabel("To state")
    plt.ylabel("From state")
    plt.title("Microstate transitions")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


__all__ = [
    "plot_band_curve",
    "plot_phase_randomization",
    "plot_cfc_heatmap",
    "plot_microstate_transition",
]
