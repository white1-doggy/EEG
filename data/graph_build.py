from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


def pairwise_distance(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def build_graph(
    channel_coords: Dict[str, Tuple[float, float, float]],
    k: int = 4,
    radius: float | None = None,
    normalize: bool = True,
) -> dict:
    """Construct adjacency and Laplacian matrices from channel coordinates."""
    names = list(channel_coords.keys())
    coords = np.stack([channel_coords[n] for n in names], axis=0)
    dist = pairwise_distance(coords)
    np.fill_diagonal(dist, np.inf)
    n = len(names)
    adjacency = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if radius is not None:
            neighbors = np.where(dist[i] <= radius)[0]
        else:
            neighbors = np.argsort(dist[i])[:k]
        adjacency[i, neighbors] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    degree = adjacency.sum(axis=-1, keepdims=True)
    degree[degree == 0] = 1.0
    if normalize:
        d_inv_sqrt = 1.0 / np.sqrt(degree)
        norm_adj = adjacency * d_inv_sqrt * d_inv_sqrt.transpose()
    else:
        norm_adj = adjacency
    laplacian = np.eye(n, dtype=np.float32) - norm_adj
    return {
        "A": torch.from_numpy(adjacency),
        "A_norm": torch.from_numpy(norm_adj),
        "L": torch.from_numpy(laplacian),
        "names": names,
    }


def save_graph(graph: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, path)


def build_default_graph(path: Path, k: int = 4, radius: float | None = None) -> dict:
    """Build the default 62-channel montage graph and persist it."""
    # Simplified 10-20 montage coordinates (x, y, z in arbitrary units)
    default_coords = {
        "Fp1": (-0.5, 1.0, 0.5),
        "Fp2": (0.5, 1.0, 0.5),
        "F7": (-1.0, 0.7, 0.2),
        "F3": (-0.5, 0.6, 0.8),
        "Fz": (0.0, 0.6, 1.0),
        "F4": (0.5, 0.6, 0.8),
        "F8": (1.0, 0.7, 0.2),
        "FC5": (-0.9, 0.3, 0.6),
        "FC1": (-0.3, 0.3, 0.9),
        "FC2": (0.3, 0.3, 0.9),
        "FC6": (0.9, 0.3, 0.6),
        "T7": (-1.1, 0.0, 0.2),
        "C3": (-0.6, 0.0, 0.9),
        "Cz": (0.0, 0.0, 1.0),
        "C4": (0.6, 0.0, 0.9),
        "T8": (1.1, 0.0, 0.2),
        "TP9": (-1.1, -0.3, 0.1),
        "CP5": (-0.9, -0.3, 0.6),
        "CP1": (-0.3, -0.3, 0.9),
        "CP2": (0.3, -0.3, 0.9),
        "CP6": (0.9, -0.3, 0.6),
        "TP10": (1.1, -0.3, 0.1),
        "P7": (-1.0, -0.6, 0.2),
        "P3": (-0.5, -0.6, 0.8),
        "Pz": (0.0, -0.6, 1.0),
        "P4": (0.5, -0.6, 0.8),
        "P8": (1.0, -0.6, 0.2),
        "POz": (0.0, -0.9, 0.9),
        "O1": (-0.5, -1.0, 0.5),
        "Oz": (0.0, -1.0, 0.6),
        "O2": (0.5, -1.0, 0.5),
    }
    graph = build_graph(default_coords, k=k, radius=radius)
    save_graph(graph, path)
    return graph


__all__ = ["build_graph", "save_graph", "build_default_graph"]
