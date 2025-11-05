# EEG Addiction Benchmark

This repository implements a full training and evaluation pipeline for multimodal EEG-based addiction classification. The project contains:

- **Data processing** (`data/`): windowing, STFT feature extraction, node feature construction, and graph generation utilities.
- **Feature utilities** (`features/`): pre-computation of Welch/FOOOF teachers and on-the-fly data augmentation.
- **Models** (`models/`): spectral CNN, graph neural network backbone, temporal transformer, and interpretable heads (band gates, aperiodic matching, cross-frequency coupling, microstates).
- **Losses** (`losses/`): multi-task objective with optional teacher supervision and regularisation.
- **Training/Evaluation scripts** (`train.py`, `evaluate.py`, `run.sh`).
- **Configuration** (`configs/default.yaml`).

## Getting started

1. Install dependencies (PyTorch, SciPy, scikit-learn, PyYAML, matplotlib).
2. Prepare your dataset as a `torch.save` file containing a list of samples with the following keys:
   - `x_raw` – `FloatTensor [C, T]`
   - `center_id` – station/domain label
   - `subject_id` – subject identifier (string)
   - `y` – class label `0..3`
   - Optional teacher annotations: `welch_rel`, `fooof_slope`, `fooof_offset`, `pac_ref`
3. Optionally build a channel graph via `data/graph_build.py` (see `build_default_graph`).

## Training

```bash
./run.sh train --config configs/default.yaml --data-file path/to/dataset.pt --output-dir checkpoints --amp
```

When provided with a single dataset file, `train.py` automatically looks for a YAML split file (defaults to `dataset.splits.yaml`).
If the file does not exist, it is generated with a 7 : 1.5 : 1.5 train/validation/test ratio (customisable via `--split-ratios`).
Re-run training with `--split-file path/to/splits.yaml` to reuse or share the same split definition.

The script uses AdamW with cosine warmup, mixed precision (optional), and supports domain adversarial training (enable via the configuration `train.use_dann`).

## Evaluation

```bash
# Leave-one-site-out metrics
./run.sh evaluate --config configs/default.yaml --data path/to/data.pt --checkpoint checkpoints/model.pt --task loso

# Band deletion/insertion fidelity curves
./run.sh evaluate --config configs/default.yaml --data path/to/data.pt --checkpoint checkpoints/model.pt --task fidelity

# Phase randomisation robustness
./run.sh evaluate --config configs/default.yaml --data path/to/data.pt --checkpoint checkpoints/model.pt --task phase
```

## Configuration

Key hyperparameters are stored in `configs/default.yaml`, including STFT settings, feature dimensions, and loss weights. Adjust the values or create new configuration files to suit your dataset.

## License

This project is provided as-is without warranty. Adapt and extend to your research needs.
