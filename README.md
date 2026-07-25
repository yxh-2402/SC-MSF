# SC-MSF

Research code for multimodal pedestrian bounding-box trajectory prediction on
the PIE and JAAD benchmarks. This repository is a cleaned public-release copy
of the experiment code used during manuscript development.

## Important release note

The public-release cleanup corrected two indexing defects in the legacy data
loader:

1. a flow-frame index overwrote the dataset sample index and could mix fields
   from different samples;
2. the key-pose index list was reset repeatedly, which duplicated selected
   pose frames.

The corrected logic is isolated in `dataset/selection.py` and covered by unit
tests. Results produced with the legacy loader must be rerun and must not be
presented as results from this corrected release. See
[`CHANGELOG.md`](CHANGELOG.md).

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install a PyTorch build appropriate for the
machine from [pytorch.org](https://pytorch.org/get-started/locally/). Install
the remaining dependencies with:

```bash
python -m pip install -r requirements.txt
```



## Data

The official PIE and JAAD datasets are not included. In addition to their
standard annotations, SC-MSF expects augmented pedestrian records containing
pose, optical-flow, and image features. Follow
[`docs/DATASET.md`](docs/DATASET.md), then validate the cache:

```bash
python tools/check_data.py --config-file configs/JAAD.yml \
  DATASET.ROOT /absolute/path/to/JAAD \
  DATASET.TRAJECTORY_PATH /absolute/path/to/JAAD/trajectories
```

## Training

JAAD:

```bash
python tools/train_jaad.py --seed 0 --pose-num 0 \
  DATASET.ROOT /absolute/path/to/JAAD \
  DATASET.TRAJECTORY_PATH /absolute/path/to/JAAD/trajectories
```

PIE:

```bash
python tools/train_pie.py --seed 0 --pose-num 0 \
  DATASET.ROOT /absolute/path/to/PIE \
  DATASET.TRAJECTORY_PATH /absolute/path/to/PIE
```

Relative paths in configuration files are resolved from the repository root.
Training keeps only `best.pth` and `last.pth` by default. Use
`--save-every N` only when numbered checkpoints are needed.

## Evaluation

```bash
python tools/test.py \
  --config-file configs/JAAD.yml \
  --checkpoint checkpoints/JAAD/seed_0/best.pth \
  DATASET.ROOT /absolute/path/to/JAAD \
  DATASET.TRAJECTORY_PATH /absolute/path/to/JAAD/trajectories
```

Evaluation does not write the potentially large prediction pickle by default.
Pass `--save-predictions` when the full output is required.

## Reproducibility

The exact paper results, hardware details, final checkpoint links, feature
extractor versions, and artifact checksums must be added after rerunning the
corrected release. The required record is listed in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

Model weights and generated predictions are intentionally ignored by Git.
Publish final weights through a versioned GitHub Release or an archival
repository such as Zenodo, and record a persistent identifier in the
manuscript.

## Repository layout

```text
configs/       PIE and JAAD experiment configurations
dataset/       dataset interfaces, sampling, and feature validation
SC_MSF/      SC-MSF model, training engine, losses, and utilities
tools/         training, evaluation, and data-check entry points
tests/         lightweight regression tests
docs/          dataset and reproducibility documentation
```

## License and attribution

New SC-MSF code is released under the MIT License. Adapted components retain
their upstream notices; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
Confirm the author name in `LICENSE` and complete the publication checklist
before making the repository public.
