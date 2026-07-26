# SC-MSF

Research code for multimodal pedestrian bounding-box trajectory prediction on
the PIE and JAAD benchmarks. This repository is a cleaned public-release copy
of the experiment code used during manuscript development.

## JAAD reproducibility snapshot

The released JAAD checkpoint corresponds to the following versioned artifacts:

| Artifact | Version |
| --- | --- |
| Source snapshot | `bc33e098d0e04af9f0676474842ca14075c6106c` |
| Configuration | [`configs/JAAD.yml`](configs/JAAD.yml) |
| Checkpoint | [`checkpoints/JAAD/TFED_Epoch_0_088.pth`](checkpoints/JAAD/TFED_Epoch_0_088.pth) |
| Checkpoint SHA-256 | `3C79AC48E37D7F95CF49EDCDC754802CB774A0C9469E3F45922204CF93ADD776` |

Use these three artifacts together when evaluating the released JAAD model.

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
  --checkpoint checkpoints/JAAD/TFED_Epoch_0_088.pth \
  DATASET.ROOT /absolute/path/to/JAAD \
  DATASET.TRAJECTORY_PATH /absolute/path/to/JAAD/trajectories
```

Evaluation does not write the potentially large prediction pickle by default.
Pass `--save-predictions` when the full output is required.

## Reproducibility

The JAAD source snapshot, evaluation configuration, checkpoint, and checksum
are listed above. The checkpoint is included in this repository and is loaded
by `tools/test.py`; the loader also maps its `flow_wfe` checkpoint prefix
to the released `flow_dtde` module name. Dataset preparation and feature
requirements are documented in [`docs/DATASET.md`](docs/DATASET.md).

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
