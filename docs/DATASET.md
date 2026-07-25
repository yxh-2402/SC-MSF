# Dataset preparation

## Official sources

- PIE: <https://github.com/aras62/PIE>
- JAAD 2.0: <https://github.com/ykotseruba/JAAD>

Download each dataset from its official source and comply with its terms.
Do not commit raw frames, annotations, cached databases, or derived features
to this repository.

## Expected roots

The default configurations use ignored local directories:

```text
data/
├── PIE/
│   ├── annotations/
│   ├── images/
│   └── data_cache/pie_database.pkl
└── JAAD/
    ├── annotations/
    ├── annotations_appearance/
    ├── annotations_attributes/
    ├── annotations_traffic/
    ├── annotations_vehicle/
    ├── images/
    ├── split_ids/
    └── data_cache/jaad_database.pkl
```

The official APIs can generate the base cache, but the model requires three
additional fields under every pedestrian annotation:

| Field | Meaning | Model expectation |
|---|---|---|
| `pose_features` | per-frame pedestrian pose | at least 15 frames, 18 joints, at least x/y coordinates |
| `optical_flow` | per-frame two-channel flow maps | model code expects 15 × 2 × 24 × 8 per sample window |
| `image_features` | per-frame image embeddings | current model projection expects 1000 values per frame |

## Feature provenance requirement

The original experiment folder did not contain the scripts that generated
these three augmented fields. Before public release, provide one of:

1. deterministic extraction scripts with model names, versions, preprocessing,
   and pretrained-weight checksums; or
2. a legally redistributable archive of the augmented caches/features with a
   DOI, license, version, and SHA-256 checksums.

Without one of these items, independent users cannot fully reproduce training.
Do not describe the repository as fully reproducible until this gap is closed.

## Validation

Validate an augmented cache before training:

```bash
python tools/check_data.py --config-file configs/PIE.yml \
  DATASET.ROOT /absolute/path/to/PIE \
  DATASET.TRAJECTORY_PATH /absolute/path/to/PIE
```

Add `--check-dataloader` to construct and inspect one batch after the cache
check succeeds.
