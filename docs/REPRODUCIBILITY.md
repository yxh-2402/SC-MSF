# Reproducibility record

Complete this record from the corrected public-release code before creating a
version tag or citing the repository.

## Software

- Repository commit:
- Release tag:
- Python version:
- PyTorch version:
- CUDA toolkit and driver:
- Dependency lock file or `pip freeze` artifact:

## Hardware

- Operating system:
- GPU model and count:
- CPU:
- RAM:
- Training time for PIE:
- Training time for JAAD:

## Data and features

- PIE release/source:
- JAAD branch/release:
- Train/validation/test split identifiers:
- Pose extractor and checkpoint:
- Optical-flow extractor and checkpoint:
- Image feature extractor and checkpoint:
- Feature archive DOI/URL:
- SHA-256 checksums:

## Training protocol

- Random seeds:
- Number of independent runs:
- Model-selection criterion:
- Selected epoch for PIE:
- Selected epoch for JAAD:
- Configuration differences from `configs/*.yml`:

## Corrected-release results

Do not copy metrics from the legacy experiment logs. Populate this table only
after rerunning the corrected data loader.

| Dataset | Seed(s) | ADE 0.5 s | ADE 1.0 s | ADE 1.5 s | FDE | C-ADE 1.5 s | C-FDE |
|---|---:|---:|---:|---:|---:|---:|---:|
| PIE | pending | pending | pending | pending | pending | pending | pending |
| JAAD | pending | pending | pending | pending | pending | pending | pending |

Report mean and standard deviation when the manuscript claims multi-seed
results. Record the exact command used for every final table.

## Artifact publication

- Final weights DOI/URL:
- Prediction outputs DOI/URL, if shared:
- Code archive DOI:
- Data Availability Statement updated:
- Code Availability Statement updated:
