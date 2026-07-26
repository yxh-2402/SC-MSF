# Changelog

## 0.1.0 - public-release cleanup

- Reduced the experiment directory to the active PIE/JAAD training and
  evaluation chain.
- Removed generated weights, outputs, logs, caches, IDE state, and unrelated
  experiments.
- Replaced private absolute paths and hard-coded checkpoints with
  configuration and command-line arguments.
- Unified training behavior and retained only best/last checkpoints by
  default.
- Removed the duplicate JAAD model file and the stale model alias.
- Made tensor allocation follow the input/model device instead of forcing
  `cuda:0`.
- Fixed sample-index overwriting in PIE and JAAD data loaders.
- Fixed repeated key-pose selection and flow-window index alignment.
- Added regression tests, dependency declarations, dataset validation, and
  release documentation.
