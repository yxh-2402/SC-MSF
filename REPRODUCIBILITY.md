# Reproducibility record

This record identifies the versioned artifacts supplied for JAAD evaluation.
Numerical results are reported in the accompanying manuscript.

## JAAD release artifacts

| Artifact | Version |
| --- | --- |
| Source snapshot | `bc33e098d0e04af9f0676474842ca14075c6106c` |
| Configuration | [`../configs/JAAD.yml`](../configs/JAAD.yml) |
| Checkpoint | [`../checkpoints/JAAD/TFED_Epoch_0_088.pth`](../checkpoints/JAAD/TFED_Epoch_0_088.pth) |
| Checkpoint SHA-256 | `3C79AC48E37D7F95CF49EDCDC754802CB774A0C9469E3F45922204CF93ADD776` |

The checkpoint was strictly loaded against the released JAAD model with no
missing or unexpected parameter keys. The project loader maps the `flow_wfe`
checkpoint prefix to the released `flow_dtde` module name.

## Evaluation

```bash
python tools/test.py \
  --config-file configs/JAAD.yml \
  --checkpoint checkpoints/JAAD/TFED_Epoch_0_088.pth \
  DATASET.ROOT /absolute/path/to/JAAD \
  DATASET.TRAJECTORY_PATH /absolute/path/to/JAAD/trajectories
```

Official datasets and derived feature inputs are not redistributed with this
repository. Their expected layout is documented in
[`DATASET.md`](DATASET.md).
