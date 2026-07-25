"""Dataset and dataloader factory functions."""

import collections.abc


def _data_layers():
    from .JAAD import JAADDataset
    from .PIE import PIEDataset

    return {
        "JAAD_JAAD": JAADDataset,
        "PIE_PIE": PIEDataset,
    }


def make_dataset(cfg, split):
    key = f"{cfg.DATASET.NAME}_{cfg.METHOD}"
    try:
        data_layer = _data_layers()[key]
    except KeyError as exc:
        raise NameError(
            f"Unknown method and dataset combination: {cfg.METHOD} + "
            f"{cfg.DATASET.NAME}"
        ) from exc
    return data_layer(cfg, split)


def make_dataloader(cfg, split="train", logger=None):
    from torch.utils.data import DataLoader

    batch_size = cfg.TEST.BATCH_SIZE if split == "test" else cfg.SOLVER.BATCH_SIZE
    dataloader = DataLoader(
        make_dataset(cfg, split),
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=collate_dict,
    )
    message = f"{split} dataloader: {len(dataloader)}"
    logger.info(message) if hasattr(logger, "info") else print(message)
    return dataloader


def collate_dict(batch):
    import dill
    import torch
    from torch.utils.data._utils.collate import default_collate

    if not batch:
        return batch
    collated = {}
    for key in batch[0]:
        value = batch[0][key]
        if value is None:
            collated[key] = None
        elif isinstance(value, collections.abc.Mapping):
            neighbors = {
                sub_key: [item[key][sub_key] for item in batch]
                for sub_key in value
            }
            collated[key] = (
                dill.dumps(neighbors)
                if torch.utils.data.get_worker_info()
                else neighbors
            )
        else:
            collated[key] = default_collate([item[key] for item in batch])
    return collated
