"""Validate the augmented PIE/JAAD cache before training."""

from __future__ import annotations

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from runtime import load_config


REQUIRED_FEATURES = ("pose_features", "optical_flow", "image_features")


def load_pickle(path: Path):
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        import dill

        with path.open("rb") as handle:
            try:
                return dill.load(handle)
            except Exception:
                handle.seek(0)
                return dill.load(handle, encoding="latin1")


def shape_of(value):
    try:
        return tuple(np.asarray(value).shape)
    except Exception:
        return None


def check_database(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Augmented dataset cache not found: {path}")
    database = load_pickle(path)
    if not isinstance(database, dict):
        raise TypeError(f"Expected a dictionary in {path}, got {type(database).__name__}")

    missing = Counter()
    shapes = defaultdict(Counter)
    pedestrian_count = 0
    for video in database.values():
        for annotation in video.get("ped_annotations", {}).values():
            pedestrian_count += 1
            for feature in REQUIRED_FEATURES:
                value = annotation.get(feature)
                if value is None:
                    missing[feature] += 1
                else:
                    shapes[feature][shape_of(value)] += 1

    print(f"cache={path}")
    print(f"videos={len(database)}; pedestrians={pedestrian_count}")
    for feature in REQUIRED_FEATURES:
        print(
            f"{feature}: missing={missing[feature]}; "
            f"common_shapes={shapes[feature].most_common(5)}"
        )

    if pedestrian_count == 0 or any(missing.values()):
        raise RuntimeError(
            "The cache is incomplete. See docs/DATASET.md for the required augmented "
            "feature fields."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-file",
        default="configs/JAAD.yml",
        help="PIE or JAAD configuration file",
    )
    parser.add_argument("--cache", default=None, help="Override the cache pickle path")
    parser.add_argument(
        "--check-dataloader",
        action="store_true",
        help="Build one batch after validating the cache",
    )
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = load_config(args.config_file, options=args.opts)
    dataset_name = cfg.DATASET.NAME.lower()
    cache_path = (
        Path(args.cache)
        if args.cache
        else Path(cfg.DATASET.ROOT)
        / "data_cache"
        / f"{dataset_name}_database.pkl"
    )
    check_database(cache_path)

    if args.check_dataloader:
        from dataset import make_dataloader

        batch = next(iter(make_dataloader(cfg, args.split)))
        print("batch_keys=" + ",".join(sorted(batch)))
        for key, value in batch.items():
            if hasattr(value, "shape"):
                print(f"{key}: shape={tuple(value.shape)}")


if __name__ == "__main__":
    main()
