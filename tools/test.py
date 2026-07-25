"""Evaluate a trained SC-MSF checkpoint."""

from __future__ import annotations

import argparse
import os
import time

import torch

from runtime import (
    ExperimentLogger,
    REPO_ROOT,
    load_config,
    load_model_weights,
    seed_everything,
    validate_device,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate SC-MSF")
    parser.add_argument(
        "--config-file",
        "--config_file",
        default="configs/JAAD.yml",
        help="YACS configuration file",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--device", default=None, help="Override DEVICE, e.g. cuda or cpu")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--pose-num", default=0, type=int)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Write the potentially large prediction pickle to OUT_DIR",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="YACS overrides, e.g. DATASET.ROOT /path/to/data",
    )
    return parser


def main():
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg = load_config(
        args.config_file,
        options=args.opts,
        device=args.device,
        pose_num=args.pose_num,
    )
    device = validate_device(cfg.DEVICE)
    seed_everything(args.seed)

    logger = ExperimentLogger(
        f"{cfg.PROJECT}_test",
        REPO_ROOT / "logs" / cfg.PROJECT / "test.log",
    )

    from dataset import make_dataloader
    from SC_MSF.engine import build_engine
    from SC_MSF.modeling import make_model

    model = make_model(cfg).to(device)
    checkpoint_path = load_model_weights(model, args.checkpoint, device)
    logger.info(f"loaded_checkpoint={checkpoint_path}")

    test_dataloader = make_dataloader(cfg, "test", logger=logger)
    _, _, inference = build_engine(cfg)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    inference(
        cfg,
        0,
        model,
        test_dataloader,
        device,
        logger=logger,
        eval_kde_nll=False,
        test_mode=args.save_predictions,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    sample_count = len(test_dataloader.dataset)
    logger.info(
        f"inference_seconds={elapsed:.3f}; samples={sample_count}; "
        f"milliseconds_per_sample={elapsed / max(sample_count, 1) * 1000:.3f}"
    )


if __name__ == "__main__":
    main()
