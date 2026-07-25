"""Unified SC-MSF training entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch import optim

from runtime import (
    ExperimentLogger,
    REPO_ROOT,
    load_config,
    resolve_repo_path,
    seed_everything,
    validate_device,
)


def build_parser(default_config: str):
    parser = argparse.ArgumentParser(description="Train SC-MSF on PIE or JAAD")
    parser.add_argument(
        "--config-file",
        "--config_file",
        default=default_config,
        help="YACS configuration file",
    )
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--device", default=None, help="Override DEVICE, e.g. cuda or cpu")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--pose-num", default=0, type=int)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--eval-every",
        default=1,
        type=int,
        help="Evaluate on the test split every N epochs; 0 disables periodic testing",
    )
    parser.add_argument(
        "--save-every",
        default=0,
        type=int,
        help="Also retain numbered checkpoints every N epochs; 0 keeps only best/last",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="YACS overrides, e.g. DATASET.ROOT /path/to/data",
    )
    return parser


def build_optimizer(cfg, model):
    return optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)


def build_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.scheduler == "exp":
        return optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=cfg.SOLVER.GAMMA
        )
    if cfg.SOLVER.scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.SOLVER.PLATEAU_FACTOR,
            patience=cfg.SOLVER.PLATEAU_PATIENCE,
            min_lr=cfg.SOLVER.MIN_LR,
        )
    return optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(cfg.SOLVER.MILESTONES), gamma=0.2
    )


def configure_parameter_schedulers(model, device):
    from SC_MSF.utils.scheduler import ParamScheduler, sigmoid_anneal

    model.param_scheduler = ParamScheduler()
    model.param_scheduler.create_new_scheduler(
        name="kld_weight",
        annealer=sigmoid_anneal,
        annealer_kws={
            "device": device,
            "start": 0,
            "finish": 100.0,
            "center_step": 400.0,
            "steps_lo_to_hi": 100.0,
        },
    )
    model.param_scheduler.create_new_scheduler(
        name="z_logit_clip",
        annealer=sigmoid_anneal,
        annealer_kws={
            "device": device,
            "start": 0.05,
            "finish": 5.0,
            "center_step": 300.0,
            "steps_lo_to_hi": 60.0,
        },
    )


def save_checkpoint(path, epoch, model, optimizer, cfg, val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": cfg.dump(),
        },
        path,
    )


def main(default_config="configs/JAAD.yml"):
    args = build_parser(default_config).parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = load_config(
        args.config_file,
        options=args.opts,
        device=args.device,
        pose_num=args.pose_num,
    )
    device = validate_device(cfg.DEVICE)
    seed_everything(args.seed)

    run_name = args.run_name or f"seed_{args.seed}"
    log_file = REPO_ROOT / "logs" / cfg.PROJECT / f"{run_name}.log"
    logger = ExperimentLogger(f"{cfg.PROJECT}_{run_name}", log_file)
    logger.info(f"config={resolve_repo_path(args.config_file)}")
    logger.info(f"device={device}; seed={args.seed}")

    from dataset import make_dataloader
    from SC_MSF.engine import build_engine
    from SC_MSF.modeling import make_model

    model = make_model(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)
    configure_parameter_schedulers(model, device)

    train_dataloader = make_dataloader(cfg, "train", logger=logger)
    val_dataloader = make_dataloader(cfg, "val", logger=logger)
    test_dataloader = (
        make_dataloader(cfg, "test", logger=logger) if args.eval_every else None
    )
    do_train, do_val, inference = build_engine(cfg)

    checkpoint_dir = Path(cfg.CKPT_DIR) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.yml").write_text(cfg.dump(), encoding="utf-8")

    best_val_loss = float("inf")
    for epoch in range(cfg.SOLVER.MAX_EPOCH):
        logger.info(f"epoch={epoch}")
        do_train(
            cfg,
            epoch,
            model,
            optimizer,
            train_dataloader,
            device,
            logger=logger,
            lr_scheduler=lr_scheduler,
        )
        val_loss = do_val(cfg, epoch, model, val_dataloader, device, logger=logger)

        if args.eval_every and (epoch + 1) % args.eval_every == 0:
            inference(
                cfg,
                epoch,
                model,
                test_dataloader,
                device,
                logger=logger,
                eval_kde_nll=False,
            )

        save_checkpoint(
            checkpoint_dir / "last.pth",
            epoch,
            model,
            optimizer,
            cfg,
            val_loss,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                checkpoint_dir / "best.pth",
                epoch,
                model,
                optimizer,
                cfg,
                val_loss,
            )
        if args.save_every and (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pth",
                epoch,
                model,
                optimizer,
                cfg,
                val_loss,
            )

        if cfg.SOLVER.scheduler != "exp":
            lr_scheduler.step(val_loss)


if __name__ == "__main__":
    main()
