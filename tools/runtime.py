"""Shared runtime helpers for training and evaluation entry points."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ExperimentLogger:
    """Small logger adapter expected by the legacy training engine."""

    def __init__(self, name: str, log_file: Path):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = name
        self._logger = logging.getLogger(f"sc_msf.{name}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)
        self._logger.addHandler(file_handler)

    def info(self, message):
        self._logger.info(message)

    def log_values(self, values, step=None):
        serializable = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            elif isinstance(value, np.generic):
                value = value.item()
            serializable[key] = value
        if step is not None:
            serializable["step"] = step
        self.info("metrics=" + json.dumps(serializable, sort_keys=True))


def resolve_repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(config_file, options=None, device=None, pose_num=None):
    from configs import cfg as default_cfg

    cfg = default_cfg.clone()
    cfg.merge_from_file(str(resolve_repo_path(config_file)))
    if options:
        cfg.merge_from_list(options)
    if device is not None:
        cfg.DEVICE = device
    if pose_num is not None:
        cfg.MODEL.POSENUM = pose_num

    cfg.CKPT_DIR = str(resolve_repo_path(cfg.CKPT_DIR))
    cfg.OUT_DIR = str(resolve_repo_path(cfg.OUT_DIR))
    cfg.DATASET.ROOT = str(resolve_repo_path(cfg.DATASET.ROOT))
    cfg.DATASET.TRAJECTORY_PATH = str(
        resolve_repo_path(cfg.DATASET.TRAJECTORY_PATH)
    )
    cfg.freeze()
    return cfg


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def validate_device(device_name: str):
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Install a CUDA-enabled PyTorch "
            "build or pass --device cpu."
        )
    return device


def load_model_weights(model, checkpoint_path, device):
    checkpoint_path = resolve_repo_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload

    state_dict = {
        key.removeprefix("module.").replace("flow_wfe", "flow_dtde"): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    return checkpoint_path
