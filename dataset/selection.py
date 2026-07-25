"""Deterministic selection utilities shared by the PIE and JAAD datasets."""

from __future__ import annotations

import numpy as np


def select_key_poses(obs_pose, num_key_poses: int = 8):
    """Return 2-D poses and the strongest candidate frames in time order.

    The legacy implementation scored each candidate frame by summing its
    second coordinate. This function preserves that scoring rule while fixing
    the repeated-index bug in the original code.
    """
    poses = np.asarray(obs_pose, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[-1] < 2:
        raise ValueError(
            "obs_pose must have shape (time, joints, coordinates) with at least "
            "two coordinates per joint"
        )

    poses_2d = poses[..., :2].copy()
    candidates = poses_2d[1:14]
    if candidates.shape[0] < num_key_poses:
        raise ValueError(
            f"Need at least {num_key_poses} candidate pose frames, "
            f"but received {candidates.shape[0]}"
        )

    scores = candidates[..., 1].sum(axis=1)
    strongest = np.argsort(-scores, kind="stable")[:num_key_poses]
    chronological = np.sort(strongest)
    return poses_2d, candidates[chronological].copy()


def select_key_flow_sequence(obs_flow, remove_count: int = 2):
    """Select the legacy 14-frame flow window and remove temporal outliers.

    Outlier scores are computed on frames 1..13 of the selected window, as in
    the original implementation. The returned removal indices are shifted by
    one so they refer to the same frames in the full window.
    """
    flows = np.asarray(obs_flow)
    key_flow_sequence = flows[1:15].copy()
    if key_flow_sequence.shape[0] < 2:
        raise ValueError("obs_flow must contain enough frames for the [1:15] window")

    candidates = key_flow_sequence[1:15, :, 1, :, :]
    if not 0 <= remove_count < candidates.shape[0]:
        raise ValueError(
            f"remove_count must be between 0 and {candidates.shape[0] - 1}"
        )
    if remove_count == 0:
        return key_flow_sequence

    overall_mean = np.mean(candidates)
    reduction_axes = tuple(range(1, candidates.ndim))
    frame_means = np.mean(candidates, axis=reduction_axes)
    errors = np.abs(frame_means - overall_mean)

    relative_indices = np.argsort(errors, kind="stable")[-remove_count:]
    window_indices = np.sort(relative_indices + 1)
    return np.delete(key_flow_sequence, window_indices, axis=0)
