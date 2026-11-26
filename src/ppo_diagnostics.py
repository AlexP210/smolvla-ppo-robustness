# Copyright 2026 the course project authors.
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Read-only summaries of one PPO cycle.

Nothing here is added to the loss, and nothing here draws a random number. The tensors it reads are the ones
the update already computed. The cycle record keeps the last optimizer step at the top level, which is what
older logs contain, and adds these summaries beside it so a last step is not mistaken for the cycle.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch


def _finite(value: float) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def distribution(values: torch.Tensor) -> dict:
    """Population mean and spread, plus the extremes. An empty tensor has no statistics."""
    flat = values.detach().float().reshape(-1)
    count = int(flat.numel())
    if count == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": count,
        "mean": float(flat.mean().item()),
        "std": float(flat.std(correction=0).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def reward_report(rewards: torch.Tensor) -> dict:
    """Sparse-reward counts. A positive entry is one successful decision, not a shaped return."""
    flat = rewards.detach().float().reshape(-1)
    positive = flat > 0
    per_env = rewards.detach().float().sum(dim=-1).reshape(-1) if rewards.ndim >= 1 else flat
    return {
        "decisions": int(flat.numel()),
        "success_decisions": int(positive.sum().item()),
        "zero_decisions": int((flat == 0).sum().item()),
        "other_decisions": int(((flat != 0) & ~positive).sum().item()),
        "reward_sum": float(flat.sum().item()) if flat.numel() else 0.0,
        "per_env_reward_sum": [float(value) for value in per_env.tolist()],
        **{key: distribution(flat)[key] for key in ("mean", "std", "min", "max")},
    }


def explained_variance(predicted: torch.Tensor, target: torch.Tensor) -> Optional[float]:
    """1 - Var(target - predicted) / Var(target), population variances. None when the target does not vary."""
    # Collection stores values on the host. GAE returns the targets on the training device.
    # The subtraction is a log, so both copies move to the host before it. Neither tensor is written back.
    y = target.detach().float().reshape(-1).cpu()
    y_hat = predicted.detach().float().reshape(-1).cpu()
    if y.numel() == 0 or y.numel() != y_hat.numel():
        return None
    var_y = float(y.var(correction=0).item())
    if var_y == 0.0:
        return None
    residual = y - y_hat
    return float(1.0 - float(residual.var(correction=0).item()) / var_y)


def approximate_kl(ratio: torch.Tensor, log_ratio: torch.Tensor) -> float:
    """Schulman's k3 estimator, mean((ratio - 1) - log_ratio), on detached values.

    `ratio` is pi_new / pi_old and `log_ratio` is its log. The estimator is not added to the surrogate.
    """
    return float(((ratio.detach() - 1.0) - log_ratio.detach()).mean().item())


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    usable = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not usable:
        return None
    return float(sum(usable) / len(usable))


def _extreme(values: Sequence[Optional[float]], larger: bool) -> Optional[float]:
    usable = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not usable:
        return None
    return max(usable) if larger else min(usable)


def aggregate_optimizer_steps(steps: Sequence[dict]) -> dict:
    """Means and extremes over every optimizer step in the cycle, not the last step alone."""
    def column(key: str) -> list:
        return [_finite(step.get(key)) for step in steps]

    return {
        "n_steps": len(steps),
        "n_skipped": int(sum(1 for step in steps if step.get("step_skipped"))),
        "approx_kl_mean": _mean(column("approx_kl")),
        "approx_kl_max": _extreme(column("approx_kl"), True),
        "ratio_mean": _mean(column("ratio_mean")),
        "ratio_std_mean": _mean(column("ratio_std")),
        "ratio_median_mean": _mean(column("ratio_median")),
        "ratio_min": _extreme(column("ratio_min"), False),
        "ratio_max": _extreme(column("ratio_max"), True),
        "clip_frac_mean": _mean(column("clip_frac")),
        "pg_loss_mean": _mean(column("pg_loss")),
        "v_loss_mean": _mean(column("v_loss")),
        "loss_mean": _mean(column("loss")),
        "grad_norm_before_clip_mean": _mean(column("grad_norm")),
        "grad_norm_before_clip_max": _extreme(column("grad_norm"), True),
        "grad_norm_after_clip_mean": _mean(column("grad_norm_applied")),
        "grad_norm_after_clip_max": _extreme(column("grad_norm_applied"), True),
        "value_mean": _mean(column("value_mean")),
    }
