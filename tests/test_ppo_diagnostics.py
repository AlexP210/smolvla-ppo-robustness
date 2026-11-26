#!/usr/bin/env python3
# Copyright 2026 the course project authors.
# Licensed under the Apache License, Version 2.0.
"""CPU checks for the PPO cycle summaries. They do not run an update."""

import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ppo_diagnostics import (aggregate_optimizer_steps, approximate_kl, distribution,  # noqa: E402
                             explained_variance, reward_report)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def test_advantage_normalization_is_visible() -> None:
    raw = torch.tensor([[0.0, 2.0], [4.0, 6.0]])
    before = distribution(raw)
    centered = (raw - raw.mean()) / raw.std(correction=0)
    after = distribution(centered)
    _check(before["mean"] == 3.0, "the raw advantage mean is wrong")
    _check(after["mean"] == 0.0, "normalization must center the advantages")
    _check(abs(after["std"] - 1.0) < 1e-6, "normalization must use the population standard deviation")


def test_reward_counts_successes() -> None:
    report = reward_report(torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
    _check(report["success_decisions"] == 1, "one positive reward must count as one success decision")
    _check(report["zero_decisions"] == 3, "the other decisions are zeros")
    _check(report["per_env_reward_sum"] == [1.0, 0.0], "per-environment sums must keep which worker succeeded")


def test_explained_variance() -> None:
    target = torch.tensor([1.0, 3.0, 5.0])
    _check(explained_variance(target, target) == 1.0, "a perfect prediction must explain all of the variance")
    _check(explained_variance(torch.zeros(3), torch.ones(3)) is None,
           "a constant target has no variance to explain")
    partial = explained_variance(torch.zeros(3), target)
    _check(partial is not None and partial < 1.0, "a zero prediction must explain less than a perfect one")


def test_kl_and_cycle_aggregate() -> None:
    ones = torch.ones(4)
    _check(approximate_kl(ones, torch.zeros(4)) == 0.0, "a ratio of one has approximate KL zero")
    ratio = torch.tensor([math.e, math.e])
    kl = approximate_kl(ratio, torch.ones(2))
    _check(abs(kl - ((math.e - 1.0) - 1.0)) < 1e-6, "the estimator must be mean((ratio - 1) - log_ratio)")
    steps = [
        {"ratio_mean": 1.0, "ratio_min": 1.0, "ratio_max": 1.0, "ratio_std": 0.0,
         "ratio_median": 1.0, "clip_frac": 0.0, "pg_loss": 0.5, "v_loss": 0.25, "loss": 0.75,
         "approx_kl": 0.0, "grad_norm": 2.0, "grad_norm_applied": 1.0, "value_mean": 0.0,
         "step_skipped": False},
        {"ratio_mean": 3.0, "ratio_min": 0.5, "ratio_max": 4.0, "ratio_std": 1.0,
         "ratio_median": 2.0, "clip_frac": 1.0, "pg_loss": 1.5, "v_loss": 0.75, "loss": 2.25,
         "approx_kl": 0.2, "grad_norm": 8.0, "grad_norm_applied": 1.0, "value_mean": 1.0,
         "step_skipped": False},
    ]
    pooled = aggregate_optimizer_steps(steps)
    _check(pooled["n_steps"] == 2, "both optimizer steps must be counted")
    _check(pooled["ratio_mean"] == 2.0, "the cycle ratio must average every step, not keep the last")
    _check(pooled["ratio_max"] == 4.0 and pooled["ratio_min"] == 0.5, "the extremes must cover every step")
    _check(pooled["pg_loss_mean"] == 1.0 and pooled["v_loss_mean"] == 0.5, "both losses must be averaged")
    _check(pooled["grad_norm_before_clip_max"] == 8.0, "the pre-clip maximum must be kept")
    _check(pooled["grad_norm_after_clip_mean"] == 1.0, "the post-clip mean must be kept")
    _check(pooled["clip_frac_mean"] == 0.5 and pooled["approx_kl_mean"] == 0.1,
           "clip fraction and approximate KL must be averaged")
    json.dumps({"diagnostics": {"optimizer_steps": pooled, "explained_variance": None}})


def main() -> None:
    test_advantage_normalization_is_visible()
    test_reward_counts_successes()
    test_explained_variance()
    test_kl_and_cycle_aggregate()
    print("PPO_DIAGNOSTICS_PASS")


if __name__ == "__main__":
    main()
