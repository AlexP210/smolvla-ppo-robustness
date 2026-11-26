#!/usr/bin/env python3
# Copyright 2026 the course project authors.
# Licensed under the Apache License, Version 2.0.
"""CPU checks for evaluation seeds and initial flow noise.

These do not start the simulator. They check the draws the evaluator would pass in, and that the
unchanged flow integrator turns identical noise into identical actions when its velocity is row-wise.
"""

import ast
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from eval_noise import env_reset_seed, initial_flow_noise  # noqa: E402
from flow_sde import sde_sample  # noqa: E402


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def test_env_seed_ignores_company() -> None:
    alone = env_reset_seed(11, episode_id=3, batch=1)
    _check(alone == env_reset_seed(11, 3, 1), "the same episode must reset the same way twice")
    _check(alone != env_reset_seed(12, 3, 1), "the environment seed must change the reset")
    _check(alone != env_reset_seed(11, 4, 1), "the episode id must change the reset")
    _check(alone != env_reset_seed(11, 3, 2), "the batch ordinal must change the reset")
    _check(0 <= alone < 2 ** 31 - 1, "the reset seed must fit the simulator's non-negative range")


def test_noise_is_common_across_batch_layouts() -> None:
    solo = initial_flow_noise(5, [3], decision=4, chunk=6, action_dim=7, batch=1)
    mixed = initial_flow_noise(5, [9, 3, 1, 3], decision=4, chunk=6, action_dim=7, batch=1)
    again = initial_flow_noise(5, [3], decision=4, chunk=6, action_dim=7, batch=1)
    _check(torch.equal(solo, again), "repeating the draw must reproduce it")
    _check(torch.equal(solo[0], mixed[1]), "episode 3 must match in a different batch position")
    _check(torch.equal(mixed[1], mixed[3]), "the same episode id twice in one batch must match")
    _check(not torch.equal(solo, initial_flow_noise(5, [3], decision=5, chunk=6, action_dim=7, batch=1)),
           "a different decision must draw different noise")
    _check(not torch.equal(solo, initial_flow_noise(6, [3], decision=4, chunk=6, action_dim=7, batch=1)),
           "a different policy seed must draw different noise")
    _check(solo.dtype == torch.float32, "the draw must stay float32, the dtype of the policy's own sampler")
    _check(tuple(solo.shape) == (1, 6, 7), "the shape must be one row of chunk by action width")


def test_distribution_is_standard_normal() -> None:
    sample = initial_flow_noise(0, list(range(256)), decision=0, chunk=16, action_dim=8)
    flat = sample.reshape(-1)
    _check(abs(float(flat.mean())) < 0.05, "the sample mean left the standard normal")
    _check(abs(float(flat.std(unbiased=False)) - 1.0) < 0.05, "the sample std left the standard normal")


def test_integrator_agrees_when_the_noise_agrees() -> None:
    """A row-wise velocity and flow_ode must not invent a batch-size difference once the noise matches."""
    def velocity(x, _t):
        return 0.1 * x, None

    solo = initial_flow_noise(2, [7], decision=1, chunk=4, action_dim=5)
    batch = initial_flow_noise(2, [0, 7, 3], decision=1, chunk=4, action_dim=5)
    torch.manual_seed(0)
    one = sde_sample(velocity, solo, 4, "flow_ode", 0.0, 4, 5, mode="eval")
    torch.manual_seed(0)
    many = sde_sample(velocity, batch, 4, "flow_ode", 0.0, 4, 5, mode="eval")
    _check(torch.equal(one["actions"][0], many["actions"][1]),
           "the same noise and the same Euler step produced different actions")


def test_noise_leaves_the_global_generator_alone() -> None:
    torch.manual_seed(123)
    before = torch.randn(4)
    torch.manual_seed(123)
    initial_flow_noise(1, [0, 4, 2], decision=3, chunk=8, action_dim=4, batch=0)
    after = torch.randn(4)
    _check(torch.equal(before, after), "initial flow noise drew from the global generator")


def test_evaluate_keeps_the_historical_defaults() -> None:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "evaluate.py")
    text = open(path).read()
    tree = ast.parse(text)
    found = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
            found = node
    _check(found is not None, "evaluate is missing")
    defaults = {arg.arg: default for arg, default in zip(found.args.kwonlyargs, found.args.kw_defaults)}
    for name in ("env_seed", "policy_seed", "episode_ids"):
        default = defaults[name]
        _check(isinstance(default, ast.Constant) and default.value is None,
               "%s must default to None so existing callers stay on the historical path" % name)
    action_default = defaults["n_action_steps"]
    _check(isinstance(action_default, ast.Constant) and action_default.value == 1,
           "the default action execution horizon must stay 1")
    _check("eval_seed_base + batch * n_envs + i" in text,
           "the historical reset formula must remain for callers that do not pass env_seed")
    _check('model.config.noise_method = "flow_ode"' in text,
           "evaluation must still integrate with flow_ode")


def main() -> None:
    test_env_seed_ignores_company()
    test_noise_is_common_across_batch_layouts()
    test_distribution_is_standard_normal()
    test_integrator_agrees_when_the_noise_agrees()
    test_noise_leaves_the_global_generator_alone()
    test_evaluate_keeps_the_historical_defaults()
    print("EVAL_NOISE_PASS")


if __name__ == "__main__":
    main()
