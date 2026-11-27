#!/usr/bin/env python3
# Copyright 2026 the course project authors.
# Licensed under the Apache License, Version 2.0.
"""CPU checks for ppo_run_v1. They do not start training."""

import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from run_config import argv_from_ppo_run, load_ppo_run, ppo_run_block, ppo_run_from_args  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK0 = os.path.join(ROOT, "configs", "nominal_ppo_libero10_task0.json")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def test_same_file_writes_the_same_block() -> None:
    first = ppo_run_block(load_ppo_run(TASK0))
    second = ppo_run_block(load_ppo_run(TASK0))
    _check(first == second, "two loads of one config must match")
    _check(first["action_steps"] == 1 and first["kl_coef"] == 0.0, "the reproduction keeps the upstream objective")
    _check(first["actor_lr"] == 1e-5 and first["critic_lr"] == 1e-3, "learning rates stay the contact example's")
    _check(first["rollout_steps"] == 32 and first["noise_level"] == 0.1, "rollout and exploration stay the original")
    _check(ppo_run_from_args(SimpleNamespace(**first)) == first,
           "reading the block back from an argument object must reproduce it")
    _check(ppo_run_block(load_ppo_run(TASK0)) == first, "a second launch of the same file must match")


def test_action_steps_other_than_one_are_rejected() -> None:
    document = json.loads(open(TASK0, encoding="utf-8").read())
    document["action_steps"] = 10
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(document, handle)
    handle.close()
    try:
        try:
            load_ppo_run(handle.name)
        except ValueError:
            return
        raise SystemExit("a config with action_steps 10 must be rejected")
    finally:
        os.remove(handle.name)


def test_expanded_command_keeps_the_original_update() -> None:
    flags = argv_from_ppo_run(load_ppo_run(TASK0))
    _check(flags[flags.index("--action_steps") + 1] == "1", "the expanded command keeps one executed action")
    _check(flags[flags.index("--epochs") + 1] == "1", "the reproduction uses one pass over the rollout")
    _check(flags[flags.index("--kl_coef") + 1] == "0.0", "the reference anchor stays off")
    _check("configs/scenarios/libero10_nominal.json" in flags[flags.index("--eval_scenario") + 1],
           "in-training evaluation points at the nominal scenario")


def test_unlaunched_ablations_stay_one_change() -> None:
    root = os.path.join(ROOT, "configs", "ablations")
    base = ppo_run_block(load_ppo_run(TASK0))
    actor = load_ppo_run(os.path.join(root, "actor_lr_1e-6.json"))
    kl = load_ppo_run(os.path.join(root, "kl_coef_0.01.json"))
    lora = load_ppo_run(os.path.join(root, "regime_lora.json"))
    _check(ppo_run_block(actor)["actor_lr"] == 1e-6, "the learning-rate ablation must lower the actor rate")
    _check(ppo_run_block(kl)["kl_coef"] == 0.01, "the anchor ablation must set the existing coefficient")
    _check(ppo_run_block(lora)["regime"] == "lora", "the surface ablation must name the lora regime")
    for document, changed in ((actor, "actor_lr"), (kl, "kl_coef"), (lora, "regime")):
        block = ppo_run_block(document)
        diffs = [key for key in base if block[key] != base[key]]
        _check(diffs == [changed], "an ablation may change only %s, changed %s" % (changed, diffs))
        _check(isinstance(document["launched"], bool), "each ablation records a launch flag")
        _check(document["tasks"].startswith("libero_10:0"), "ablations stay on the reproduction task")
    flags = argv_from_ppo_run(lora)
    _check("--control_arm" in flags and "--control_reason" in flags,
           "the lora config must declare itself a control so the trainer will accept it")


def main() -> None:
    test_same_file_writes_the_same_block()
    test_action_steps_other_than_one_are_rejected()
    test_expanded_command_keeps_the_original_update()
    test_unlaunched_ablations_stay_one_change()
    print("NOMINAL_CONFIG_PASS")


if __name__ == "__main__":
    main()
