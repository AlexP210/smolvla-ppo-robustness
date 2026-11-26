#!/usr/bin/env python3
# Copyright 2026 the course project authors.
# Licensed under the Apache License, Version 2.0.
"""CPU checks for eval_scenario_v1. They do not import LIBERO."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from eval_noise import env_reset_seed  # noqa: E402
from eval_scenario import (episode_id, episode_id_rows, load_eval_scenario,  # noqa: E402
                           success_by_interaction, success_by_task)
from seeding import train_reset_seeds  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOMINAL = os.path.join(ROOT, "configs", "scenarios", "libero10_nominal.json")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _bad(**overrides) -> str:
    document = json.loads(open(NOMINAL, encoding="utf-8").read())
    document.update(overrides)
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(document, handle)
    handle.close()
    return handle.name


def test_nominal_file() -> None:
    scenario = load_eval_scenario(NOMINAL)
    _check(scenario["perturbation"] is None, "the nominal scenario must leave perturbations off")
    _check(scenario["action_steps"] == 1 and scenario["horizon"] == 520, "nominal horizon and actions")
    _check(scenario["episodes"] == 10 and scenario["task_ids"] == [0], "nominal coverage")
    _check(scenario["env_seed"] != 0 and scenario["policy_seed"] != scenario["env_seed"],
           "scenario seeds stay distinct from each other and from the training seed 0")


def test_loader_rejects_a_reported_run_that_is_not_one() -> None:
    for overrides, label in (
            ({"action_steps": 10}, "action_steps"),
            ({"horizon": 200}, "horizon"),
            ({"episodes": 8, "init_state_ids": list(range(8))}, "episodes"),
            ({"perturbation": {"schema": "perturbation_v1"}}, "perturbation"),
    ):
        path = _bad(**overrides)
        try:
            try:
                load_eval_scenario(path)
            except ValueError:
                continue
            raise SystemExit("loader accepted a bad %s" % label)
        finally:
            os.remove(path)


def test_episode_ids_follow_the_task_not_the_row() -> None:
    scenario = load_eval_scenario(NOMINAL)
    forward = episode_id_rows(scenario, [("libero_10", 0), ("libero_10", 1)])
    backward = episode_id_rows(scenario, [("libero_10", 1), ("libero_10", 0)])
    _check(forward[0][0] == backward[0][1], "task 0 must keep its episode id when the batch is reordered")
    _check(forward[0][0] % 50 == 0 and forward[3][0] % 50 == 3, "the stored state is the scenario's init id")
    copies = episode_id_rows(scenario, [("libero_10", 0)] * 4)
    _check(len(set(copies[0])) == 4, "four copies of one task must not share a noise id")
    _check(all(value % 50 == 0 for value in copies[0]), "replicates still start at the same stored state")
    _check(episode_id(1, 4) != episode_id(1, 5), "a different initial state changes the id")


def test_eval_seeds_are_not_the_training_stream() -> None:
    scenario = load_eval_scenario(NOMINAL)
    training = train_reset_seeds(0, 0, 1)[0]
    evaluation = env_reset_seed(scenario["env_seed"], episode_id(0, 0), 0)
    _check(training != evaluation, "training reset seeds and scenario reset seeds must differ")


def test_interaction_table_does_not_invent_rates() -> None:
    rows = [
        {"task_idx": 0, "success": True, "succ_step": 150, "horizon": 520},
        {"task_idx": 0, "success": False, "succ_step": -1, "horizon": 520},
        {"task_idx": 1, "success": True, "succ_step": 400, "horizon": 520},
    ]
    per_task = success_by_task(rows, [0, 9])
    _check(per_task[0]["successes"] == 1 and per_task[0]["attempts"] == 2, "task slot 0")
    _check(per_task[1]["successes"] == 1 and per_task[1]["task_id"] == 9, "task slot 1")
    table = success_by_interaction(rows, [100, 200, 520])
    _check(table[0] == {"simulator_steps": 100, "successes": 0, "attempts": 3}, "nothing has succeeded by step 100")
    _check(table[1]["successes"] == 1 and table[2]["successes"] == 2, "later budgets only count successes already observed")


def main() -> None:
    test_nominal_file()
    test_loader_rejects_a_reported_run_that_is_not_one()
    test_episode_ids_follow_the_task_not_the_row()
    test_eval_seeds_are_not_the_training_stream()
    test_interaction_table_does_not_invent_rates()
    print("EVAL_SCENARIO_PASS")


if __name__ == "__main__":
    main()
