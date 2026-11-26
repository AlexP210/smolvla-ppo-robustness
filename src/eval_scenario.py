# Copyright 2026 the course project authors.
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Fixed evaluation scenarios, shared with Workstream C.

The object is `eval_scenario_v1` from docs/PROJECT_PLAN.md. Workstream C owns later
seen, unseen, and severity files. This module is the loader and the episode-id
rule those files use. It does not sample a perturbation and it does not import
the simulator.

An episode id is a function of the task id, the stored initial-state id, and a
replicate index. The replicate is zero when each task appears once. Repeated
copies, which the trainer uses to widen the PPO batch, get the next index so
they do not share a noise row. The stored LIBERO state is `episode_id % 50`,
the same modulus `evaluate` applies. Training reset seeds live in `seeding.py`
and are not this function.
"""

from __future__ import annotations

import json
from typing import Sequence

SCHEMA = "eval_scenario_v1"
INIT_STATE_STRIDE = 50  # matches evaluate.N_INIT_STATES
REPLICATE_STRIDE = 500  # divisible by the stride, so it does not change the stored state

_REQUIRED = (
    "schema", "suite", "task_ids", "episodes", "init_state_ids",
    "env_seed", "policy_seed", "horizon", "action_steps", "perturbation",
)


def load_eval_scenario(path: str) -> dict:
    """Load a scenario used for a reported run. Rejects a file that cannot be one."""
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("a scenario file must contain one JSON object")
    missing = [key for key in _REQUIRED if key not in document]
    if missing:
        raise ValueError("scenario is missing %s" % ", ".join(missing))
    if document["schema"] != SCHEMA:
        raise ValueError("scenario schema is %r, and a reported run wants %s" % (document["schema"], SCHEMA))
    if int(document["action_steps"]) != 1:
        raise ValueError("a reported scenario keeps action_steps at 1, got %r" % (document["action_steps"],))
    episodes = int(document["episodes"])
    if episodes < 10:
        raise ValueError("a reported scenario needs at least 10 episodes, got %d" % episodes)
    if document["suite"] == "libero_10" and int(document["horizon"]) != 520:
        raise ValueError("libero_10 scenarios use horizon 520, got %r" % (document["horizon"],))
    init_state_ids = document["init_state_ids"]
    task_ids = document["task_ids"]
    if not isinstance(init_state_ids, list) or len(init_state_ids) != episodes:
        raise ValueError("init_state_ids must list one stored state per episode")
    if not isinstance(task_ids, list) or len(task_ids) < 1:
        raise ValueError("task_ids must name at least one task")
    if any(int(state) < 0 or int(state) >= INIT_STATE_STRIDE for state in init_state_ids):
        raise ValueError("init_state_ids must lie in 0..%d" % (INIT_STATE_STRIDE - 1))
    if document["perturbation"] is not None:
        raise ValueError("this slice accepts perturbation null only")
    return document


def episode_id(task_id: int, init_state_id: int, replicate: int = 0) -> int:
    """Identity of one episode. Independent of the worker's position in a batch."""
    if int(replicate) < 0:
        raise ValueError("replicate must be non-negative, got %r" % (replicate,))
    return int(init_state_id) + int(task_id) * INIT_STATE_STRIDE + int(replicate) * REPLICATE_STRIDE


def episode_id_rows(scenario: dict, tasks: Sequence[tuple]) -> list:
    """One row of episode ids per scenario episode, in the vector's worker order.

    `tasks` is the `(suite, task index)` list the vector environment was built with.
    Repeated task indexes receive replicate 0, 1, 2, ... in that order.
    """
    replicates: dict = {}
    assignment = []
    for task in tasks:
        task_id = int(task[1])
        replicate = replicates.get(task_id, 0)
        replicates[task_id] = replicate + 1
        assignment.append((task_id, replicate))
    rows = []
    for init_state_id in scenario["init_state_ids"]:
        rows.append([episode_id(task_id, int(init_state_id), replicate)
                     for task_id, replicate in assignment])
    return rows


def assert_scenario_covers(scenario: dict, tasks: Sequence[tuple]) -> None:
    """The workers about to be scored must be tasks this scenario named."""
    suites = {task[0] for task in tasks}
    if suites != {scenario["suite"]}:
        raise ValueError("scenario suite is %s and the vector is %s" % (scenario["suite"], sorted(suites)))
    allowed = {int(task_id) for task_id in scenario["task_ids"]}
    present = {int(task[1]) for task in tasks}
    if not present <= allowed:
        raise ValueError("vector tasks %s are outside the scenario tasks %s"
                         % (sorted(present), sorted(allowed)))


def success_by_task(rows: Sequence[dict], task_ids: Sequence[int]) -> list:
    """Per-task counts from evaluate()'s event records. Missing tasks stay at zero attempts."""
    report = []
    for slot, task_id in enumerate(task_ids):
        mine = [row for row in rows if int(row.get("task_idx", -1)) == slot]
        steps = [int(row["succ_step"]) for row in mine if row.get("success")]
        report.append({
            "task_id": int(task_id),
            "successes": int(sum(1 for row in mine if row.get("success"))),
            "attempts": len(mine),
            "success_steps": steps,
        })
    return report


def success_by_interaction(rows: Sequence[dict], budgets: Sequence[int]) -> list:
    """How many episodes have already succeeded at or before each simulator step.

    An episode with no success contributes to the attempt count and not to the
    numerator. Budgets the log does not contain are not filled with a guessed rate.
    """
    table = []
    for budget in budgets:
        successes = 0
        attempts = 0
        for row in rows:
            horizon = int(row.get("horizon", 0))
            step = int(row.get("succ_step", -1))
            succeeded_in_time = bool(row.get("success")) and 0 < step <= int(budget)
            if horizon < int(budget) and not succeeded_in_time:
                continue
            attempts += 1
            if succeeded_in_time:
                successes += 1
        table.append({"simulator_steps": int(budget), "successes": successes, "attempts": attempts})
    return table
