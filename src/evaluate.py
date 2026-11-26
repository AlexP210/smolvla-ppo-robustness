# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Deterministic evaluation that writes one record per episode.

Three properties make an evaluation comparable across runs, and all three are decisions rather than accidents.

Sampling is deterministic. Every integration step is taken without noise, so two evaluations of the same weights
under the same seed produce the same episodes, and a difference between two evaluations is therefore a difference
in the weights. Evaluating with exploration noise on measures the noise as much as the policy.

Initial states are pinned by batch index rather than drawn. The same batch index gives the same starting
configuration in every run, which is what makes two evaluation runs comparable episode by episode.

The step at which an episode succeeded is recorded, not merely whether it did. That single number lets one
evaluation be read at several budgets afterwards, so asking how the policy does under a shorter horizon does not
require running anything again. It also separates two episode records that share an outcome but not a
trajectory, which the outcome alone does not.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from .env_libero import adapt_observation, set_initial_states, terminal_success
    from .eval_noise import env_reset_seed, initial_flow_noise, resolve_episode_ids
    from .rollout import exec_action
except ImportError:
    from env_libero import adapt_observation, set_initial_states, terminal_success
    from eval_noise import env_reset_seed, initial_flow_noise, resolve_episode_ids
    from rollout import exec_action

N_INIT_STATES = 50  # how many stored starting configurations the suite provides


@torch.no_grad()
def evaluate(model, ve, env_pre, env_post, task_prompts, tasks: Sequence[Tuple[str, int]], *,
             episodes_per_task: int, seed: int, horizon: int, device,
             events_path: Optional[str] = None, n_action_steps: int = 1,
             cycle: Optional[int] = None,
             env_seed: Optional[int] = None, policy_seed: Optional[int] = None,
             episode_ids: Optional[Sequence[int]] = None,
             episode_ids_by_batch: Optional[Sequence[Sequence[int]]] = None) -> float:
    """Run a fixed number of episodes per task and return the pooled success rate.

    One episode per task runs in each batch, so the batch index and the task together identify an episode. Every
    episode is written out as it finishes, which means a run interrupted halfway leaves a file that is honestly
    short rather than one that looks complete.

    The cycle label is written into every record and identifies which set of weights produced it. A run that
    evaluates periodically appends to the same file, so without it the records of two checkpoints are
    indistinguishable, and a reader keyed on task and batch alone silently keeps whichever arrived last while
    reporting the file as complete.

    `env_seed` and `policy_seed` are optional and independent. Leave both unset and the reset rule and the
    global noise draw are the historical ones, including their dependence on the batch width. Set them and each
    episode id gets its own reset seed and its own standard-normal flow noise, neither of which depends on the
    other ids in the batch. `episode_ids_by_batch` replaces that id list on each outer episode, which is how a
    scenario gives every episode its own stored initial state. The integration itself is still the deterministic
    Euler step.

    `horizon` counts ENVIRONMENT STEPS, the same unit the episode cap uses, and so does the recorded success step.
    That has to be said because the loop below is over decisions, and each decision executes several environment
    steps: measuring the budget in one unit and capping the episode in the other silently shortens every episode
    by that factor while the record still claims the full length, and any budget read off the file afterwards is
    then wrong by the same factor.
    """
    n_envs = ve.num_envs
    was_training = model.training
    prev_method = model.config.noise_method
    prev_steps = model.lerobot_cfg.num_steps
    model.eval()
    model.config.noise_method = "flow_ode"

    use_cuda = torch.device(device).type == "cuda"
    # Offset so that evaluation episodes never draw the same seeds as collection does. The value is arbitrary;
    # only its being fixed and far from zero matters.
    eval_seed_base = seed + 7000
    successes = np.zeros(len(tasks), dtype=np.int64)
    attempts = np.zeros(len(tasks), dtype=np.int64)

    pinned = env_seed is not None or policy_seed is not None
    ids = resolve_episode_ids(episode_ids, n_envs) if pinned else None
    if episode_ids_by_batch is not None:
        if env_seed is None:
            raise ValueError("episode_ids_by_batch requires env_seed, so the stored initial state follows the ids")
        if len(episode_ids_by_batch) != episodes_per_task:
            raise ValueError("episode_ids_by_batch has %d rows and episodes_per_task is %d"
                             % (len(episode_ids_by_batch), episodes_per_task))
    steps_per_decision = max(1, int(n_action_steps))
    n_decisions = max(1, int(horizon) // steps_per_decision)
    if int(horizon) % steps_per_decision:
        print("  [eval] horizon %d is not a whole number of decisions at %d steps each; scoring %d decisions, "
              "which is %d environment steps" % (horizon, steps_per_decision, n_decisions,
                                                 n_decisions * steps_per_decision), flush=True)

    try:
        for batch in range(episodes_per_task):
            row_ids = ids
            if episode_ids_by_batch is not None:
                row_ids = resolve_episode_ids(episode_ids_by_batch[batch], n_envs)
            if env_seed is None:
                set_initial_states(ve, "eval", n_envs, per_task=1, eval_batch=batch, n_init_states=N_INIT_STATES)
                reset_seeds = [eval_seed_base + batch * n_envs + i for i in range(n_envs)]
            else:
                # One initial state per episode id. The previous rule gave every worker in a batch the same
                # id, and the reset seed grew with the batch width, so removing a worker changed the others.
                reset_seeds = [env_reset_seed(env_seed, row_ids[i], batch) for i in range(n_envs)]
                ve.set_attr("init_state_id", [int(row_ids[i]) % N_INIT_STATES for i in range(n_envs)])
                ve.set_attr("_reset_stride", [0] * n_envs)
            raw, _ = ve.reset(seed=reset_seeds)

            solved = np.zeros(n_envs, dtype=bool)
            finished = np.zeros(n_envs, dtype=bool)
            success_step = np.full(n_envs, -1, dtype=int)

            for t in range(n_decisions):
                obs = adapt_observation(raw, env_pre, task_prompts)
                noise = None
                if policy_seed is not None:
                    noise = initial_flow_noise(
                        policy_seed, row_ids, t,
                        int(model.lerobot_cfg.chunk_size), int(model.lerobot_cfg.max_action_dim),
                        batch=batch, device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
                    sampled = model.sample_actions(obs, mode="eval", noise=noise)
                for k in range(steps_per_decision):
                    raw, _r, term, trunc, info = ve.step(exec_action(model, sampled, env_post, step=k))
                    term_a = np.asarray(term).reshape(-1).astype(bool)
                    trunc_a = np.asarray(trunc).reshape(-1).astype(bool)
                    hit = terminal_success(term_a, info).astype(bool)
                    first = hit & ~finished & (success_step < 0)
                    # Environment steps, not decisions, so that a budget read off this file later means what the
                    # episode cap and the horizon argument mean.
                    success_step[first] = t * steps_per_decision + k + 1
                    solved |= hit & ~finished
                    finished |= term_a | trunc_a
                if finished.all():
                    break

            for i in range(n_envs):
                successes[i] += int(solved[i])
                attempts[i] += 1
            if events_path:
                os.makedirs(os.path.dirname(events_path) or ".", exist_ok=True)
                with open(events_path, "a") as fh:
                    for i in range(n_envs):
                        record = {
                            "cycle": None if cycle is None else int(cycle),
                            "task_idx": int(i), "batch": int(batch), "env": int(i),
                            "seed": int(reset_seeds[i]),
                            "success": bool(solved[i]), "succ_step": int(success_step[i]),
                        }
                        if pinned:
                            record["episode_id"] = int(row_ids[i])
                            record["init_state_id"] = int(row_ids[i]) % N_INIT_STATES
                            record["task_id"] = int(tasks[i][1])
                            record["env_seed"] = None if env_seed is None else int(env_seed)
                            record["policy_seed"] = None if policy_seed is None else int(policy_seed)
                        # Both units, written down, so no reader has to work out which one a number is in.
                        record["horizon"] = int(n_decisions * steps_per_decision)
                        record["decisions"] = int(n_decisions)
                        record["steps_per_decision"] = int(steps_per_decision)
                        fh.write(json.dumps(record) + "\n")
    finally:
        model.config.noise_method = prev_method
        model.lerobot_cfg.num_steps = prev_steps
        if was_training:
            model.train()

    rates = [successes[i] / max(1, attempts[i]) for i in range(len(tasks))]
    return float(sum(rates) / len(rates))
