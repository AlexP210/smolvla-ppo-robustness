#!/usr/bin/env python3
# Copyright 2026 the course project authors.
# Calls the upstream evaluator. It does not change the update, the sampler, or the reward.
"""Matched deterministic evaluation, and a first-step probe of seed 7000.

Both commands use src/evaluate.py. The comparison builds the policy the way
src/eval_cli.py does: expert_full_vlm_lora, rank 8, alpha 16, value head on,
then optionally loads one checkpoint with load_policy_only. The initialization
is the pretrained weights plus zero-initialized language adapters. The probe
uses the plain pretrained policy, with no adapters and no value head, because
that is the policy whose single-environment and ten-task runs disagreed.

Usage, from the repository root:

  python scripts/matched_checkpoint_eval.py probe-seed --base-policy /path/to/policy --output-dir runs/matched/probe
  python scripts/matched_checkpoint_eval.py compare --base-policy /path/to/policy --checkpoint runs/.../step_000020.pt --output-dir runs/matched/compare
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from action_model import SmolVLAConfig, SmolVLAForRLActionPrediction  # noqa: E402
from checkpoint import load_policy_only  # noqa: E402
from env_libero import adapt_observation, build_env, get_task_prompts, set_initial_states  # noqa: E402
from env_processors import make_processors  # noqa: E402
from evaluate import N_INIT_STATES, evaluate  # noqa: E402
from model_setup import build_model  # noqa: E402
from rollout import exec_action  # noqa: E402
from train_flow_rl import ADAPTER_ALPHA_DEFAULT, ADAPTER_RANK_DEFAULT  # noqa: E402


TASKS = "libero_10:0,libero_10:0,libero_10:0,libero_10:0"
HORIZON = 520
EPISODES = 2
SEED = 0
DENOISE = 10
ACTION_STEPS = 1


def _as_float(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _summary_vector(value) -> dict:
    array = _as_float(value)
    return {
        "shape": list(array.shape),
        "sum": float(array.sum()),
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
    }


def _max_abs(left, right) -> float:
    return float(np.max(np.abs(_as_float(left) - _as_float(right))))


def _plain_policy(base_policy: str, device: torch.device):
    model = SmolVLAForRLActionPrediction(SmolVLAConfig(
        model_path=base_policy, num_steps=DENOISE, noise_method="flow_ode",
        noise_level=0.0, device=str(device), add_value_head=False)).to(device)
    model.eval()
    model.config.noise_method = "flow_ode"
    return model


def _first_decision(model, tasks, device):
    """One reset and one sample, using the same seed and initial-state rule as evaluate."""
    ve, _, _, _ = build_env(tasks, HORIZON, SEED)
    try:
        n_envs = ve.num_envs
        set_initial_states(ve, "eval", n_envs, per_task=1, eval_batch=0, n_init_states=N_INIT_STATES)
        seeds = [7000 + i for i in range(n_envs)]
        raw, _ = ve.reset(seed=seeds)
        prompts = get_task_prompts(ve)
        env_pre, env_post = make_processors(tasks[0][0], model.lerobot_cfg)
        obs = adapt_observation(raw, env_pre, prompts)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            sampled = model.sample_actions(obs, mode="eval")
        action = exec_action(model, sampled, env_post, step=0)
        within = {"image": 0.0, "state": 0.0, "action": 0.0}
        for index in range(1, n_envs):
            within["image"] = max(within["image"], _max_abs(obs["main_images"][index], obs["main_images"][0]))
            within["state"] = max(within["state"], _max_abs(obs["states"][index], obs["states"][0]))
            within["action"] = max(within["action"], _max_abs(action[index], action[0]))
        return {
            "tasks": ["%s:%d" % pair for pair in tasks],
            "init_state_id": [int(v) for v in ve.get_attr("init_state_id")],
            "seeds": seeds,
            "prompts": prompts,
            "image_sum": _summary_vector(obs["main_images"][0]),
            "state": _as_float(obs["states"][0]).reshape(-1).tolist(),
            "action": _as_float(action[0]).reshape(-1).tolist(),
            "image": obs["main_images"][0],
            "state_tensor": obs["states"][0],
            "action_row": action[0],
            "within_batch_max_abs_vs_row0": within,
        }
    finally:
        ve.close()


def command_probe(args) -> None:
    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)
    model = _plain_policy(args.base_policy, device)
    conditions = {
        "single": [("libero_10", 0)],
        "four_copies": [("libero_10", 0)] * 4,
        "ten_tasks": [("libero_10", index) for index in range(10)],
    }
    measured = {name: _first_decision(model, tasks, device) for name, tasks in conditions.items()}
    reference = measured["single"]
    report = {"policy": "pretrained, no value head, no adapters", "noise_method": "flow_ode",
              "autocast": "bfloat16", "conditions": {}}
    for name, row in measured.items():
        report["conditions"][name] = {
            "tasks": row["tasks"],
            "init_state_id": row["init_state_id"],
            "seeds": row["seeds"],
            "prompt0": row["prompts"][0],
            "image_sum": row["image_sum"],
            "state": row["state"],
            "action": row["action"],
            "max_abs_image_vs_single": _max_abs(row["image"], reference["image"]),
            "max_abs_state_vs_single": _max_abs(row["state_tensor"], reference["state_tensor"]),
            "max_abs_action_vs_single": _max_abs(row["action_row"], reference["action_row"]),
            "within_batch_max_abs_vs_row0": row["within_batch_max_abs_vs_row0"],
        }
    path = os.path.join(args.output_dir, "seed7000_first_step.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


def _training_policy(base_policy: str, device: torch.device, checkpoint: str | None):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cfg = SmolVLAConfig(
        model_path=base_policy, num_steps=DENOISE, noise_method="flow_ode",
        noise_level=0.0, device=str(device), add_value_head=True)
    model = build_model(
        SmolVLAForRLActionPrediction, cfg, device, "expert_full_vlm_lora",
        ADAPTER_RANK_DEFAULT, ADAPTER_ALPHA_DEFAULT, enforce_surface=False)
    loaded = None
    if checkpoint:
        step, changed, extra = load_policy_only(checkpoint, model)
        loaded = {"step": step, "tensors_changed": changed, "extra": extra}
    return model, loaded


def _run_eval(model, output_dir: str, cycle: int) -> dict:
    tasks = [("libero_10", 0)] * 4
    ve, _, _, suite_limit = build_env(tasks, HORIZON, SEED)
    env_pre, env_post = make_processors(tasks[0][0], model.lerobot_cfg)
    prompts = get_task_prompts(ve)
    events = os.path.join(output_dir, "events.jsonl")
    if os.path.exists(events):
        os.remove(events)
    try:
        rate = evaluate(
            model, ve, env_pre, env_post, prompts, tasks,
            episodes_per_task=EPISODES, seed=SEED, horizon=HORIZON,
            events_path=events, device=next(model.parameters()).device,
            n_action_steps=ACTION_STEPS, cycle=cycle,
            env_seed=SEED, policy_seed=SEED,
            episode_ids=list(range(4)))
    finally:
        ve.close()
    rows = [json.loads(line) for line in open(events)]
    return {
        "suite_limit": suite_limit,
        "pooled_success_rate": rate,
        "episodes": rows,
        "successes": int(sum(bool(row["success"]) for row in rows)),
        "attempts": len(rows),
    }


def command_compare(args) -> None:
    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "tasks": TASKS,
        "episodes_per_task": EPISODES,
        "horizon": HORIZON,
        "seed": SEED,
        "eval_seeds": list(range(7000, 7008)),
        "n_action_steps": ACTION_STEPS,
        "denoise_steps": DENOISE,
        "noise_method": "flow_ode",
        "regime": "expert_full_vlm_lora",
        "adapter_rank": ADAPTER_RANK_DEFAULT,
        "adapter_alpha": ADAPTER_ALPHA_DEFAULT,
        "env_seed": SEED,
        "policy_seed": SEED,
        "episode_ids": [0, 1, 2, 3],
        "note": "Same evaluate() call, four copies of libero_10 task 0, with a shared environment seed and a shared policy seed. Initialization is the pretrained policy with zero-initialized adapters.",
    }
    for label, checkpoint, cycle in (
            ("initialization", None, 0),
            ("step_000020", args.checkpoint, 20)):
        model, loaded = _training_policy(args.base_policy, device, checkpoint)
        out = os.path.join(args.output_dir, label)
        os.makedirs(out, exist_ok=True)
        result = _run_eval(model, out, cycle)
        result["checkpoint"] = checkpoint
        result["load"] = loaded
        report[label] = result
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    path = os.path.join(args.output_dir, "comparison.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe-seed")
    probe.add_argument("--base-policy", required=True)
    probe.add_argument("--output-dir", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--base-policy", required=True)
    compare.add_argument("--checkpoint", required=True)
    compare.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "probe-seed":
        command_probe(args)
    else:
        command_compare(args)


if __name__ == "__main__":
    main()
