#!/usr/bin/env python3
# Copyright 2026 the course project authors.
# Measurement harness only. It calls the upstream trainer and evaluator and does not
# change their update, sampler, or reward.
"""One deterministic LIBERO episode, then one stochastic PPO update.

Usage, from the repository root, with the smolvla-flow-rl environment active:

  python scripts/baseline_reproduction.py infer --base-policy /path/to/policy --output-dir runs/baseline/infer
  python scripts/baseline_reproduction.py ppo --base-policy /path/to/policy --output-dir runs/baseline/ppo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from action_model import SmolVLAConfig, SmolVLAForRLActionPrediction  # noqa: E402
from env_libero import build_env, get_task_prompts, set_initial_states  # noqa: E402
from env_processors import make_processors  # noqa: E402
from evaluate import evaluate  # noqa: E402
from gae import compute_gae, normalize_advantages  # noqa: E402
from model_setup import GROUP_KEYS, build_model, build_optimizer  # noqa: E402
from ppo_step import ppo_step_stored  # noqa: E402
from rollout import collect_rollout  # noqa: E402


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this process")
    return torch.device("cuda")


def _peak_mib() -> dict:
    return {
        "allocated_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "reserved_mib": torch.cuda.max_memory_reserved() / (1024 ** 2),
    }


def _frame(obs):
    pixels = obs.get("pixels") if isinstance(obs, dict) else None
    if not isinstance(pixels, dict) or "image" not in pixels:
        return None
    frame = np.asarray(pixels["image"])
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _write_video(frames, path: str) -> str | None:
    """Write frames with matplotlib, which requirements.txt already declares.

    imageio can write the same file and is present in this environment, but it is not a direct requirement.
    The import-closure gate rejects a committed module that imports it.
    """
    if not frames:
        return None
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import animation
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    height, width = frames[0].shape[:2]
    figure = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.set_axis_off()
    shown = axes.imshow(frames[0])

    def _draw(frame):
        shown.set_data(frame)
        return (shown,)

    clip = animation.FuncAnimation(figure, _draw, frames=frames, interval=50, blit=False)
    clip.save(path, writer=animation.FFMpegWriter(fps=20))
    plt.close(figure)
    return path


def _fingerprint(parameters) -> tuple[float, int]:
    total = torch.zeros((), dtype=torch.float64)
    count = 0
    for parameter in parameters:
        total = total + parameter.detach().double().sum().cpu()
        count += parameter.numel()
    return float(total), count


def command_infer(args) -> None:
    device = _device()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    model = SmolVLAForRLActionPrediction(SmolVLAConfig(
        model_path=args.base_policy, num_steps=args.denoise_steps,
        noise_method="flow_ode", noise_level=0.0, device=str(device),
        add_value_head=False)).to(device)
    model.eval()
    load_s = time.perf_counter() - t_load
    load_mem = _peak_mib()
    declared = {
        "images": sorted(k for k in model.lerobot_cfg.input_features if "image" in k),
        "state_dim": int(model.lerobot_cfg.input_features["observation.state"].shape[0]),
        "action_dim": int(model.action_env_dim),
        "chunk_size": int(model.action_chunk),
        "num_steps": int(model.lerobot_cfg.num_steps),
    }
    tasks = [("libero_10", 0)]
    ve, _task_of, _init, suite_limit = build_env(tasks, args.horizon, args.seed)
    env_pre, env_post = make_processors(tasks[0][0], model.lerobot_cfg)
    prompts = get_task_prompts(ve)
    frames = []
    infer_s = []
    original_step = ve.step
    original_sample = model.sample_actions

    def step(action):
        out = original_step(action)
        frame = _frame(out[0])
        if frame is not None:
            frames.append(frame)
        return out

    def sample(*sample_args, **sample_kwargs):
        started = time.perf_counter()
        result = original_sample(*sample_args, **sample_kwargs)
        infer_s.append(time.perf_counter() - started)
        return result

    ve.step = step
    model.sample_actions = sample
    events = os.path.join(args.output_dir, "events.jsonl")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rate = evaluate(
        model, ve, env_pre, env_post, prompts, tasks,
        episodes_per_task=1, seed=args.seed, horizon=args.horizon,
        events_path=events, device=device, n_action_steps=1)
    elapsed = time.perf_counter() - started
    episode_mem = _peak_mib()
    ve.close()
    video = _write_video(frames, os.path.join(args.output_dir, "episode.mp4"))
    record = json.loads(open(events).read().strip().splitlines()[-1])
    summary = {
        "task": "libero_10:0",
        "checkpoint_declared": declared,
        "suite_limit": suite_limit,
        "horizon": args.horizon,
        "seed": args.seed,
        "success": bool(record["success"]),
        "success_step": int(record["succ_step"]),
        "pooled_success_rate": rate,
        "load_seconds": load_s,
        "rollout_seconds": elapsed,
        "inference_calls": len(infer_s),
        "inference_seconds_total": float(sum(infer_s)),
        "inference_seconds_mean": float(sum(infer_s) / len(infer_s)) if infer_s else None,
        "load_gpu_mib": load_mem,
        "episode_gpu_mib": episode_mem,
        "video": video,
        "frames": len(frames),
    }
    path = os.path.join(args.output_dir, "summary.json")
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def _group(name: str) -> str:
    for group, keys in GROUP_KEYS.items():
        if any(key in name for key in keys) or (group == "backbone_adapters" and "lora" in name.lower()):
            return group
    return "other"


def command_ppo(args) -> None:
    device = _device()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    cfg = SmolVLAConfig(
        model_path=args.base_policy, num_steps=args.denoise_steps,
        noise_method="flow_sde", noise_level=args.noise_level, device=str(device),
        add_value_head=True)
    model = build_model(
        SmolVLAForRLActionPrediction, cfg, device, "expert_full_vlm_lora",
        adapter_rank=8, adapter_alpha=16)
    optimizer = build_optimizer(model, args.actor_lr, args.critic_lr)
    load_s = time.perf_counter() - t_load
    tasks = [("libero_10", 0)]
    ve, _task_of, init_obs, suite_limit = build_env(tasks, args.episode_length, args.seed)
    env_pre, env_post = make_processors(tasks[0][0], model.lerobot_cfg)
    prompts = get_task_prompts(ve)
    set_initial_states(ve, "train", ve.num_envs, per_task=1, eval_batch=0, n_init_states=ve.num_envs)
    frozen = [(name, p) for name, p in model.named_parameters() if not p.requires_grad]
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    frozen_before = _fingerprint(p for _, p in frozen)
    trainable_before = _fingerprint(p for _, p in trainable)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train()
    buffer = collect_rollout(
        model, ve, env_pre, env_post, prompts, args.rollout_steps,
        initial_obs=init_obs, device=device, mode="train", n_action_steps=1)
    rollout_s = time.perf_counter() - started
    advantages, returns = compute_gae(
        buffer["rewards"].to(device), buffer["values"].to(device),
        buffer["dones"].to(device), buffer["bootstrap"].to(device),
        gamma=args.gamma, lam_gae=args.gae_lambda)
    advantages = normalize_advantages(advantages)
    update_started = time.perf_counter()
    metrics = ppo_step_stored(
        model, buffer["env_obs"][0], chains=buffer["chains"][:, 0],
        denoise_inds=buffer["denoise_inds"][:, 0], prev_logp=buffer["prev_logprobs"][:, 0],
        noise_level=args.noise_level, optimizer=optimizer,
        advantages=advantages[:, 0], returns=returns[:, 0], n_action_steps=1)
    update_s = time.perf_counter() - update_started
    ve.close()
    frozen_after = _fingerprint(p for _, p in frozen)
    trainable_after = _fingerprint(p for _, p in trainable)
    grad_groups = {name: {"with_grad": 0, "numel": 0} for name in list(GROUP_KEYS) + ["other"]}
    frozen_with_grad = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            bucket = grad_groups[_group(name)]
            bucket["numel"] += parameter.numel()
            if parameter.grad is not None and torch.isfinite(parameter.grad).all() and float(parameter.grad.detach().abs().sum()) > 0:
                bucket["with_grad"] += parameter.numel()
        elif parameter.grad is not None and float(parameter.grad.detach().abs().sum()) > 0:
            frozen_with_grad.append(name)
    summary = {
        "regime": "expert_full_vlm_lora",
        "task": "libero_10:0",
        "suite_limit": suite_limit,
        "episode_length": args.episode_length,
        "rollout_steps": args.rollout_steps,
        "noise_level": args.noise_level,
        "seed": args.seed,
        "load_seconds": load_s,
        "rollout_seconds": rollout_s,
        "update_seconds": update_s,
        "gpu_mib": _peak_mib(),
        "reward_mean": float(buffer["rewards"].mean()),
        "prev_logp_mean": metrics.get("prev_logp_mean"),
        "new_logp_mean": metrics.get("new_logp_mean"),
        "ratio_mean": metrics.get("ratio_mean"),
        "step_skipped": metrics.get("step_skipped"),
        "pg_loss": metrics.get("pg_loss"),
        "v_loss": metrics.get("v_loss"),
        "grad_norm": metrics.get("grad_norm"),
        "n_trainable_with_grad": metrics.get("n_trainable_with_grad"),
        "frozen_fingerprint_before": frozen_before,
        "frozen_fingerprint_after": frozen_after,
        "frozen_unchanged": frozen_before == frozen_after,
        "trainable_fingerprint_before": trainable_before,
        "trainable_fingerprint_after": trainable_after,
        "trainable_changed": trainable_before != trainable_after,
        "grad_groups": grad_groups,
        "frozen_with_grad": frozen_with_grad,
        "note": "One update on a short segment. Reward and success here are not evidence of learning.",
    }
    path = os.path.join(args.output_dir, "summary.json")
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    infer = sub.add_parser("infer")
    infer.add_argument("--base-policy", required=True)
    infer.add_argument("--output-dir", required=True)
    infer.add_argument("--horizon", type=int, default=520)
    infer.add_argument("--denoise-steps", type=int, default=10)
    infer.add_argument("--seed", type=int, default=0)
    ppo = sub.add_parser("ppo")
    ppo.add_argument("--base-policy", required=True)
    ppo.add_argument("--output-dir", required=True)
    ppo.add_argument("--episode-length", type=int, default=4)
    ppo.add_argument("--rollout-steps", type=int, default=2)
    ppo.add_argument("--denoise-steps", type=int, default=10)
    ppo.add_argument("--noise-level", type=float, default=0.1)
    ppo.add_argument("--actor-lr", type=float, default=1e-5)
    ppo.add_argument("--critic-lr", type=float, default=1e-3)
    ppo.add_argument("--gamma", type=float, default=0.99)
    ppo.add_argument("--gae-lambda", type=float, default=0.95)
    ppo.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "infer":
        command_infer(args)
    else:
        command_ppo(args)


if __name__ == "__main__":
    main()
