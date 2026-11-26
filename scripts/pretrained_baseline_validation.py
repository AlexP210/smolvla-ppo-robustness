#!/usr/bin/env python3
# Copyright 2026 the course project authors.
# Licensed under the Apache License, Version 2.0.
"""Compare the pretrained SmolVLA checkpoint with native LeRobot 0.4.4 evaluation.

This script does not train, and it does not change the trainer's default action horizon. The Flow RL
rollouts below are a measurement harness: each episode is pinned to one stored initial state, and each
replanning step draws its own standard-normal noise from a seed that does not depend on the horizon.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
import torch
from lerobot.envs.utils import preprocess_observation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from action_model import SmolVLAConfig, SmolVLAForRLActionPrediction  # noqa: E402
from env_libero import (adapt_observation, build_env, get_task_prompts,  # noqa: E402
                        set_initial_states, terminal_success)
from env_processors import make_processors  # noqa: E402
from rollout import exec_action  # noqa: E402

CHECKPOINT = "/user_data/Mozhgan/Projects/RL_Project/checkpoints/HuggingFaceVLA_smolvla_libero"
REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
SUITE = "libero_10"
HORIZON = 520
N_EPISODES = 10
POLICY_SEED = 0
ENV_SEED = 0


def _seed_int(parts: list[int]) -> int:
    sequence = np.random.SeedSequence(parts)
    return int(sequence.generate_state(1, dtype=np.uint32)[0]) % (2**31 - 1)


def flow_noise(policy_seed: int, episode: int, env_step: int, n_envs: int,
               chunk: int, action_dim: int, device) -> torch.Tensor:
    """One standard-normal row per environment, seeded by that row's identity and the step."""
    rows = []
    for index in range(n_envs):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_seed_int([policy_seed, index, episode, env_step, 2]))
        rows.append(torch.normal(
            mean=0.0, std=1.0, size=(chunk, action_dim),
            dtype=torch.float32, generator=generator))
    return torch.stack(rows, dim=0).to(device)


def _versions() -> dict:
    import importlib.metadata as metadata

    def version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "not installed"

    return {
        "python": sys.version.split()[0],
        "lerobot": version("lerobot"),
        "mujoco": version("mujoco"),
        "hf-libero": version("hf-libero"),
        "torch": version("torch"),
        "gymnasium": version("gymnasium"),
        "numpy": version("numpy"),
        "transformers": version("transformers"),
    }


def load_flow(device: str):
    config = SmolVLAConfig(
        model_path=CHECKPOINT, num_steps=10, noise_level=0.0, device=device,
        noise_method="flow_ode", add_value_head=False)
    model = SmolVLAForRLActionPrediction(config).to(device)
    model.eval()
    return model


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def run_parity(out_path: str, device: str) -> None:
    """One shared observation and one shared noise tensor, through both integrators."""
    model = load_flow(device)
    env_pre, env_post = make_processors(SUITE, model.policy.config)
    tasks = [(SUITE, 0)]
    vector, _, _obs, _limit = build_env(tasks, HORIZON, seed=0)
    try:
        set_initial_states(vector, "eval", 1, per_task=1, eval_batch=0, n_init_states=50)
        raw, _info = vector.reset(seed=[0])
        prompts = get_task_prompts(vector)
        env_obs = adapt_observation(raw, env_pre, prompts)
        images, img_masks, lang_tokens, lang_masks, state, processed = model.prep_inputs(env_obs)
        native_obs = preprocess_observation(raw)
        native_obs["task"] = list(prompts)
        native_obs = env_pre(native_obs)
        native_obs = model.preprocessor(native_obs)
        chunk = int(model.lerobot_cfg.chunk_size)
        width = int(model.lerobot_cfg.max_action_dim)
        noise = flow_noise(POLICY_SEED, 0, 0, 1, chunk, width, device)

        with torch.no_grad():
            native = model.policy.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone())
            flow = model.sample_actions(env_obs, mode="eval", noise=noise.clone())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                flow_bf16 = model.sample_actions(env_obs, mode="eval", noise=noise.clone())

        native_env = model.postprocessor(native[:, :, :7])
        flow_env = model.output_transform({"actions": flow["actions"]})["actions"]
        flow_bf16_env = model.output_transform({"actions": flow_bf16["actions"]})["actions"]
        widths = vector.get_attr("observation_width")
        heights = vector.get_attr("observation_height")
        report = {
            "checkpoint": CHECKPOINT,
            "revision": REVISION,
            "versions": _versions(),
            "task": prompts[0],
            "init_state_id": 0,
            "reset_seed": 0,
            "observation_width": int(widths[0]),
            "observation_height": int(heights[0]),
            "raw_image_shape": list(np.asarray(raw["pixels"]["image"]).shape),
            "processed_image_shape": list(processed["observation.images.image"].shape),
            "processed_wrist_shape": list(processed["observation.images.image2"].shape),
            "prepared_image_shape": list(images[0].shape),
            "prepared_wrist_shape": list(images[1].shape),
            "state_before_pad": [float(value) for value in processed["observation.state"].detach().float().cpu().reshape(-1).tolist()],
            "state_shape": list(state.shape),
            "state": [float(value) for value in state.detach().float().cpu().reshape(-1).tolist()],
            "preprocessor_image_max_abs": _max_abs(processed["observation.images.image"], native_obs["observation.images.image"]),
            "preprocessor_wrist_max_abs": _max_abs(processed["observation.images.image2"], native_obs["observation.images.image2"]),
            "preprocessor_state_max_abs": _max_abs(processed["observation.state"], native_obs["observation.state"]),
            "task_matches": processed["task"] == native_obs["task"],
            "resize_imgs_with_padding": list(model.lerobot_cfg.resize_imgs_with_padding),
            "chunk_size": chunk,
            "n_action_steps_in_checkpoint": int(model.lerobot_cfg.n_action_steps),
            "num_steps": int(model.lerobot_cfg.num_steps),
            "max_abs_normalized_fp32": _max_abs(native, flow["actions"]),
            "max_abs_normalized_bf16_vs_native": _max_abs(native, flow_bf16["actions"]),
            "max_abs_unnormalized_fp32": _max_abs(native_env, flow_env),
            "max_abs_unnormalized_bf16_vs_native": _max_abs(native_env, flow_bf16_env),
            "max_abs_executed_action_fp32": _max_abs(native_env[:, 0, :], flow_env[:, 0, :]),
            "max_abs_executed_action_bf16_vs_native": _max_abs(native_env[:, 0, :], flow_bf16_env[:, 0, :]),
        }
    finally:
        vector.close()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({key: report[key] for key in report if key.startswith("max_abs") or key in
                      ("observation_width", "observation_height", "raw_image_shape",
                       "processed_image_shape", "state_shape")}, indent=2))


def run_flow(out_path: str, device: str, horizons: list[int], precision: str) -> None:
    model = load_flow(device)
    env_pre, env_post = make_processors(SUITE, model.policy.config)
    tasks = [(SUITE, index) for index in range(N_EPISODES)]
    vector, _, _obs, suite_limit = build_env(tasks, HORIZON, seed=0)
    prompts = get_task_prompts(vector)
    use_cuda = device.startswith("cuda")
    chunk = int(model.lerobot_cfg.chunk_size)
    width = int(model.lerobot_cfg.max_action_dim)
    results = []
    try:
        for action_steps in horizons:
            started = time.perf_counter()
            inference = 0.0
            per_task = [{"task_id": index, "prompt": prompts[index], "successes": 0,
                         "success_steps": []} for index in range(len(tasks))]
            for episode in range(N_EPISODES):
                set_initial_states(vector, "eval", vector.num_envs, per_task=1,
                                   eval_batch=episode, n_init_states=50)
                raw, _info = vector.reset(seed=[
                    _seed_int([ENV_SEED, index, episode, 1]) for index in range(vector.num_envs)])
                finished = np.zeros(vector.num_envs, dtype=bool)
                solved = np.zeros(vector.num_envs, dtype=bool)
                success_step = np.full(vector.num_envs, -1, dtype=int)
                env_step = 0
                while env_step < HORIZON and not finished.all():
                    obs = adapt_observation(raw, env_pre, prompts)
                    noise = flow_noise(POLICY_SEED, episode, env_step, vector.num_envs, chunk, width, device)
                    if use_cuda:
                        torch.cuda.synchronize()
                    tick = time.perf_counter()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                        enabled=use_cuda and precision == "bf16"):
                        sampled = model.sample_actions(obs, mode="eval", noise=noise)
                    if use_cuda:
                        torch.cuda.synchronize()
                    inference += time.perf_counter() - tick
                    for offset in range(action_steps):
                        if env_step >= HORIZON or finished.all():
                            break
                        raw, _reward, term, trunc, info = vector.step(
                            exec_action(model, sampled, env_post, step=offset))
                        term_a = np.asarray(term).reshape(-1).astype(bool)
                        trunc_a = np.asarray(trunc).reshape(-1).astype(bool)
                        hit = terminal_success(term_a, info).astype(bool)
                        first = hit & ~finished & (success_step < 0)
                        success_step[first] = env_step + 1
                        solved |= hit & ~finished
                        finished |= term_a | trunc_a
                        env_step += 1
                for index in range(vector.num_envs):
                    per_task[index]["successes"] += int(solved[index])
                    per_task[index]["success_steps"].append(int(success_step[index]))
            successes = sum(item["successes"] for item in per_task)
            attempts = N_EPISODES * len(tasks)
            row = {
                "action_steps": action_steps,
                "precision": precision,
                "episodes_per_task": N_EPISODES,
                "horizon": HORIZON,
                "suite_limit": suite_limit,
                "init_state_ids": list(range(N_EPISODES)),
                "policy_seed": POLICY_SEED,
                "env_seed": ENV_SEED,
                "successes": successes,
                "attempts": attempts,
                "rate": successes / attempts,
                "inference_seconds": round(inference, 3),
                "wall_seconds": round(time.perf_counter() - started, 3),
                "per_task": per_task,
            }
            results.append(row)
            print("flow action_steps=%d precision=%s success=%d/%d wall=%.1fs inference=%.1fs" % (
                action_steps, precision, successes, attempts, row["wall_seconds"], inference), flush=True)
    finally:
        vector.close()
    payload = {"checkpoint": CHECKPOINT, "revision": REVISION, "versions": _versions(),
               "prompts": prompts, "results": results}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)


def run_native(out_dir: str, n_action_steps: int) -> None:
    """Call the installed LeRobot 0.4.4 evaluator. This process does not import the Flow RL sampler."""
    os.makedirs(out_dir, exist_ok=True)
    command = [
        sys.executable, "-m", "lerobot.scripts.lerobot_eval",
        "--policy.path=%s" % CHECKPOINT,
        "--policy.n_action_steps=%d" % n_action_steps,
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--env.type=libero",
        "--env.task=%s" % SUITE,
        "--env.max_parallel_tasks=1",
        "--eval.batch_size=1",
        "--eval.n_episodes=%d" % N_EPISODES,
        "--eval.use_async_envs=false",
        "--seed=0",
        "--output_dir=%s" % out_dir,
    ]
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.perf_counter()
    log_path = os.path.join(out_dir, "native_eval.log")
    with open(log_path, "w") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - started
    summary = {"returncode": completed.returncode, "wall_seconds": round(wall, 3),
               "n_action_steps": n_action_steps, "command": command}
    with open(os.path.join(out_dir, "native_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    if completed.returncode != 0:
        raise SystemExit("native evaluator exited %d; see %s" % (completed.returncode, log_path))
    print("native action_steps=%d wall=%.1fs" % (n_action_steps, wall), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("parity", "flow", "native"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--horizons", default="1,5,10,25,50")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--n-action-steps", type=int, default=10)
    args = parser.parse_args()
    if args.mode == "parity":
        run_parity(args.out, args.device)
    elif args.mode == "flow":
        horizons = [int(item) for item in args.horizons.split(",") if item]
        run_flow(args.out, args.device, horizons, args.precision)
    else:
        run_native(args.out, args.n_action_steps)


if __name__ == "__main__":
    main()
