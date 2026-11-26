This file provides guidance to AI agents when working with code in this repository.

> Project README: [`README.md`](./README.md). Preserved upstream README: [`README_UPSTREAM.md`](./README_UPSTREAM.md). Attribution: [`NOTICE`](./NOTICE), [`CHANGES_FROM_UPSTREAM.md`](./CHANGES_FROM_UPSTREAM.md). Plan: [`docs/PROJECT_PLAN.md`](./docs/PROJECT_PLAN.md).

`CLAUDE.md` is a symbolic link to this file. Edit this file only.

## Project Overview

Course project on robustness of vision-language-action policies under PPO post-training. The research branch `flow-rl` is the git history of [smolvla_flow_rl](https://github.com/mturan33/smolvla_flow_rl). Tag `flow-rl-baseline` is the unmodified upstream commit `c161af2100610cf9f411bce23ad57180eb771d44`. The LeRobot 0.7.1 history is `archive/lerobot-0.7.1`, the same commit as GitHub's current `main` (`462e4571`; tag `original-baseline`). Do not merge those histories into `flow-rl`.

The planned comparison is three configurations on LIBERO: pretrained `HuggingFaceVLA/smolvla_libero` with deterministic sampling, nominal PPO, and perturbation-aware PPO. Perturbations are not implemented. Do not claim that PPO training works until a run has been executed and recorded.

## Remotes and branches

- `origin`: `git@github.com:AlexP210/smolvla-ppo-robustness.git`
- `upstream`: `https://github.com/mturan33/smolvla_flow_rl`
- `upstream-lerobot`: `https://github.com/huggingface/lerobot` (the archived line only)
- Do not force-push. Do not delete `main` or `archive/lerobot-0.7.1`. GitHub's default branch stays `main` until an admin renames `flow-rl` to `main`. Pull requests for this project target `flow-rl`.

## What not to change

- Do not edit `src/flow_sde.py`, `src/gae.py`, `src/ppo_step.py`, `src/reference_ops.py`, `src/value_head.py`, or `src/action_model.py` to satisfy Ruff, formatting, or CI. Those files are the upstream PPO definition, and several are derivative works of RLinf and LeRobot. Keep their copyright headers and `NOTICE`.
- Do not add perturbations inside those files. A future perturbation belongs on the observation path in `src/rollout.py`, before the policy preprocessor.
- Do not load `lerobot/smolvla_libero` or `lerobot/smolvla_base` into this trainer. The accepted checkpoint is `HuggingFaceVLA/smolvla_libero` (image keys `observation.images.image` and `observation.images.image2`, state 8, action 7).
- Do not install this stack into the LeRobot 0.7.1 conda env named `lerobot`. Use the separate env `smolvla-flow-rl` (Python 3.10). Leave `requirements.txt` and `environment.yml` pins as upstream wrote them until a dependency change is an explicit task.
- Do not register a second PPO implementation beside this one.

## Active CI

[`.github/workflows/ci.yml`](./.github/workflows/ci.yml) runs on pull requests and pushes to `flow-rl` and `main`, and on manual runs. It does not install LeRobot, simulators, CUDA, or checkpoints, and it does not use secrets. From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/check_syntax.py
PYTHONDONTWRITEBYTECODE=1 python scripts/check_import_closure.py
```

`scripts/check_syntax.py` fails if bytecode or a carriage return is present. Do not use `compileall` as the syntax check.

Not in CI, and not yet verified as a training result:

- `python tests/test_invariants.py` — imports the trainer, so it needs PyTorch, NumPy, `peft`, `lerobot[smolvla]>=0.4.4,<0.5`, and `hf-libero`.
- `bash scripts/run_gates.sh` — the two gates above, then the invariant file. Missing `~/.libero/config.yaml` is a failure. The first `import libero.libero` is interactive.
- `bash examples/single_task_contact.sh` and `src/train_flow_rl.py` — GPU, bf16, checkpoint, and LIBERO.

## Layout

Training entry point: `src/train_flow_rl.py` (fourteen required arguments, no defaults). Eval: `src/eval_cli.py`. Sampling: `src/flow_sde.py`. Executed-chunk log-probability and the PPO step: `src/ppo_step.py`. GAE: `src/gae.py`. Critic: `src/value_head.py`. Rollout buffer: `src/rollout.py`. LIBERO harness: `src/env_libero.py`. Trainable regimes: `src/model_setup.py` (`expert_full_vlm_lora` is the upstream recommendation; this project has not measured it). Single horizon: `src/horizon_gate.py`.

LeRobot is a dependency at 0.4.x, imported from the environment, not a directory in this branch. `src/lerobot/` on `archive/lerobot-0.7.1` is a different tree.
