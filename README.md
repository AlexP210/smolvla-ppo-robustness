# Robustness of Vision-Language-Action Models through Reinforcement Learning Post-Training

This repository is a course research project on reinforcement-learning post-training for vision-language-action (VLA) policies. The training code is the history of [smolvla_flow_rl](https://github.com/mturan33/smolvla_flow_rl) (Mehmet Turan Yardimci). The upstream README is preserved, unmodified, in [README_UPSTREAM.md](./README_UPSTREAM.md). The tag `flow-rl-baseline` is that upstream commit, `c161af2100610cf9f411bce23ad57180eb771d44`, before the documentation and CI added for this project.

The previous LeRobot 0.7.1 line is not in this branch. It is `archive/lerobot-0.7.1`. The tag `original-baseline` (`b07adcc5e3ce42c26a248236dbb3571f84c28c43`) marks the LeRobot commit that line started from. The two histories are not merged.

A passing CI check is not evidence that training works. The measurements that do exist are separated below. Robustness perturbations are still not implemented.

## What has been measured

Reports: [docs/BASELINE_REPRODUCTION.md](./docs/BASELINE_REPRODUCTION.md), [docs/NOMINAL_PPO_PILOT.md](./docs/NOMINAL_PPO_PILOT.md), and [docs/PRETRAINED_BASELINE_VALIDATION.md](./docs/PRETRAINED_BASELINE_VALIDATION.md). The policy predicts 50 actions per call. Upstream training and evaluation execute one of them (`--action_steps 1`, and `n_action_steps` is 1 in the checkpoint). That training default was not changed.

These four statements are different, and only the first three have been observed:

| Claim | Status |
| --- | --- |
| Pretrained inference runs | One deterministic `libero_10` task 0 episode succeeded, at environment step 369 of a 520-step cap. A later 10-episode evaluation of every LIBERO-Long task with the native LeRobot 0.4.4 evaluator scored 53/100 at the checkpoint's `n_action_steps=1` and 46/100 at `n_action_steps=10`. The leaderboard's Long column is 0.60. The details are in the validation report. |
| PPO optimization runs | One stored-versus-recomputed update had ratio 1.0 before any parameter moved. Gradients reached the action expert, action head, and critic. Frozen parameters, including the vision tower, did not change. |
| Multi-cycle training collects reward | Twenty upstream updates on four copies of task 0 finished. The first-step ratio was 1.0. Three collection windows each contained one sparse success. |
| PPO improves task success | Not established. On the same eight seeds, the initialization succeeded twice and the checkpoint after 20 updates succeeded zero times. Eight episodes do not measure a change in success. |

The upstream execution path is unchanged: each policy call is a new flow sample, and positions 1–49 of the chunk are discarded. Likelihood, reward, and GAE are computed per executed action, so with the default they see one simulator step per decision. On one shared observation and one shared noise tensor, that float32 integration matches LeRobot's own sampler exactly. The validation report also runs horizons 1, 5, 10, 25, and 50 without making any of them the trainer default.

## Project Title

Robustness of Vision-Language-Action Models through Reinforcement Learning Post-Training

## Overview

Supervised VLA policies imitate demonstration data. A policy that succeeds on the nominal training distribution can still fail when appearance, object placement, or physical conditions shift. Task success on the original distribution does not, by itself, measure that sensitivity.

This project studies reinforcement-learning post-training as a way to improve both nominal task performance and robustness under those shifts. The testbed is a pretrained SmolVLA policy in simulation on LIBERO. The trainer in this tree is proximal policy optimization on a flow-matching action head: one integration step is stochastic, the log-probability is restricted to the executed part of the action chunk, advantages come from generalized advantage estimation on a sparse success reward, and the critic is a separate value head. That code is upstream. The runs in [docs/BASELINE_REPRODUCTION.md](./docs/BASELINE_REPRODUCTION.md) and [docs/NOMINAL_PPO_PILOT.md](./docs/NOMINAL_PPO_PILOT.md) show that this trainer can be executed and can observe a sparse reward. They do not show that PPO improves success. The upstream README reports no success rate of its own, and the clip in that README is footage rather than a measurement.

## Research Questions

- Does nominal RL post-training improve robustness beyond nominal task success?
- Does exposure to perturbations during RL training improve robustness?
- Do robustness improvements generalize to unseen perturbations?
- What are the trade-offs between nominal performance, robustness, and learning efficiency?

## Methodology

The planned comparison uses three policy configurations on LIBERO:

- **Pretrained supervised SmolVLA without RL.** Load `HuggingFaceVLA/smolvla_libero` and evaluate with deterministic integration (`mode="eval"` in `src/flow_sde.py`). No reinforcement-learning update. One episode has been run. A success rate has not.
- **Nominal PPO post-training.** The same initialization, then the upstream trainer (`src/train_flow_rl.py`) on unperturbed LIBERO tasks. A 20-update pilot on one task has been run. It collected reward. It did not establish a gain in success.
- **Perturbation-aware PPO post-training.** The same initialization, then PPO while training episodes include the perturbations below. Not implemented. The upstream rollout has no perturbation hook.

The upstream reward is sparse and terminal (`terminal_success` in `src/env_libero.py`): 1 when the task is solved on that step, otherwise 0. There is no shaping term. `src/gae.py` computes advantages over fixed-length segments and uses a bootstrap value when a segment ends before the episode does. `src/ppo_step.py` applies a clipped policy surrogate, a dual clip, and a clipped Huber value loss to the stored trajectory. `src/horizon_gate.py` refuses a run whose training horizon and evaluation horizon disagree.

SmolVLA itself is not copied into this repository. It is imported from LeRobot at the version pinned in `requirements.txt` (`lerobot[smolvla]>=0.4.4,<0.5`). Four files are derivative works of RLinf and, for two of them, of LeRobot 0.4.4. Their headers, `NOTICE`, and `CHANGES_FROM_UPSTREAM.md` record that. Those files are unchanged in the migration commit.

## Environment Perturbations

Perturbations are planned and are not implemented. The intended families are:

- **Visual.** Changes to appearance, such as color, texture, or lighting. Appearance changes are a first priority.
- **Spatial.** Changes to the initial object position. Initial-position changes are a first priority.
- **Physical.** Changes to dynamics or object properties. These are in scope after the appearance and initial-position perturbations.

Each family is meant to be split later into perturbations used during training ("seen") and perturbations held out for testing ("unseen"). The specific parameter ranges are not fixed yet. When they are added, they belong on the observation path in `src/rollout.py` (`adapt_observation` / `collect_rollout`), not inside `src/flow_sde.py`, `src/gae.py`, or `src/ppo_step.py`.

## Evaluation

Evaluation of the three configurations is planned. The intended protocol is:

- **Nominal.** Success on unperturbed LIBERO tasks, deterministic sampling, the same horizon as training.
- **Seen perturbations.** Success under perturbations that perturbation-aware training was allowed to use.
- **Unseen perturbations.** Success under held-out perturbations.

Reported quantities, once the robustness experiments exist, will be task success rate, robustness degradation relative to the nominal evaluation, learning curves, and training efficiency. The numbers recorded so far are the execution checks in the two reports linked above. They are not those robustness results. The upstream README does not report success rates either.

The upstream scoring entry point is `src/eval_cli.py`. It refuses a success rate until `src/completeness_gate.py` accepts the log, and it refuses a budget longer than the horizon that was actually run. `scripts/run_gates.sh` is the upstream mechanical check. It is not the robustness protocol above. On the `smolvla-flow-rl` environment it has been run, including the invariant file, and it passed. That pass is a software check. It is not a LIBERO result.

## Checkpoint

The trainer's download helper defaults to `HuggingFaceVLA/smolvla_libero`, revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` (`scripts/download_base_policy.py`). The harness requires image keys `observation.images.image` and `observation.images.image2`, an 8-dimensional state, and 7-dimensional actions (`REQUIRED_IMAGE_KEYS`, `REQUIRED_STATE_DIM`, and `REQUIRED_ACTION_DIM` in `src/action_model.py`).

`lerobot/smolvla_libero` is a different Hub repository. Its config declares `camera1` and `camera2`. `src/action_model.py` refuses that checkpoint. `lerobot/smolvla_base` is also refused: the helper's own comments say the base checkpoint declares different cameras and different widths. Do not use either as the initialization for this trainer.

No weights are stored in this repository. The first download also fetches the backbone named in the policy config. See `NOTICE` for the license statements recorded for those Hub revisions.

## Installation and Getting Started

Use a new environment. Do not install these pins into the existing LeRobot 0.7.1 environment (`conda` env `lerobot`): that environment is Python 3.12 and LeRobot 0.7.1, and this trainer imports `lerobot.envs.libero`, which that release moved.

`environment.yml` is the upstream resolved set (Python 3.10.20, LeRobot 0.4.4, and the simulator packages listed there). `requirements.txt` is the constraint file (`lerobot[smolvla]>=0.4.4,<0.5`, `hf-libero>=0.1.3,<0.2.0`, `peft>=0.19,<0.20`). Both are unchanged. A course environment was created separately as the conda env `smolvla-flow-rl` (Python 3.10) and installed from `requirements.txt` without editing those pins. That install is not a claim that training runs.

Upstream install, from [README_UPSTREAM.md](./README_UPSTREAM.md):

```bash
conda env create -f environment.yml
conda activate smolvla-flow-rl

# or, into an environment you already have:
# pip install -r requirements.txt
python -c "import libero.libero"          # answer the prompt once
python scripts/download_base_policy.py --out /path/to/base
bash examples/single_task_contact.sh /path/to/base
```

The contact script is the upstream setup check. It needs a CUDA GPU, bf16, a downloaded checkpoint, and LIBERO. That script itself has not been launched. The pilot in [docs/NOMINAL_PPO_PILOT.md](./docs/NOMINAL_PPO_PILOT.md) calls `src/train_flow_rl.py` directly, with the contact example's noise, learning rates, discount, and rollout length, and with the recommended `expert_full_vlm_lora` regime rather than the example's illustrative `full` regime. Training requires fourteen arguments with no defaults (`python src/train_flow_rl.py --help`). Evaluation is `python src/eval_cli.py --help`. For headless rendering the upstream README sets `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`. On this machine, robosuite's default file log at `/tmp/robosuite.log` is owned by another account. The conda environment sets `FILE_LOGGING_LEVEL = None` so that handler is not opened. The workaround is local to the environment and is written up in the baseline report.

There is no project command yet for perturbation generation or the robustness comparison.

## Repository branches

GitHub's default branch is still `main` until a repository admin changes it. `main` is the LeRobot 0.7.1 history. The branch that should become the default is `flow-rl`. Changing the default branch is a GitHub setting (Settings → General → Default branch). It does not require a force-push. Renaming `flow-rl` to `main`, or deleting `main`, would remove or rewrite the existing `main` branch. That is not done in this migration. `archive/lerobot-0.7.1` is the same commit as the current `main`, so the LeRobot history stays reachable either way.

Remotes:

| Remote | URL |
| --- | --- |
| `origin` | `git@github.com:AlexP210/smolvla-ppo-robustness.git` |
| `upstream` | `https://github.com/mturan33/smolvla_flow_rl` |
| `upstream-lerobot` | `https://github.com/huggingface/lerobot` |

## Continuous Integration

[`.github/workflows/ci.yml`](./.github/workflows/ci.yml) runs on pull requests and pushes to `flow-rl` and to `main`, and on manual runs. A newer commit cancels the older run for that branch or pull request. The job installs no packages. It runs the two upstream gates that need only the Python 3.10 standard library:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/check_syntax.py
PYTHONDONTWRITEBYTECODE=1 python scripts/check_import_closure.py
```

`scripts/check_syntax.py` parses every text file and fails on a syntax error, a carriage return, or a compiled artefact. Do not run `compileall` as a substitute: it writes bytecode, and this gate then fails. Ruff is not run. The PPO sources are upstream code and are not reformatted to satisfy a linter.

### Checks that are not in CI

These need dependencies or hardware this job does not install. None of them has been accepted as a project result.

| Check | Why it is outside CI |
| --- | --- |
| `python tests/test_invariants.py` | Imports `train_flow_rl`, which imports LeRobot (`lerobot.policies.smolvla`, `lerobot.envs.libero`) and the LIBERO package. Also needs PyTorch, NumPy, and `peft` for the surface-gate tests. CPU is enough for the arithmetic tests; the import is not. |
| `bash scripts/run_gates.sh` | Runs the two commands above, then the invariant file. If `~/.libero/config.yaml` is missing it skips the invariants and exits non-zero. The first `import libero.libero` is interactive. |
| `bash examples/single_task_contact.sh` | GPU, bf16, a downloaded `HuggingFaceVLA/smolvla_libero` checkpoint, and LIBERO. The script itself has not been launched. A separate 20-update call to the trainer has. |
| `python src/train_flow_rl.py`, `python src/eval_cli.py` | Same simulator, checkpoint, and (for training) bf16 GPU requirements. The trainer has been run for 20 updates. `eval_cli.py` has not been the entry point; `src/evaluate.py` has. |

## Development Roadmap

- [x] Adopt `smolvla_flow_rl` as the training history (`flow-rl`, tag `flow-rl-baseline`)
- [x] Pretrained inference on one LIBERO episode, and a ten-task screen that is not a success rate
- [x] One PPO update on one LIBERO task, with peak memory recorded
- [x] A 20-update nominal pilot that collects sparse reward and does not establish learning
- [ ] A matched evaluation large enough to say whether PPO changed success
- [ ] Perturbation framework (seen and unseen)
- [ ] Robustness evaluation (nominal, seen, unseen)
- [ ] Nominal PPO post-training long enough to test a change in success
- [ ] Perturbation-aware PPO post-training
- [ ] Generalization and failure analysis

## Repository Structure

| Path | Role |
| --- | --- |
| `src/flow_sde.py` | Stochastic flow sampling and log-probability. Derivative of RLinf and LeRobot. |
| `src/ppo_step.py`, `src/reference_ops.py`, `src/gae.py` | PPO update, reference loss operators, GAE |
| `src/value_head.py` | Critic, outside the action head. Derivative of RLinf. |
| `src/action_model.py` | Wrapper around LeRobot's SmolVLA policy. Derivative of RLinf and LeRobot. |
| `src/rollout.py`, `src/env_libero.py` | Rollout buffer and LIBERO process harness |
| `src/train_flow_rl.py`, `src/eval_cli.py`, `src/evaluate.py` | Training and deterministic evaluation entry points |
| `src/model_setup.py`, `src/lora_utils.py` | Trainable-surface regimes and LoRA |
| `src/horizon_gate.py`, `src/completeness_gate.py` | Single-horizon rule and evaluation completeness |
| `tests/test_invariants.py` | Upstream invariant tests. Not all of them run without LeRobot and LIBERO. |
| `scripts/check_syntax.py`, `scripts/check_import_closure.py` | Gates the active CI runs |
| `requirements.txt`, `environment.yml` | Upstream dependency constraints, unchanged |
| `protocol/VERIFICATION_PROTOCOL.md` | Upstream verification rules |
| `NOTICE`, `LICENSE`, `THIRD_PARTY_LICENSES/`, `CHANGES_FROM_UPSTREAM.md` | Apache-2.0, attribution, and the record of what was adapted |
| `README_UPSTREAM.md` | Preserved upstream README |
| `docs/PROJECT_PLAN.md` | Technical plan for this course project |
| `docs/BASELINE_REPRODUCTION.md` | Invariants, one deterministic episode, and one PPO update |
| `docs/NOMINAL_PPO_PILOT.md` | Twenty-update nominal pilot, diagnostics, and the matched checkpoint comparison |
| `docs/PRETRAINED_BASELINE_VALIDATION.md` | Native LeRobot evaluation of the pretrained checkpoint, and the horizon comparison |
| `scripts/pretrained_baseline_validation.py` | Measurement harness for that validation |
| `scripts/baseline_reproduction.py` | Measurement harness for the baseline report |
| `scripts/matched_checkpoint_eval.py` | Same evaluator for the pretrained initialization and a saved checkpoint |
| `.github/workflows/ci.yml` | Active project CI |

Planned project components, which are not in the tree yet: a perturbation framework, wiring those perturbations into the rollout, and the robustness evaluation that compares nominal, seen, and unseen conditions.

## Acknowledgments and License

The training code is `smolvla_flow_rl` by Mehmet Turan Yardimci, Apache License 2.0: [github.com/mturan33/smolvla_flow_rl](https://github.com/mturan33/smolvla_flow_rl). Cite it and the works it cites; see [CITATION.cff](./CITATION.cff). Upstream documentation, copyright lines, and license text are preserved. See [README_UPSTREAM.md](./README_UPSTREAM.md), [LICENSE](./LICENSE), and [NOTICE](./NOTICE).

- **RLinf.** The sampler, critic, PPO step, and action-model wrapper derive from RLinf commit `832db3f5c7aa6c61d72f20aaf43b7ce0f4ad2b03`. Apache-2.0. <https://github.com/RLinf/RLinf>
- **LeRobot.** Dependency and, for `src/flow_sde.py` and `src/action_model.py`, an upstream at version 0.4.4. Apache-2.0. <https://github.com/huggingface/lerobot>
- **SmolVLA.** Shukor et al., "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics," arXiv:2506.01844, 2025.
- **LIBERO.** Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning," arXiv:2306.03310, 2023. No LIBERO source is vendored. The MIT notices for LIBERO and robosuite are in `THIRD_PARTY_LICENSES/`.
- **PPO.** Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017. The implementation in this tree follows the RLinf operators named in `src/reference_ops.py`. This project has checked that one update reproduces its stored likelihood and that a short run can apply further updates. It has not shown that those updates improve LIBERO success.

Copyright for the upstream files remains with their authors, as stated in those files and in `NOTICE`. This repository is distributed under the Apache License, Version 2.0. See [LICENSE](./LICENSE).
