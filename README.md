# Robustness of Vision-Language-Action Models through Reinforcement Learning Post-Training

This repository is a course research project on reinforcement-learning post-training for vision-language-action (VLA) policies. It builds on [LeRobot](https://github.com/huggingface/lerobot) (Hugging Face). The upstream project documentation is preserved, unmodified, in [README_LEROBOT.md](./README_LEROBOT.md). The commit tagged `original-baseline` is the LeRobot revision this project started from, before these documentation changes.

No robustness experiments have been run in this project yet. Sections below separate functionality that already exists in LeRobot from work that is planned.

## Project Title

Robustness of Vision-Language-Action Models through Reinforcement Learning Post-Training

## Overview

Supervised VLA policies imitate demonstration data. A policy that succeeds on the nominal training distribution can still fail when appearance, object placement, or physical conditions shift. Task success on the original distribution does not, by itself, measure that sensitivity.

This project studies reinforcement-learning post-training as a way to improve both nominal task performance and robustness under those shifts. The intended testbed is the pretrained [SmolVLA](https://arxiv.org/abs/2506.01844) policy in simulation on [LIBERO](https://arxiv.org/abs/2306.03310). Proximal Policy Optimization (PPO) is the planned post-training algorithm. That study has not been implemented in this repository yet.

## Research Questions

- Does nominal RL post-training improve robustness beyond nominal task success?
- Does exposure to perturbations during RL training improve robustness?
- Do robustness improvements generalize to unseen perturbations?
- What are the trade-offs between nominal performance, robustness, and learning efficiency?

## Methodology

The planned comparison uses three policy configurations on LIBERO:

- **Pretrained supervised SmolVLA without RL.** The SmolVLA policy already in LeRobot, loaded from a supervised checkpoint and evaluated with no reinforcement-learning update.
- **Nominal PPO post-training.** The same SmolVLA initialization, then PPO on unperturbed LIBERO tasks. Planned; there is no PPO trainer in this repository.
- **Perturbation-aware PPO post-training.** The same initialization, then PPO while training episodes include the perturbations described below. Planned; there is no perturbation-aware trainer in this repository.

SmolVLA is implemented under `src/lerobot/policies/smolvla`, with upstream instructions in [docs/source/smolvla.mdx](./docs/source/smolvla.mdx). LIBERO is integrated under `src/lerobot/sim/integrations/libero.py`, with upstream instructions in [docs/source/libero.mdx](./docs/source/libero.mdx).

LeRobot's existing reinforcement-learning code is a separate stack. `src/lerobot/rl` implements Soft Actor-Critic for the HIL-SERL workflow ([docs/source/hilserl.mdx](./docs/source/hilserl.mdx)). TD-MPC is a separate policy ([docs/source/policy_tdmpc_README.md](./docs/source/policy_tdmpc_README.md)). Neither is the PPO post-training method planned here.

## Environment Perturbations

Perturbations are planned and are not implemented. The intended families are:

- **Visual.** Changes to appearance, such as color, texture, or lighting. Appearance changes are a first priority.
- **Spatial.** Changes to the initial object position. Initial-position changes are a first priority.
- **Physical.** Changes to dynamics or object properties. These are in scope after the appearance and initial-position perturbations.

Each family is meant to be split later into perturbations used during training ("seen") and perturbations held out for testing ("unseen"). The specific parameter ranges are not fixed yet.

## Evaluation

Evaluation of the three configurations is planned. The intended protocol is:

- **Nominal.** Success on unperturbed LIBERO tasks.
- **Seen perturbations.** Success under perturbations that perturbation-aware training was allowed to use.
- **Unseen perturbations.** Success under held-out perturbations.

Reported quantities, once experiments exist, will be task success rate, robustness degradation relative to the nominal evaluation, learning curves, and training efficiency. This README does not report any of those numbers.

LeRobot already evaluates a policy on LIBERO with `lerobot-eval` and reports per-task success rates, including a 95% Wilson interval (`pc_success_ci95`). That entry point is documented in [docs/source/libero.mdx](./docs/source/libero.mdx). It does not apply the perturbation protocol above.

## Installation and Getting Started

General installation, hardware setup, and dataset tools are documented in the preserved upstream guide, [README_LEROBOT.md](./README_LEROBOT.md), and in the [LeRobot installation documentation](https://huggingface.co/docs/lerobot/installation). This tree requires Python >= 3.12 (`pyproject.toml`).

Project-relevant extras that this repository already defines:

```bash
pip install -e ".[smolvla]"   # docs/source/smolvla.mdx
pip install -e ".[libero]"    # docs/source/libero.mdx; Linux only
```

For headless LIBERO rendering, the LIBERO guide sets `MUJOCO_GL=egl` before training or evaluation. Supervised SmolVLA fine-tuning uses `lerobot-train`. The SmolVLA and LIBERO guides above give the working commands, including a SmolVLA training example on the `lerobot/libero` dataset. `lerobot-train --help` lists the trainer options. Nominal LIBERO evaluation uses `lerobot-eval`; the suite names and flags are in [docs/source/libero.mdx](./docs/source/libero.mdx).

There is no project command yet for PPO post-training, perturbation generation, or the robustness comparison.

## Development Roadmap

- [ ] Baseline setup and feasibility study
- [ ] Nominal policy evaluation
- [ ] Perturbation framework
- [ ] Robustness evaluation
- [ ] Nominal PPO post-training
- [ ] Perturbation-aware PPO post-training
- [ ] Generalization and failure analysis

## Repository Structure

Existing LeRobot directories used by this project:

| Path | Role in this project |
| --- | --- |
| `src/lerobot/policies/smolvla` | SmolVLA policy implementation |
| `src/lerobot/sim` | Simulator integrations, including LIBERO (`integrations/libero.py`) |
| `docs/source/smolvla.mdx`, `docs/source/libero.mdx` | Upstream SmolVLA and LIBERO instructions |
| `src/lerobot/rl` | Upstream HIL-SERL actor-learner stack (SAC), not the planned PPO experiments |
| `docs/`, `tests/`, `examples/`, `docker/`, `scripts/`, `benchmarks/` | Upstream documentation, tests, examples, and tooling |
| `LICENSE` | Apache License 2.0, unchanged |
| `README_LEROBOT.md` | Preserved upstream README |

`README_LEROBOT.md` remains at the repository root, so its relative links to `./docs/` and `./media/` still resolve. `setup.py` and `pyproject.toml` (`dynamic = ["readme"]`) read the root `README.md` path. That path is this project README. `scripts/ci/benchmark_matrix.py` also hashes `README.md` when fingerprinting simulator images.

Planned project components, which are not in the tree yet: a perturbation framework, PPO post-training for the nominal and perturbation-aware configurations, and the robustness evaluation that compares nominal, seen, and unseen conditions.

## Acknowledgments and License

This project builds upon LeRobot by Hugging Face: [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot). Upstream documentation, copyright notices, and the license text are preserved. See [README_LEROBOT.md](./README_LEROBOT.md) and [LICENSE](./LICENSE).

- **LeRobot.** Cadène et al., Hugging Face. <https://github.com/huggingface/lerobot>
- **SmolVLA.** Shukor et al., "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics," arXiv:2506.01844, 2025. Citation in [src/lerobot/policies/smolvla/README.md](./src/lerobot/policies/smolvla/README.md).
- **LIBERO.** Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning," arXiv:2306.03310, 2023. <https://github.com/Lifelong-Robot-Learning/LIBERO>
- **PPO.** Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017. This is the planned post-training algorithm. It is not implemented in this repository.
- **Existing RL code.** LeRobot's HIL-SERL implementation (SAC) in `src/lerobot/rl`, following Luo et al., and the TD-MPC policy. These are upstream algorithms already in the tree. They are not the PPO experiments planned above.

Copyright 2024 The Hugging Face team. All rights reserved. The original work and this repository are distributed under the Apache License, Version 2.0. See [LICENSE](./LICENSE).
