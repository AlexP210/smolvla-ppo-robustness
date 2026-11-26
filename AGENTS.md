This file provides guidance to AI agents when working with code in this repository.

> Project README: [`README.md`](./README.md). Preserved upstream README: [`README_LEROBOT.md`](./README_LEROBOT.md). Upstream hardware and training guide: [`AGENT_GUIDE.md`](./AGENT_GUIDE.md).

## Project Overview

This repository is a course research project on the robustness of vision-language-action policies under reinforcement-learning post-training. The planned comparison has three configurations on LIBERO: pretrained supervised SmolVLA without RL, nominal PPO post-training, and perturbation-aware PPO post-training. Planned perturbations prioritize appearance and initial object position, then physical changes. The questions are whether nominal RL improves robustness, whether training on perturbations helps, whether that improvement holds for unseen perturbations, and what that costs in nominal success and training efficiency. PPO post-training and the perturbation framework are not implemented yet.

The tree is LeRobot. SmolVLA is `src/lerobot/policies/smolvla`. LIBERO is `src/lerobot/sim/integrations/libero.py`. `src/lerobot/rl` is the upstream HIL-SERL (SAC) stack, not the planned PPO experiments.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (upstream package management)

## Active development workflow

GitHub Actions runs only [`.github/workflows/ci.yml`](./.github/workflows/ci.yml), on pull requests and pushes to `main` and on manual runs. Inherited LeRobot workflows are archived in [`.github/upstream-workflows/`](./.github/upstream-workflows/) and do not run. Do not add GPU jobs, simulator installs, checkpoint downloads, Docker publishing, or repository secrets to the active workflow.

Project CPU tests go under `tests/robustness/{perturbations,rollouts,config,evaluation}`. From the repository root, the CI commands are:

```bash
python -m pip install "ruff==0.14.1" "pytest>=8.1.0,<10.0.0"
python -m compileall -q -f src tests/robustness
ruff check src tests/robustness
ruff format --check src tests/robustness
python -m pytest tests/robustness --confcutdir=tests/robustness -q
```

Ruff `0.14.1` is the version in `.pre-commit-config.yaml` and reads `[tool.ruff]` in `pyproject.toml`. `--confcutdir` avoids the root `conftest.py` and `tests/conftest.py`, which import LeRobot.

## Upstream commands

Use these for LeRobot package work. They are not the project CI.

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
uv run pytest tests -svv --maxfail=10       # Upstream suite; requires a LeRobot install
DEVICE=cuda make test-end-to-end            # Upstream E2E tests
pre-commit run --all-files                  # Upstream hooks (ruff, typos, bandit, mypy, ...)
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` options in `configs.py`; `make_env()` / `make_task_env()` in `factory.py` build the envs. A config's `sim_settings()` maps its options to its `lerobot.sim.integrations.<type>` module's `env_fns()` kwargs.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Upstream pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`. Project CPU tests live in `tests/robustness/` and are the only tests the active CI runs.
- **`.github/workflows/`** — Active CI is `ci.yml` only (syntax, Ruff, and `tests/robustness` on Python 3.12).
- **`.github/upstream-workflows/`** — Original LeRobot workflows, preserved and inactive: `quality.yml` (pre-commit), `fast_tests.yml` (one tier per extra), `full_tests.yml` (all extras, E2E, and `-m gpu`), `latest_deps_tests.yml`, `security.yml`, `release.yml`, plus Docker, docs, benchmark, and labeling workflows.
- **`docs/source/`** — Upstream HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (upstream E2E targets), `uv.lock`, `CONTRIBUTING.md`, `README.md` (this project), `README_LEROBOT.md` (preserved upstream README).

## Notes

- **Mypy**: one ruleset (`[tool.mypy]` in `pyproject.toml`) checks the paths in its `files` setting, except `lerobot.rl`, the vendored `molmoact2_hf_model` and the generated `*_pb2` modules. Run it with `pre-commit run mypy --all-files`. Add type annotations when modifying code.
- **Imports**: prefer top-level imports; relative (`from .sibling import X`) across sibling files within a module, absolute (`from lerobot.module import X`) across modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`, see `pyproject.toml`). Guard optional imports with `TYPE_CHECKING or _foo_available` at module top + a `require_package(...)` check at use time. Reuse the `_foo_available` flags in `utils/import_utils.py`; don't call `is_package_available`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Commands**: project CI uses the `python -m` commands in "Active development workflow". For LeRobot package work, prefer `uv run` over raw `python` or `pip`.
- **Test marks**: the archived Fast Tests workflow runs one tier per extra (`-m base|dataset|hardware|viz`) and its GPU job runs `-m gpu`. A test that needs the `dataset`, `hardware` or `viz` extra must be marked (`pytestmark = pytest.mark.dataset`), otherwise it only runs in `base` and skips there. A test that uses CUDA outside `tests/policies` and without `@require_cuda` needs `@pytest.mark.gpu`. Details in CONTRIBUTING.md, "Running Tests". Project tests under `tests/robustness/` stay free of those marks and of GPU or simulator imports.
