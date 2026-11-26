# Project plan

`origin/flow-rl` is the development branch. This plan was written against `3c357d62`. The baseline reproduction, the pretrained baseline validation, the PPO diagnostics, and the evaluation seeds are merged. The nominal reproduction and its ten-episode paired comparison are committed on `feature/a-nominal-rl` and are not yet merged. The moderate-task training config and the three one-change ablations on that branch were not launched. This file is the contract for the next parallel work. It does not implement that work.

Completed measurements, which this plan does not reopen:

- [docs/BASELINE_REPRODUCTION.md](./BASELINE_REPRODUCTION.md): one deterministic LIBERO-Long episode and one PPO update with ratio 1.0.
- [docs/NOMINAL_PPO_PILOT.md](./NOMINAL_PPO_PILOT.md): 20 upstream updates, three sparse successes, and a matched 2/8 versus 0/8 comparison that does not establish learning.
- [docs/PRETRAINED_BASELINE_VALIDATION.md](./PRETRAINED_BASELINE_VALIDATION.md): native LeRobot 0.4.4 scored 53/100 and 46/100 on LIBERO-Long at one and ten executed actions. Float32 Flow RL integration matched the native sampler on a shared observation and a shared noise tensor. The published leaderboard Long entry is 0.60. The exact protocol behind that entry is not fully established.

The trainer default remains one executed action out of a 50-action chunk. The checkpoint remains `HuggingFaceVLA/smolvla_libero`, revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.

## Frozen surface

These files stay as they are on `3c357d62` unless a later task names a concrete bug:

| File | What stays |
| --- | --- |
| `src/flow_sde.py` | Euler step, one stochastic training step, deterministic eval |
| `src/reference_ops.py` | Clipped surrogate, dual clip, clipped Huber value loss |
| `src/gae.py` | GAE recursion and advantage standardization |
| `src/ppo_step.py` | The loss line `pg_loss + value_coef * v_loss + kl_coef * kl_penalty` and `optimizer.step` |
| `src/value_head.py`, `src/action_model.py` | Critic hook and pretrained wrapper |
| `src/horizon_gate.py` | Train and eval horizons must agree, in environment steps |
| `requirements.txt`, `environment.yml` | `lerobot[smolvla]>=0.4.4,<0.5` and `hf-libero>=0.1.3,<0.2` |

Workstream A may append detached statistics to the dictionary `ppo_step_stored` already returns. That addition is not a change to the loss, the generator, or the optimizer. `--action_steps` stays default 1 in `src/train_flow_rl.py` and `src/eval_cli.py`.

`archive/lerobot-0.7.1` and `origin/main` stay at `462e4571`. Do not merge them into `flow-rl`.

## How the three workstreams share the tree

Each workstream branches from current `flow-rl` and opens its own pull request back to `flow-rl`. CI on that base runs `scripts/check_syntax.py` and `scripts/check_import_closure.py`. New tests are CPU scripts under `tests/`. They must not import LIBERO, MuJoCo, or a checkpoint. Do not edit `scripts/run_gates.sh` or `.github/workflows/ci.yml` in a feature branch: every workstream would then conflict on the same two files. A test is run locally with `python tests/test_<name>.py`.

Import closure reads the git index. A new module is invisible to the gate until `git add`. Third-party imports must already be in `requirements.txt`. No new package pin is part of this contract.

Two older branches are duplicates. Their replacements are already on `flow-rl`:

- `feature/ppo-diagnostics` at `dcf05837` is the earlier diagnostics commit. `feature/a-ppo-diagnostics` (`ae6be93e`) is the copy on `flow-rl`.
- `feature/reproducible-eval` at `9b09e993` is the earlier seed commit. `feature/c-reproducible-eval` (`c42639e2`) is the copy on `flow-rl`.

Both older branches also rewrite `README.md` and `docs/NOMINAL_PPO_PILOT.md` against an older text. Do not merge either branch whole.

### Shared schemas

One JSON object, written by the feature that owns it and read by the others. Fields are additive. A reader ignores unknown keys. A writer does not remove a key that is already on `flow-rl`.

**Run record.** `src/run_config.py` already stores the training argv. Workstream A extends that record with the hyperparameter block below. Workstreams B and C do not write it.

```json
{
  "schema": "ppo_run_v1",
  "seed": 0,
  "regime": "expert_full_vlm_lora",
  "action_steps": 1,
  "episode_length": 520,
  "rollout_steps": 32,
  "noise_level": 0.1,
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "clip_range": 0.2,
  "value_coef": 0.5,
  "grad_clip": 1.0,
  "kl_coef": 0.0,
  "actor_lr": 1e-5,
  "critic_lr": 1e-3
}
```

**Perturbation spec.** Owned by Workstream B. Severity 0 is the identity.

```json
{
  "schema": "perturbation_v1",
  "name": "brightness",
  "family": "visual",
  "split": "seen",
  "severity": 2,
  "params": {"delta": 0.2}
}
```

`family` is `visual` or `spatial`. `split` is `seen`, `unseen`, or `nominal`. `nominal` and severity 0 leave the observation and the simulator state unchanged. A spec cannot carry both `seen` and `unseen`.

**Evaluation scenario.** Owned by Workstream C. `action_steps` is recorded and stays 1 unless a later task changes the trainer default.

```json
{
  "schema": "eval_scenario_v1",
  "suite": "libero_10",
  "task_ids": [0],
  "episodes": 10,
  "init_state_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "env_seed": 0,
  "policy_seed": 0,
  "horizon": 520,
  "action_steps": 1,
  "perturbation": null
}
```

`perturbation` is either `null` or a `perturbation_v1` object. The same scenario file is the paired comparison: two checkpoints, one scenario, including the same noise draws.

### Call order that keeps the branches apart

1. Workstream C lands keyword-only `env_seed` and `policy_seed` on `evaluate`, both defaulting to `None`. `None` keeps today's reset and today's global noise draw.
2. Workstream B then lands keyword-only `perturbation=None` on `collect_rollout` and `evaluate`. `None` is the identity.
3. Workstream A edits `ppo_step_stored`'s returned dictionary and the cycle record in `train_flow_rl.py`. It does not take the observation hook.

`README.md` is append-only for these branches. Each workstream adds its own subsection and leaves "What has been measured" unchanged.

## Workstream A: RL post-training

Nominal PPO only. Perturbation-aware training waits until B's hook is on `flow-rl`, and it is not one of the four features below.

### Features

| Branch | Delivers |
| --- | --- |
| `feature/a-ppo-diagnostics` | Cycle log of advantages, rewards, values, returns, explained variance, approximate KL, ratios, clip fraction, losses, and gradient norms |
| `feature/a-reward-advantage` | A CPU report over a stored rollout: reward counts, per-environment sums, advantage moments before and after standardization, and the fraction of the return that is the terminal success |
| `feature/a-nominal-learning` | One documented training run long enough to plot success against update, using the diagnostics from the first branch |
| `feature/a-hparams` | A checked-in config file for that run, and a loader that refuses a file whose `action_steps` is not 1 |

`feature/a-ppo-diagnostics` is on `flow-rl`. The analysis branch can be written against synthetic tensors in parallel. The learning run on `feature/a-nominal-rl` has a filled paired comparison and does not establish that PPO improved success. It is not merged into `flow-rl`. The hyperparameter file is accepted only when a later learning run uses it. The saved ablation configs on that branch were not launched.

### Files

Modify, narrowly:

- `src/ppo_step.py`: add `approx_kl` and `ratio_min` computed from `ratio.detach()` after the surrogate is formed. The loss expression stays on its current line.
- `src/train_flow_rl.py`: snapshot advantages before and after `normalize_advantages`, aggregate every optimizer step, and store the aggregate under a `diagnostics` key beside the existing last-step fields. The first-update ratio check still reads the first step's `ratio_mean`.
- `src/run_config.py`: accept the `ppo_run_v1` block when the hyperparameter feature writes a config. Existing argv dumps stay valid.

New:

- `src/ppo_diagnostics.py`
- `tests/test_ppo_diagnostics.py`
- `tests/test_reward_advantage.py`
- `docs/PPO_DIAGNOSTICS.md`
- `configs/nominal_ppo_libero10.json` on the hyperparameter branch only
- `scripts/report_rollout_advantages.py` on the analysis branch only

Do not modify `src/gae.py`, `src/flow_sde.py`, `src/reference_ops.py`, `src/rollout.py`, or `src/evaluate.py`.

### Acceptance and tests

Diagnostics: a unit test shows approximate KL is 0 when the ratio is 1, explained variance is 1 when predictions equal returns and is null when returns are constant, and the cycle aggregate averages every step rather than the last. The loss tensor's formula is unchanged, and the test does not call the optimizer.

Reward analysis: on a synthetic 0/1 reward rectangle, the report counts successes per environment and shows the advantage mean is 0 after the existing standardization. It does not invent values for the 20-update pilot log. That log has no advantage vectors.

Learning run: the metrics file contains the diagnostics object on every cycle, the first ratio is 1 within the existing tolerance, and the report states success counts without calling them evidence of learning until the curve and the matched eval both say so. No change to `--action_steps`.

Hyperparameters: loading `configs/nominal_ppo_libero10.json` reproduces the argv of the learning run. A file with `action_steps` other than 1 is rejected. Two launches with the same file and seed write the same `ppo_run_v1` record.

### Conflicts

A is the only workstream that edits `src/ppo_step.py` and `src/train_flow_rl.py` during this slice. B and C stay out of those files. The learning run will later call B's hook with `perturbation=None`; that call is a one-line follow-up after B has merged, not part of the first diagnostics commit.

## Workstream B: Perturbation framework

Pixel and initial-state perturbations on the current LIBERO installation. No fork of LIBERO, and no edit of the PPO objective.

### Features

| Branch | Delivers |
| --- | --- |
| `feature/b-perturbation-spec` | `perturbation_v1` load, save, and validation |
| `feature/b-visual` | Brightness, contrast, and Gaussian observation noise |
| `feature/b-spatial` | A bounded shift of the target object's initial position |
| `feature/b-sampling` | Draws of spec parameters from a `SeedSequence` keyed by spec name, severity, episode id, and split |
| `feature/b-hooks` | The identity-preserving call in `collect_rollout` and `evaluate` |

The spec branch lands first. Visual and spatial then proceed in parallel. Sampling lands after the spec schema, and it can land before the image and pose functions. The hook branch is last: it calls the functions the other four branches exported.

### Files

New package `src/perturbations/`:

- `spec.py` for `perturbation_v1`
- `visual.py` for brightness, contrast, and noise
- `spatial.py` for the initial-position shift
- `sample.py` for the seeded draw

Modify:

- `src/rollout.py` and `src/evaluate.py`, in `feature/b-hooks` only, to take `perturbation=None`.
- `src/env_libero.py`, in the spatial branch, to expose one worker-side reset callback. The callback runs after `set_init_state` and before the existing settle steps. The default callback is a no-op, so current resets stay valid.

Visual functions operate on the float CHW image in `[0, 1]` after `LiberoProcessorStep` and before the policy preprocessor resizes to 512. Both cameras go through the same spec. The function returns a new tensor and does not write into the simulator.

### Acceptance and tests

Spec: a severity-0 spec of every family compares equal to the identity. A document that sets `split` to both `seen` and `unseen` is rejected. Round-trip JSON preserves `params`.

Visual: brightness and contrast match a closed-form scale on a 2×2 image. Noise with a fixed seed is bitwise identical across two calls and changes when the episode id changes. Values stay inside `[0, 1]`.

Spatial: a fake object pose moves by the requested delta and by nothing when severity is 0. The test does not import LIBERO. The delta is in meters, clipped to a declared bound, and stored on the spec.

Sampling: the same `(name, severity, episode, split)` returns the same parameter dict. A seen draw and an unseen draw at the same severity differ.

Hooks: with `perturbation=None`, a fake observation dict is unchanged and `collect_rollout`'s stored row is still one decision. With a brightness spec, only the image tensors change. `terminal_success` and `compute_gae` are not called by these tests and are not edited.

### Conflicts

B is the only workstream that adds `src/perturbations/` and the only one that edits `src/rollout.py` in this slice. `src/evaluate.py` is also edited by C. C merges the seed arguments first. B adds `perturbation=None` beside those arguments and does not change the seed defaults.

## Workstream C: Robustness evaluation

Same checkpoints, same initial states, same policy noise. The suite of perturbations comes from B's schema. The paired comparison can be built against the existing `step_000020.pt` before any new training run.

### Features

| Branch | Delivers |
| --- | --- |
| `feature/c-reproducible-eval` | Separate environment and policy seeds. Initial flow noise is standard normal and independent of batch width and row order |
| `feature/c-scenarios` | `eval_scenario_v1` files for LIBERO-Long, 10 episodes, initial states 0–9, `action_steps` 1, horizon 520 |
| `feature/c-robustness-suites` | Nominal, seen, unseen, and a severity ladder, each pointing at one scenario plus one spec |
| `feature/c-curves` | Per-task success and a success-versus-severity table from a finished scenario log |
| `feature/c-paired-compare` | Two checkpoints, one scenario, one noise sequence, per-task counts for each |

Seeds land first and do not wait on A or B. Scenarios wait on the seed arguments. Suites wait on B's spec schema and on scenarios. Curves and the paired comparison wait on a scenario log. The paired script can target the pretrained initialization and `runs/nominal_ppo/libero10_task0/step_000020.pt` once the scenario format exists. That checkpoint is gitignored. The script records its path and does not copy it into git.

### Files

Modify:

- `src/evaluate.py` and `src/eval_cli.py`, seed arguments only, on `feature/c-reproducible-eval`.
- `scripts/matched_checkpoint_eval.py`, so the paired comparison calls `evaluate` with the scenario's seeds and episode ids.

New:

- `src/eval_noise.py`
- `src/eval_scenario.py`
- `tests/test_eval_noise.py`
- `tests/test_eval_scenario.py`
- `tests/test_robustness_curves.py`
- `configs/scenarios/libero10_nominal.json` and the seen, unseen, and severity files
- `scripts/robustness_curves.py`
- `docs/ROBUSTNESS_EVAL.md`

Do not modify `src/ppo_step.py`, `src/flow_sde.py`, or the default of `--action_steps`.

### Acceptance and tests

Seeds: the same episode id produces the same reset seed and the same float32 noise row when the batch is reordered or resized. A different decision index changes the noise. The Gaussian mean and standard deviation stay 0 and 1. `evaluate` without the new arguments keeps the historical reset. The Euler step is untouched. A test may call `sde_sample` with a constant velocity. It does not claim that a batched bfloat16 network forward is bitwise identical across batch sizes.

Scenarios: a file with `action_steps` other than 1, a horizon other than 520 for `libero_10`, or an episode count below 10 is rejected by the loader used for reported runs. The nominal file has `perturbation: null`.

Suites: seen and unseen files differ in `split` and in at least one parameter. The severity ladder is the same family at severities 0, 1, 2, and 3. Severity 0 matches the nominal spec.

Curves: a synthetic per-task log becomes a table of successes over attempts at each severity. The script does not fill missing severities with a guessed rate.

Paired comparison: both checkpoints receive the same scenario object. The report has one row per task for each checkpoint and states the episode count. Eight episodes are not a reported comparison under this contract. Ten per task is the minimum.

### Conflicts

C owns `src/eval_noise.py`, `src/eval_scenario.py`, and `configs/scenarios/`. It touches `src/evaluate.py` before B. It does not touch `src/train_flow_rl.py` or `src/ppo_step.py`. `scripts/matched_checkpoint_eval.py` is C's file for this slice. A and B leave it alone.

## LIBERO-Plus

[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) (arXiv:2510.13626) defines seven perturbation dimensions: object layout, camera viewpoint, robot initial state, language, light, background texture, and sensor noise, with five severity levels. The current LeRobot documentation installs it by cloning that repository and uninstalling `hf-libero`, so `import libero` resolves to the fork. The same page evaluates it with `--env.type=libero_plus` and `lerobot-eval`.

This environment is LeRobot 0.4.4 and `hf-libero` 0.1.5. The trainer imports `lerobot.envs.libero._make_env_fns` and `LiberoEnv` from that pin. Replacing `hf-libero` would remove the simulator the baseline runs used. `libero_plus` is not an environment type in the installed 0.4.4 package, and the fork is not a `requirements.txt` extra.

The reuse that stays compatible is the vocabulary. Spec names may be `light`, `sensor_noise`, and `layout`, mapped onto brightness and contrast, observation noise, and the initial object shift this plan actually implements. Camera, robot initial state, language rewrites, and background textures need the fork's BDDL and assets. They are out of this slice.

A separate conda environment may install the fork later for a comparison study. That environment is not `smolvla-flow-rl`, and its results are not numbers from this trainer until a task says they are.

## Dependency order

```text
flow-rl (3c357d62)
├── A  diagnostics ──► reward analysis (synthetic, parallel)
│                  └─► nominal learning run ──► hyperparameter file
├── B  spec ──► visual ──┐
│         └──► spatial ──┼──► sampling (after spec) ──► hooks
└── C  seeds ──► scenarios ──► suites (after B spec) ──► curves
                            └─► paired comparison
```

A's learning run, B's hooks, and C's suites are the three merges that must be sequenced. Everything to their left can be reviewed in parallel.

Perturbation-aware PPO is a later branch. It depends on A's diagnostics being on `flow-rl` and on B's hook accepting a seen spec during `collect_rollout` while evaluation still applies seen and unseen. It is not started in this slice. It still does not change `src/gae.py` or the PPO loss.
