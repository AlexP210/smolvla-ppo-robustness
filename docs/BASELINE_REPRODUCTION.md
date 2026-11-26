# Baseline reproduction and feasibility

Branch `feature/baseline-reproduction`, created from `flow-rl` at `ee1f5a83` (the migration commit whose parent is `flow-rl-baseline` `c161af21`). No file under `src/` was changed. The PPO update, sampler, reward, and horizon gate are the upstream implementations.

These runs show that one deterministic episode can be executed and that one PPO update can be applied. They do not show that PPO improves task success. A single success, and a ratio of one before any learning has had time to matter, are checks that the machinery runs.

## Environment

Conda env `smolvla-flow-rl`, Python 3.10.22, `PYTHONNOUSERSITE=1`. Installed from the unchanged `requirements.txt`. Resolved pins used here: `lerobot==0.4.4`, `hf-libero==0.1.5`, `peft==0.19.1`, `torch==2.10.0+cu128`.

The existing LeRobot 0.7.1 env was not used.

### Robosuite log file

This workaround is local to the `smolvla-flow-rl` conda environment. It is not in the repository, and recreating the environment from `requirements.txt` restores the stock setting.

`/tmp/robosuite.log` is owned by another account (`nabail`, mode `664`, last written 2026-11-16). The sticky bit on `/tmp` prevents this account from replacing it. The file was not modified. sudo was not used, and no other user's files were changed.

`robosuite.utils.log_utils.DefaultLogger` opens that path only when its file level is not `None`:

```python
if file_logging_level is not None:
    fh = logging.FileHandler("/tmp/robosuite.log")
```

The level comes from `robosuite.macros.FILE_LOGGING_LEVEL`. The installed default is `"DEBUG"`. The comment in `macros.py` says that setting a logging level to `None` turns that logger off. The file handler was disabled by this assignment, in this environment only:

```python
FILE_LOGGING_LEVEL = None  # user config: do not open /tmp/robosuite.log
```

The two files that received it are:

- `/home/Mozhgan/miniconda3/envs/smolvla-flow-rl/lib/python3.10/site-packages/robosuite/macros.py`
- the same path's `macros_private.py`

`log_utils` reads `robosuite.macros`. A private macros file alone does not stop the handler, because the package's star import does not rebind `robosuite.macros.FILE_LOGGING_LEVEL`. Console logging stays at `WARN` and goes to stderr.

Check, without touching the other account's file:

```bash
PYTHONNOUSERSITE=1 python -c "import os, robosuite.macros as m; print(m.FILE_LOGGING_LEVEL); print(os.stat('/tmp/robosuite.log').st_mtime)"
```

`FILE_LOGGING_LEVEL` prints `None`, and the mtime stays at the other account's last write.

### Other environment notes

- `~/.libero/config.yaml` was created with the package's default paths, which is the result of answering "N" to LIBERO's one-time dataset-path question. The question is interactive and cannot be answered under a captured shell.
- The `hf-libero` wheel does not ship simulation assets. The first environment construction downloaded them to `~/.cache/libero/assets`.
- Transformers prints `` `torch_dtype` is deprecated! Use `dtype` instead! `` while loading SmolVLM. The load still completed.

### Invariant suite

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /home/Mozhgan/miniconda3/envs/smolvla-flow-rl/bin/python tests/test_invariants.py
```

Exit code 0. The last line is `INVARIANTS_PASS`. Lines that print `FAIL` inside the import-closure and syntax negative tests are the defects those tests plant on purpose; the corresponding check lines are `PASS`.

The measurement harness writes its episode video through `matplotlib.animation`, whose package is named in `requirements.txt`. `imageio` can write the same file and is pinned in `environment.yml`, but it is not a direct line of `requirements.txt`, so a committed `import imageio` fails `scripts/check_import_closure.py`. `requirements.txt` was not edited. The video already on disk was written before this switch; the writer was checked on a synthetic clip rather than by repeating the episode.

`runs/`, `outputs/`, `checkpoints/`, `*.mp4`, `*.gif`, `*.jsonl`, and the weight suffixes `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, and `*.ckpt` are listed in `.gitignore`. The downloaded policy used here lives outside the tree, at `/user_data/Mozhgan/Projects/RL_Project/checkpoints/HuggingFaceVLA_smolvla_libero`.

## Hardware

Inspected before the runs. Four GPUs, driver 580.65.06:

| GPU | Name | Total | Used at inspection | Compute capability |
| --- | --- | --- | --- | --- |
| 0 | NVIDIA RTX PRO 6000 Blackwell Max-Q | 97887 MiB | 53985 MiB | 12.0 |
| 1 | NVIDIA RTX PRO 6000 Blackwell Max-Q | 97887 MiB | 1289 MiB | 12.0 |
| 2 | NVIDIA RTX PRO 6000 Blackwell Max-Q | 97887 MiB | 38393 MiB | 12.0 |
| 3 | NVIDIA RTX PRO 6000 Blackwell Max-Q | 97887 MiB | 39075 MiB | 12.0 |

`torch 2.10.0+cu128` reports CUDA 12.8, `torch.cuda.is_available() is True`, and `torch.cuda.is_bf16_supported() is True`. A one-element tensor allocated on `cuda:0` of the visible device. Both experiment processes set `CUDA_VISIBLE_DEVICES=1` because GPU 0 was already heavily occupied. Peak figures below are this process's PyTorch allocator, not the whole card.

## Checkpoint

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  python scripts/download_base_policy.py \
  --out /user_data/Mozhgan/Projects/RL_Project/checkpoints/HuggingFaceVLA_smolvla_libero
```

The helper pinned and downloaded:

- `HuggingFaceVLA/smolvla_libero` at `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`
- backbone `HuggingFaceTB/SmolVLM2-500M-Instruct` at `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, into the Hugging Face cache

`SmolVLAForRLActionPrediction` ran `_check_checkpoint_fits_the_harness` and constructed. Declared features read back from the loaded config:

| Field | Value |
| --- | --- |
| Images | `observation.images.image`, `observation.images.image2` |
| State | 8 |
| Action | 7 |
| Chunk size | 50 |
| Denoise steps | 10 |

That matches the harness constants in `src/action_model.py`. `lerobot/smolvla_libero` was not loaded.

## Deterministic inference

One episode of `libero_10` task 0, the suite's own horizon of 520, seed 0, deterministic `flow_ode` sampling (`evaluate` sets `noise_method` to `flow_ode`). No LoRA and no value head were attached, so the action weights are the downloaded checkpoint. `n_action_steps` is 1, matching the checkpoint config.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
python scripts/baseline_reproduction.py infer \
  --base-policy /user_data/Mozhgan/Projects/RL_Project/checkpoints/HuggingFaceVLA_smolvla_libero \
  --output-dir runs/baseline/infer \
  --horizon 520 --denoise-steps 10 --seed 0
```

Exit code 0. Recorded in `runs/baseline/infer/summary.json` and `runs/baseline/infer/events.jsonl`:

| Quantity | Value |
| --- | --- |
| Success | yes, on environment step 369 |
| Episodes | 1 |
| Horizon cap | 520 |
| Model load | 10.28 s |
| Rollout wall clock | 123.10 s |
| Policy calls | 369 |
| Inference time, sum / mean | 111.05 s / 0.301 s |
| Peak GPU allocated / reserved during load | 1161.5 MiB / 1178 MiB |
| Peak GPU allocated / reserved during the episode | 1201.4 MiB / 1242 MiB |

The episode stopped at the success step. The loop does not continue to 520 after `finished` is set. The video is `runs/baseline/infer/episode.mp4`: 369 frames of the agentview camera, written at 20 fps (18.45 s of video, 192 KiB). Playback rate is not the simulator control rate.

This is one episode. A success rate of 1.0 in the summary is that one success divided by one attempt. It is not an estimate of LIBERO-10 performance.

## Action-chunk execution

The policy predicts a chunk of 50 actions. The episode still made one policy call per simulator step. That is the upstream execution horizon, not a shortcut added by the measurement harness. Action execution was not changed.

Each call to `SmolVLAForRLActionPrediction.sample_actions` draws one chunk. It builds the prefix cache and calls `sde_sample`, which integrates all `num_steps` (10) flow steps. In evaluation, `evaluate` sets `noise_method` to `flow_ode` before the first call, so every integration step is deterministic. In training, `sde_sample` keeps one randomly chosen integration step stochastic (`flow_sde`) and takes the rest as `flow_ode`. A policy call is a new flow sampling operation either way. It does not replay a previously sampled chunk.

How many of those 50 actions are executed is `--action_steps` / `n_action_steps`, default 1:

| Call site | Default |
| --- | --- |
| `evaluate` | `n_action_steps=1` |
| `collect_rollout` | `n_action_steps=1` |
| `ppo_step_stored` | `n_action_steps=1` |
| `train_flow_rl.py` and `eval_cli.py` | `--action_steps`, default 1 |
| `examples/single_task_contact.sh` | `--action_steps 1` |
| `scripts/baseline_reproduction.py` | passes `1` to both `evaluate` and `collect_rollout` |

`exec_action` selects `chunk[:, step, :]`. With `n_action_steps` of 1, only position 0 is sent to the simulator. Positions 1 through 49 are computed and discarded. The checkpoint's own `n_action_steps` field is also 1, and its `chunk_size` is 50. The 369 calls are 369 decisions of one environment step each. The harness and the trainer agree.

Training and evaluation share that length when both are launched by `train_flow_rl.py`: the same `args.action_steps` is passed to `collect_rollout` and to `evaluate`. The horizon gate checks episode length against the evaluation horizon, both counted in environment steps. It does not check `action_steps`. The trainer comment says an evaluation that executed a different number of actions per plan would score a policy the run did not train.

What the executed length changes, at the upstream default of 1:

- Likelihood. `sde_sample` stores the log-probability of the one stochastic integration step, cropped to the full predicted chunk and the environment action width, shape `[batch, 50, 7]`. `ppo_step_stored` then sums only `[:, :n_action_steps, :action_env_dim]`. At 1, the ratio grades the first action's seven dimensions. The other 49 positions do not enter the ratio.
- Reward. `collect_rollout` steps the simulator `n_action_steps` times and keeps the maximum sparse success in that window. One stored reward per decision. At 1, the reward is the success flag of that single step.
- GAE. Advantages are computed on the stored rectangle of decisions, not on raw simulator steps. `gamma` and `gae_lambda` therefore discount per executed decision. Termination inside the window sets `done` for that decision, and the bootstrap does not cross it. A longer `action_steps` would make one decision cover more simulator time while the discount stayed per decision.
- Termination and storage. Any termination or truncation inside the window marks the decision done. Chains, denoise indices, and log-probabilities are stored once per decision. The recorded success step in `evaluate` is still an environment-step index (`t * steps_per_decision + k + 1`).

The nominal pilot keeps `--action_steps 1`. Changing it would change the likelihood surface, the reward timing, and the meaning of the discount, which this phase does not do.

## One PPO update

Regime `expert_full_vlm_lora` (full-rank action expert, LoRA on the language backbone, full-rank action head, vision tower and connector frozen, critic trained). Noise level 0.1. Two stored decisions, one environment, episode cap 4, which is below the suite limit of 520. Seed 0. Learning rates `1e-5` (actor) and `1e-3` (critic) are the illustrative values from `examples/single_task_contact.sh`, not a tuned recipe. One call to `ppo_step_stored` on the first stored decision. No further cycles.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
python scripts/baseline_reproduction.py ppo \
  --base-policy /user_data/Mozhgan/Projects/RL_Project/checkpoints/HuggingFaceVLA_smolvla_libero \
  --output-dir runs/baseline/ppo \
  --episode-length 4 --rollout-steps 2 --denoise-steps 10 --noise-level 0.1 \
  --actor-lr 1e-5 --critic-lr 1e-3 --gamma 0.99 --gae-lambda 0.95 --seed 0
```

Exit code 0. The manifest printed before the rollout:

```text
trainable=99.40M of 606.88M (16.379%): action_expert=96.70M action_head=0.76M
backbone_adapters=1.64M critic=0.31M other=0.00M
action expert trainable fraction=1.0000
optimizer group0 lr=1e-05 n=555, group1 lr=0.001 n=5
```

| Check | Result |
| --- | --- |
| Stored vs recomputed log-probability | both means `19.009685516357422` |
| PPO ratio before this update | `ratio_mean = 1.0` |
| Update applied | `step_skipped = false` |
| Pre-clip gradient norm | `848.11` (default clip in `ppo_step_stored` is 1.0) |
| Load / rollout / update wall clock | 8.77 s / 1.64 s / 0.45 s |
| Peak GPU allocated / reserved after the load, through the update | 2137.6 MiB / 2236 MiB |
| Mean sparse reward on the 2 decisions | 0.0 |
| Frozen parameter sum, 507,482,304 values | unchanged (`-97669.1801738806` before and after) |
| Frozen parameters with a nonzero gradient | none |
| Trainable parameter sum | changed (`31764.13277114454` to `31764.06574331307`) |

Nonzero gradients covered every action-expert parameter (96.70M), every action-head parameter (0.76M), and every critic parameter (0.31M). Of the 1.64M language-adapter parameters, 0.64M had a nonzero gradient on this first step and the rest were zero. That was not treated as a bug: one LoRA factor is initialized at zero, so its pair has a zero gradient until that factor moves, and this run did not retune or edit the adapter code.

`pg_loss` was `-1.0` and `v_loss` was `0.002`. With a ratio of one, the policy term is the negative of the normalized advantage. Those scalars are the loss on two steps of an episode that was capped at 4, not a learning curve.

## What this does not show

- It does not show that the pretrained policy succeeds at any particular rate. The inference result is one episode of one task.
- It does not show that PPO improves success, sample efficiency, or robustness. The reward on the training segment was zero because the segment was two steps long, and there was no before/after evaluation.
- It does not measure the 12 GB laptop configuration in the upstream README. This card has about 96 GB, and the process peaks above were about 1.2 GiB for deterministic inference and about 2.1 GiB for one update. Peaks on a busier GPU, with a longer rollout, or with gradient accumulation were not measured.

## Not done

No multi-cycle training, no full-horizon PPO rollout, no perturbation code, and no edit to `src/flow_sde.py`, `src/ppo_step.py`, `src/gae.py`, or `src/rollout.py`.
