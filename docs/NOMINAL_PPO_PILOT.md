# Nominal PPO pilot

One short run of the upstream trainer, `src/train_flow_rl.py`, on a single LIBERO-10 task. No perturbation was added. The PPO update, sampler, reward, and horizon gate were not modified. Action execution stays at one simulator step per policy call, for the reason in `docs/BASELINE_REPRODUCTION.md`.

The run completed and the sparse reward fired. It does not show that PPO improves task success. The deterministic evaluation after these 20 updates succeeded on none of its eight episodes.

## Action execution, left as it is

The pretrained policy emits a chunk of 50 actions. The trainer's default, the contact example, and this pilot all execute one of them (`--action_steps 1`). Each policy call is a new 10-step flow sample. The other 49 actions are discarded. Training and evaluation in this run both use that length, so the likelihood sum, the stored reward, and GAE all see one simulator step per decision. `gamma` 0.99 is therefore a per-step discount. Changing the executed length would change that surface, and this pilot does not.

## Pretrained screen

Before choosing the task, the downloaded `HuggingFaceVLA/smolvla_libero` checkpoint was evaluated deterministically on all ten `libero_10` tasks. Two episodes per task, horizon 520, `n_action_steps` 1, run seed 0. `evaluate` offsets evaluation seeds by 7000, so batch 0 uses seeds 7000–7009 and batch 1 uses 7010–7019. The ten tasks ran in one vector environment, so each policy call was a batch of ten. Wall clock 807 s after a 8.0 s load. Peak PyTorch allocated memory 1479 MiB on physical GPU 2 (`CUDA_VISIBLE_DEVICES=2`). Pooled success 0.35 (7 of 20).

| Task | Instruction | Result | Success step |
| --- | --- | --- | --- |
| 0 | put both the alphabet soup and the tomato sauce in the basket | 1/2 | —, 316 |
| 1 | put both the cream cheese box and the butter in the basket | 0/2 | — |
| 2 | turn on the stove and put the moka pot on it | 0/2 | — |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 2/2 | 249, 267 |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 0/2 | — |
| 5 | pick up the book and place it in the back compartment of the caddy | 2/2 | 204, 256 |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 1/2 | —, 408 |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 0/2 | — |
| 8 | put both moka pots on the stove | 1/2 | —, 390 |
| 9 | put the yellow and white mug in the microwave and close it | 0/2 | — |

Task 0 was selected. `examples/single_task_contact.sh` already uses it. In this screen it is solved on one of two seeds, so the pretrained policy is not saturated on the sample, and the success that did occur was at step 316, inside the suite horizon of 520. Tasks 3 and 5 were solved on both seeds. Tasks that failed both seeds can leave a short pilot with no reward at all.

An earlier one-environment episode of task 0, seed 7000, succeeded at step 369 (`docs/BASELINE_REPRODUCTION.md`). The same recorded seed failed in this ten-task batch. The two measurements do not share a batch size. A batched forward is not assumed to reproduce the single-environment episode, and the screen is not a substitute for that episode.

Records: `runs/baseline/screen/summary.json` and `events.jsonl`. The script that produced them is `runs/baseline/screen_pretrained.py`, which is untracked because `runs/` is gitignored.

## Configuration

Hyperparameters were read from `examples/single_task_contact.sh` first. That file labels its values, including `--regime full`, as illustrative rather than a tuned recipe. The episode length is the exception it calls out: 520, the suite's own limit. This pilot keeps the example's noise, learning rates, discount, trace decay, rollout length, denoise steps, and action execution. It uses `--regime expert_full_vlm_lora` because that is the surface the trainer's help text and the upstream README name as the recommended default, and the example says its own regime is not one.

The trainer applies one optimizer step per stored timestep. The batch at that step is the number of environments, not the rollout length. The contact example uses one environment, so its batch is one transition. This pilot passes the same task four times, which is the upstream way to widen that batch without pooling timesteps the update is not written to pool. Four is still a small PPO batch.

Evaluation is only at cycle 20. An earlier evaluation resets the workers, and the pretrained successes on this task were past step 300, so an eval at cycle 8 would have cut the episode before that region. Training seed 0 and the evaluation seeds are different namespaces: collection is seeded from the run seed, and `evaluate` adds 7000 before assigning per-environment seeds. No robustness perturbation is implemented in this trainer. `--noise_level 0.1` is the sampler's exploration scale, not an observation perturbation. The tripwire was left off. The thresholds in the contact example are part of the illustrative command, and an unchosen guard can stop a short run before it has recorded anything.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2
python src/train_flow_rl.py \
  --tasks libero_10:0,libero_10:0,libero_10:0,libero_10:0 \
  --base_policy /user_data/Mozhgan/Projects/RL_Project/checkpoints/HuggingFaceVLA_smolvla_libero \
  --output_dir runs/nominal_ppo/libero10_task0 \
  --noise_level 0.1 \
  --actor_lr 1e-5 --critic_lr 1e-3 \
  --gamma 0.99 --gae_lambda 0.95 \
  --regime expert_full_vlm_lora \
  --updates 20 --rollout_steps 32 --action_steps 1 --epochs 1 \
  --episode_length 520 --denoise_steps 10 \
  --eval_every 20 --eval_episodes 2 --eval_horizon 520 \
  --ckpt_every 20 --seed 0
```

Physical GPU 2 was used because, at launch, it had the most free memory of the four cards. It was not idle: about 39 GB of 96 GB was already in use, at high utilization. The process's own nvidia-smi memory, sampled every 20 seconds, peaked at 3714 MiB.

The command above is what `runs/nominal_ppo/libero10_task0/run_config.json` recorded, including the defaults the trainer filled in: adapter rank 8, alpha 16, clip 0.2, value coefficient 0.5, gradient clip 1.0, KL coefficient 0.

## What the run did

Exit code 0. `TRAINING_COMPLETE`. Wall clock 1229 s, of which the 20 collect-and-update cycles account for 623.8 s in `metrics.jsonl`. The rest is model load, environment startup, and the final evaluation.

Manifest, same surface as the one-update feasibility check:

```text
trainable=99.40M of 606.88M (16.379%)
action_expert=96.70M action_head=0.76M backbone_adapters=1.64M critic=0.31M
```

The first optimizer step, before any parameter had moved, had ratio 1.000000. The tolerance printed by the trainer is 0.01. Later lines labelled `ratio` are the last of that cycle's 32 steps, not the first-step check and not a cycle mean.

| Quantity | Value |
| --- | --- |
| Environments | 4 copies of `libero_10:0` |
| Decisions per cycle | 32 per environment |
| Optimizer steps per cycle | 32, batch 4, one epoch |
| Training environment steps | 20 × 32 × 4 × 1 = 2560 |
| Cycles with a nonzero reward | 9, 11, and 12, each `reward_mean` 0.0078125 |
| Successes those means represent | 1 per cycle, because 0.0078125 × 128 = 1 |
| Updates skipped | 0 of 640 |
| Final deterministic eval | 0/8, seeds 7000–7007, horizon 520 |
| Eval environment steps | 2 × 4 × 520 = 4160, all ran to the cap |
| Checkpoint | `runs/nominal_ppo/libero10_task0/step_000020.pt`, 1.53 GiB |
| Sampled process GPU memory | 3714 MiB |

The reward is the sparse terminal success flag. A mean of 0.0078125 on a 4 × 32 window is one success inside that window, not a success rate. Three such windows fired. That is evidence the training loop can observe a solved episode and put a nonzero reward into GAE. It is not a count of how often the policy solves the task, because the log stores the mean rather than per-environment outcomes.

The final evaluation is the comparable deterministic number for these weights: zero successes in eight episodes. Those seeds overlap the screen's seed 7000, which had already failed in the ten-task batch, and they do not include seed 7010, which is where the screen's task-0 success was. This eval does not show that the 20 updates improved on the pretrained policy, and it is too small and too unmatched to show that they made it worse.

Ratio and gradient figures moved as soon as the updates started. On the logged last step of each cycle, `clip_frac` was often 0.75 or 1.0, and pre-clip gradient norms reached 449 and 401. The applied norm after the clip of 1.0 stayed near 1 on those steps. Cycle 10's last step had `ratio_mean` 3.01 and `ratio_max` 4.90. Those numbers say the clipped update was active. They do not say the policy got better.

## Diagnostics the log actually contains

`metrics.jsonl` has one record per cycle. The PPO fields in that record are the last of the cycle's 32 optimizer steps, merged from the dictionary `ppo_step_stored` returns. They are not means over the cycle. Reading them as a cycle summary overstates whatever the last step happened to be.

Over those 20 last steps, the values that were written are:

| Field | What was stored | Range across the 20 records |
| --- | --- | --- |
| `reward_mean` | Mean of the 4×32 sparse rewards in the cycle | 0, except cycles 9, 11, and 12 at 0.0078125 |
| `pg_loss`, `v_loss`, `loss` | Last step only | policy loss −1.067 to 1.658; value loss 9.9e−5 to 0.080; total loss −1.066 to 1.658 |
| `clip_frac` | Fraction of the last step's ratios outside the clip range | 0.25 to 1.0 |
| `grad_norm` | Norm before the clip of 1.0, last step | 0.048 to 449 |
| `grad_norm_applied` | Norm after that clip, last step | 0.048 to 1.001 |
| `value_mean` | Mean of the recomputed values on the last step | −0.695 to 0.348 |
| `ratio_mean`, `ratio_median`, `ratio_max` | Last step | mean 0.078 to 3.01; max 0.140 to 4.90 |
| `kl_to_reference` | Anchor term | 0 on every record |

`kl_to_reference` is the optional penalty against a frozen copy of the policy. The pilot set `--kl_coef 0`, which leaves that anchor off, so the field is zero because it was not computed. It is not an approximate KL between the stored policy and the updated one.

The following were computed during the update and not written down. No number for them is inferred here:

- Advantage mean, standard deviation, or range. `normalize_advantages` standardizes the whole 4×32 batch, then the tensor is consumed by the update and dropped.
- A reward distribution. Only the window mean is stored, so which of the four workers succeeded is not in the file. The three nonzero means equal `1/128`, which identifies one success in that window and nothing finer.
- Stored value predictions across the rollout. `value_mean` is the recomputed value on the last optimizer step.
- Approximate KL of the probability ratio.
- Clip fraction, losses, and gradient norms for the other 31 steps of each cycle.

All 640 optimizer steps were applied (`updates_skipped` is 0 on every record).

## Why seed 7000 disagreed with itself

The single-environment episode and the ten-task screen both name seed 7000 for task 0, and they ended differently: success at step 369, then failure at the horizon. A first-step probe reran the pretrained policy with no adapters and no value head, the same policy those two runs used, under `flow_ode` and bfloat16 autocast. The script is `scripts/matched_checkpoint_eval.py probe-seed`. Output: `runs/matched/probe/seed7000_first_step.json`.

Worker 0 was the same situation in a batch of 1, a batch of 4 copies of task 0, and the ten-task batch:

- `init_state_id` 0. In `evaluate`, `per_task` is 1, and `set_initial_states` uses `i % per_task`. That remainder is 0 for every worker, so a whole eval batch shares one initial-state id. The next batch shares the next id. Workers are not spread across initial states during evaluation.
- Reset seeds 7000, 7001, … for workers 0, 1, …
- The same instruction.
- The same image and the same 8-dimensional state. The maximum absolute difference against the single-environment observation was 0. In the four-copy batch the other rows matched row 0 as well.

The action did not match. Against the single-environment action, the maximum absolute difference was 0.119 in the four-copy batch and 0.151 in the ten-task batch. Inside the four-copy batch, where every observation was identical, the actions still differed by up to 0.316. A second process, given the same observation, also produced a different single-environment action from the first process (first component 0.0076, then −0.056).

That is what the sampler does. `evaluate` calls `sample_actions` without a noise tensor. `sample_actions` then draws `torch.normal` of shape `(batch, chunk, action width)`. `sde_sample` documents that evaluation is reproducible given the same starting noise, and in `flow_ode` the integration standard deviation is zero. The draw that is not pinned is the starting noise. A different batch shape is a different `torch.normal` shape, and rows inside one batch do not share a noise vector. Neither the baseline episode nor the ten-task screen called `torch.manual_seed` before sampling. The env seed and the initial state were matched. The action sample was not. The outcome difference does not show that the initial state differed.

## Matched checkpoint comparison

The comparison uses `evaluate` for both policies, four copies of `libero_10:0`, horizon 520, `n_action_steps` 1, `flow_ode`, and run seed 0. That seed is what `evaluate` turns into environment seeds 7000–7007. Both builds are `expert_full_vlm_lora` with rank 8 and alpha 16, which is the architecture the 20 updates used. `torch.manual_seed(0)` and the NumPy seed run before each build. Loading the checkpoint does not draw further random numbers. The initialization is the pretrained weights with zero-initialized adapters, not a separately downloaded file.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2
python scripts/matched_checkpoint_eval.py compare \
  --base-policy /user_data/Mozhgan/Projects/RL_Project/checkpoints/HuggingFaceVLA_smolvla_libero \
  --checkpoint runs/nominal_ppo/libero10_task0/step_000020.pt \
  --output-dir runs/matched/compare
```

Exit code 0. Loading `step_000020.pt` changed 491 tensors. Records are in `runs/matched/compare/comparison.json`.

| Policy | Successes | Which seeds |
| --- | --- | --- |
| Initialization, cycle label 0 | 2/8 | 7001 at step 329, 7003 at step 390 |
| `step_000020.pt` | 0/8 | none of 7000–7007 |

Seed 7000 failed for both. This is the controlled sample the in-training evaluation was missing: the same evaluator, the same batch, and the same seeds, with an initialization that has not been through the 20 updates. In this sample the updated policy succeeded less often. Eight episodes, one task, and one training seed are not enough to conclude that PPO reduced success, and they do not show that it improved it. Those episodes also did not pin the flow's starting noise. [REPRODUCIBLE_EVAL.md](./REPRODUCIBLE_EVAL.md) is the later mechanism for doing that. This comparison was not rerun under it.

The in-training evaluation was also 0/8, but its random stream had already been consumed by training. It is not this comparison.

## Limitations

- Twenty updates on four workers is a contact-scale pilot. It is not a training result.
- The logged PPO statistics are the last timestep of the cycle. A reader who averages them will not recover the cycle.
- GPU memory is the process figure from nvidia-smi at 20 second intervals, on a card shared with other jobs. It is not `torch.cuda.max_memory_allocated`, and a spike between samples would be missed. The samples sat at 3714 MiB from the first cycle through the evaluation.
- The ten-task screen and the four-copy evaluation do not isolate task difficulty from batch size. The single-environment success on task 0 remains the one in the baseline report.
- No second seed, no held-out initial states beyond the two eval batches, and no perturbation condition were run.
