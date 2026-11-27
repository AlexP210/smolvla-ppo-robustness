# Nominal RL experiments

This page is the first nominal-PPO learning pass after the diagnostics landed. It does not change the PPO loss, the Euler step, or the default of one executed action. Perturbations stay off. No learning-rate, clip, or noise sweep was launched.

The question is whether the upstream update improves SmolVLA on LIBERO-Long. The controlled reproduction of the original 20-update configuration does not show that it does. The diagnostics from that run are why the next measurement was three one-change stability pilots, recorded below, rather than a sweep. The moderate-task training config is still not launched.

## What was reviewed before any new run

`examples/single_task_contact.sh` labels its values as illustrative, including `--regime full`. The one value it treats as fixed is the episode length, 520, which is the suite limit. The rest are textbook settings: exploration noise 0.1, actor learning rate 1e-5, critic learning rate 1e-3, discount 0.99, GAE λ 0.95, rollout of 32 decisions, one epoch, one executed action, ten integration steps. One environment makes the PPO batch a single transition. The script turns the tripwire on. Its own comment says a cycle is about fifty seconds and one evaluation episode is about ten minutes, and that the script has not been tuned.

The pilot in [NOMINAL_PPO_PILOT.md](./NOMINAL_PPO_PILOT.md) kept that noise, those learning rates, the discount, the trace, the rollout, and one executed action. It switched the regime to `expert_full_vlm_lora`, the surface the trainer names as the recommended one, and it widened the batch by repeating `libero_10:0` four times. It ran 20 updates, left the tripwire off, and evaluated only at the end. The manifest was 99.40M trainable parameters out of 606.88M. The vision tower and connector stay frozen. The action expert and the flow action head train at full rank. The language backbone trains through adapters of rank 8 and alpha 16. The reference anchor stays at `kl_coef` 0, so the loss is the clipped surrogate plus `0.5` times the value loss.

Two facts from that pilot decide what not to sweep yet. The rollout is 32 environment steps, and the successes in the pretrained screen were past step 200. A 32-step window usually does not contain the terminal reward. The logged ratio and gradient were the last of 32 optimizer steps, so they were not a cycle summary.

## Tasks

The ten-episode Flow RL bfloat16 screen at one executed action, and the native LeRobot screen at the same horizon, are in [PRETRAINED_BASELINE_VALIDATION.md](./PRETRAINED_BASELINE_VALIDATION.md). Per task, out of 10:

| Task | Instruction | Flow bf16 | Native |
| --- | --- | --- | --- |
| 0 | alphabet soup and tomato sauce | 0 | 4 |
| 1 | cream cheese box and butter | 6 | 6 |
| 2 | stove and moka pot | 5 | 5 |
| 3 | black bowl in the drawer | 9 | 9 |
| 4 | two mugs on two plates | 0 | 3 |
| 5 | book into the caddy | 9 | 8 |
| 6 | mug and chocolate pudding | 8 | 9 |
| 7 | alphabet soup and cream cheese | 3 | 3 |
| 8 | both moka pots | 2 | 2 |
| 9 | mug in the microwave | 4 | 4 |

Task 0 is the reproduction task because it is the task the original 20 updates trained. It is not a moderate task under the Flow RL screen. Tasks 1, 2, and 9 sit between 4/10 and 6/10 on both evaluators. Tasks 3, 5, and 6 are already high. Tasks 4 and 8 are low under Flow RL. The saved next configuration uses tasks 1, 2, and 9. It was not launched.

## Scenarios

`eval_scenario_v1` is the object in [PROJECT_PLAN.md](./PROJECT_PLAN.md). Workstream C can add seen, unseen, and severity files with the same loader, `src/eval_scenario.py`. This branch adds the two nominal files and nothing with a perturbation.

Both files use horizon 520, one executed action, ten episodes, initial states 0 through 9, `env_seed` 1000, and `policy_seed` 2000. The training seed stays 0. Collection still resets through `train_reset_seeds`. The scenario seeds are a different `SeedSequence`, and the integers are not 0, so an evaluation episode is not a replay of a training reset. `perturbation` is null. A file with `action_steps` other than 1, a `libero_10` horizon other than 520, or fewer than 10 episodes is rejected.

The episode id is `init_state + task_id * 50 + replicate * 500`. The stored LIBERO state is that id modulo 50. A task keeps the same id if the workers are reordered. Four copies of one task, which is how the trainer widens the batch, get four noise rows and the same stored state.

`scripts/matched_checkpoint_eval.py scenario` gives one scenario object to the pretrained initialization and to a checkpoint. `compare` is still the earlier eight-episode call and is not a reported comparison. In-training evaluation passes the scenario only when `--eval_scenario` is set. The reproduction asks for two episodes, so it scores the first two initial states, four replicates each. That sample is not the ten-episode comparison.

## Reproduction

`configs/nominal_ppo_libero10_task0.json` is the pilot command plus the nominal scenario. The update itself is unchanged: noise 0.1, actor 1e-5, critic 1e-3, γ 0.99, λ 0.95, clip 0.2, value coefficient 0.5, gradient clip 1.0, KL coefficient 0, one epoch, rollout 32, one executed action, 20 updates, four copies of task 0, seed 0.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
python src/train_flow_rl.py --config configs/nominal_ppo_libero10_task0.json
```

The run completed. `TRAINING_COMPLETE`. The first update ratio was 1.000000, inside the trainer's tolerance of 0.01. All 640 optimizer steps were applied. The 20 collect-and-update cycles took 517.9 s. Output is `runs/nominal_rl/task0_repro/`, which is gitignored. The card was shared with other jobs.

The reward pattern matches the original pilot. Cycles 9, 11, and 12 each contain one success decision, on workers 0, 1, and 2 respectively. Every other cycle is 128 zeros. That is 3 successes in 20 × 32 × 4 = 2560 environment steps. The previously published last-step figures match as well, including cycle 10's last-step ratio 3.01 and cycle 8's last-step pre-clip gradient 449. The diagnostics below are a reading of that same configuration, not of a different draw.

### Two checkpoints

The file `runs/nominal_ppo/libero10_task0/step_000020.pt` is the earlier pilot. The file `runs/nominal_rl/task0_repro/step_000020.pt` is this reproduction. They used the same update, the same seed, and the same task. They are not the same file. The paired tables below name which file was loaded. The reproduction's own eight-episode sample is a third, shorter measurement and is not one of those tables.

### What one cycle actually did

Each record's top-level `ratio_mean` is still the last of the 32 optimizer steps. The numbers below are `diagnostics`, aggregated across all 32. Advantage mean and standard deviation are the raw GAE tensor, before `normalize_advantages`. After that call the mean is 0 and the population standard deviation is 1 on every cycle, including the 17 cycles with no reward. Value and return are the stored rollout tensors, not the values recomputed inside the update. Approximate KL is Schulman's `mean((ratio - 1) - log(ratio))` on the detached PPO ratio `pi_new / pi_old`. `kl_to_reference` is the optional anchor against a frozen copy. It is 0 on all 20 records because `kl_coef` is 0, so that field was not computed. It is not the approximate KL in this table.

| Cycle | Successes | Adv mean | Adv std | Value mean | Return mean | Expl. var | KL mean | KL max | Ratio mean | Ratio min | Ratio max | Clip | PG loss | V loss | Grad max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0.046 | −0.411 | −0.411 | −0.23 | 1.98 | 16.4 | 0.963 | 1e-13 | 19.2 | 0.766 | 0.153 | 0.085 | 369 |
| 1 | 0 | 0.086 | 0.052 | −0.390 | −0.304 | −9.48 | 0.385 | 4.60 | 0.938 | 1e-5 | 3.86 | 0.516 | 0.058 | 0.171 | 790 |
| 2 | 0 | 0.037 | 0.038 | −0.084 | −0.047 | −3.98 | 0.506 | 5.06 | 0.891 | 2e-4 | 5.38 | 0.578 | 0.087 | 0.047 | 418 |
| 3 | 0 | 0.042 | 0.034 | −0.073 | −0.031 | −1.14 | 0.504 | 4.71 | 0.973 | 7e-5 | 5.63 | 0.625 | 0.116 | 0.001 | 421 |
| 4 | 0 | −0.004 | 0.028 | −0.017 | −0.022 | −1.26 | 0.682 | 7.10 | 0.915 | 9e-8 | 5.67 | 0.539 | 0.098 | 0.001 | 702 |
| 5 | 0 | 0.009 | 0.035 | −0.025 | −0.016 | 0.30 | 0.905 | 9.74 | 0.810 | 2e-9 | 3.75 | 0.547 | 0.078 | 0.008 | 268 |
| 6 | 0 | −0.019 | 0.021 | 0.190 | 0.171 | 0.89 | 0.822 | 7.22 | 0.864 | 5e-8 | 3.04 | 0.602 | 0.166 | 0.027 | 542 |
| 7 | 0 | −0.025 | 0.035 | 0.293 | 0.268 | −1.25 | 0.426 | 2.62 | 0.842 | 1e-4 | 5.63 | 0.656 | 0.083 | 0.028 | 935 |
| 8 | 0 | −0.033 | 0.027 | 0.343 | 0.310 | 0.57 | 0.303 | 2.15 | 0.997 | 0.009 | 9.83 | 0.570 | 0.085 | 0.001 | 449 |
| 9 | 1 | 0.042 | 0.180 | 0.295 | 0.337 | −0.15 | 0.394 | 2.12 | 0.902 | 0.002 | 6.23 | 0.617 | 0.085 | 0.029 | 178 |
| 10 | 0 | −0.004 | 0.024 | 0.088 | 0.084 | 0.77 | 0.268 | 2.13 | 1.06 | 0.004 | 6.67 | 0.539 | 0.132 | 0.014 | 372 |
| 11 | 1 | 0.050 | 0.189 | −0.017 | 0.032 | −0.02 | 0.289 | 2.59 | 0.866 | 3e-4 | 3.62 | 0.586 | 0.012 | 0.019 | 829 |
| 12 | 1 | 0.009 | 0.101 | −0.047 | −0.038 | −0.04 | 0.874 | 16.9 | 0.964 | 3e-11 | 6.47 | 0.578 | 0.008 | 0.007 | 1763 |
| 13 | 0 | 0.012 | 0.009 | −0.132 | −0.121 | 0.94 | 0.425 | 3.40 | 0.823 | 1e-4 | 2.43 | 0.664 | 0.127 | 0.002 | 891 |
| 14 | 0 | 0.015 | 0.013 | −0.170 | −0.155 | 0.89 | 0.498 | 3.35 | 0.932 | 1e-4 | 6.15 | 0.570 | 0.023 | 0.001 | 415 |
| 15 | 0 | 0.021 | 0.016 | −0.190 | −0.168 | 0.91 | 0.370 | 2.50 | 0.957 | 0.004 | 4.57 | 0.609 | 0.070 | 0.011 | 803 |
| 16 | 0 | 0.018 | 0.033 | −0.101 | −0.083 | −1.10 | 0.485 | 4.22 | 0.956 | 7e-5 | 7.54 | 0.523 | 0.065 | 0.004 | 328 |
| 17 | 0 | 0.005 | 0.013 | −0.105 | −0.101 | 0.61 | 0.374 | 2.30 | 0.882 | 0.004 | 8.45 | 0.625 | 0.125 | 0.006 | 525 |
| 18 | 0 | −0.003 | 0.013 | −0.019 | −0.022 | 0.82 | 0.221 | 1.19 | 0.927 | 0.005 | 3.97 | 0.477 | 0.081 | 0.001 | 589 |
| 19 | 0 | 0.002 | 0.010 | −0.035 | −0.033 | 0.90 | 0.230 | 2.56 | 1.00 | 2e-5 | 4.26 | 0.445 | 0.125 | 0.0004 | 401 |

The reward column is success decisions in that cycle's 128 steps, not a count of episode identifiers. Cycles 9, 11, and 12 have one each, on workers 0, 1, and 2. The other 17 cycles are all zeros. Three decisions in 2560 environment steps is the whole sparse-reward signal this run observed. The gradient max is the largest pre-clip norm among the 32 steps. The cycle mean of that pre-clip norm stays between 62.6 and 135.4. After the clip of 1.0 the applied norm's cycle mean stays between 0.87 and 1.00, and the largest applied norm in a cycle is about 1.00. No step was skipped.

The cycle-mean ratio stays between 0.81 and 1.06. The last step, which is all the old pilot stored, swings from 0.078 to 3.01. Inside a cycle the ratio still reaches 1e-13 and 19.2. Those extremes are in the approximate KL. They are not in `kl_to_reference`.

The trust region is already broken on cycle 0, before any success: mean approximate KL 1.98, one step at 16.4, ratio up to 19. The later cycles do not run away into non-finite updates. Mean KL stays between 0.22 and 0.91, and all 640 steps are applied. The run is stable in the narrow sense that it finishes. It is not stable in the PPO sense that the probability ratio stays near 1. That departure starts at the first cycle and is still present at cycle 19.

### Why the actor moves when the task does not

`normalize_advantages` divides by the window's own standard deviation. A zero-reward window has raw advantage standard deviation 0.009 to 0.052, which is the critic residual. A window with one success has standard deviation 0.10 to 0.19. After standardization both have standard deviation 1, so the actor sees the same scale either way. Seventeen of the twenty updates are the residual kind.

The critic does not explain the returns on the three cycles that contain the only rewards. Explained variance there is −0.15, −0.02, and −0.04. It is high on several zero-reward cycles, where the target is close to the critic's own bootstrap. Stored values and returns track each other across the run, from about −0.41 at cycle 0 to a peak near 0.34 at cycle 8 and back toward 0, while the reward is still almost always zero. The value targets are moving. The task successes are not what is moving them.

### What in the configuration can do that

The loss line is unchanged: `pg_loss + 0.5 * v_loss + kl_coef * kl_penalty`, with `kl_coef` 0. The actor learning rate is 1e-5 and the critic learning rate is 1e-3, both from the illustrative contact script. The gradient is clipped to norm 1, so a pre-clip norm of several hundred still takes a step of size about the learning rate. There are 32 such steps per cycle, one per stored timestep, on a batch of four. `epochs` is already 1. The trainer has no flag for fewer steps than the rollout length. Taking fewer steps would mean changing that loop, which this page does not do.

Exploration noise is 0.1 during collection. Evaluation integrates with `flow_ode`, whose extra noise scale is zero. The first optimizer step, before any parameter moved, had ratio 1.000000. The stored likelihood and the recomputed likelihood match, so the drift after that step is the update, not a mismatch between collection and the PPO ratio. One related detail is not that check: the bootstrap value at the end of a rollout is computed in eval mode, while the stored values are computed in train mode. It can move the GAE tail. It does not explain a first-step ratio of 1, and it is not changed here.

The reward is the sparse terminal success, max-folded over the one executed action. GAE carries it backward only inside the 32-step window, with γ 0.99 and λ 0.95, and a done flag stops the trace at an episode boundary. The paired successes on these tasks occur between simulator steps 217 and 437. A 32-step window inside a 520-step episode usually does not contain that event. When it does not, the advantage is a rescaled value residual and the actor still steps.

The trainable surface is 99.40M parameters of 606.88M: the action expert and the action head at full rank, language adapters of rank 8, and the critic. The vision tower stays frozen. With `kl_coef` 0 nothing pulls those 99.40M parameters back toward the pretrained policy. The clip limits a single step. It does not limit the distance after 640 steps.

### In-training sample of the reproduction

The reproduction's own evaluation scores the new file, `runs/nominal_rl/task0_repro/step_000020.pt`, on the first two initial states, four noise replicates each. All eight failed. Pooled success printed by the trainer was 0. Eight episodes are not the reported comparison. The ten-episode score of this same file is the section below the earlier checkpoint.

## Paired comparison of the earlier checkpoint

The checkpoint in this section is the earlier pilot, `runs/nominal_ppo/libero10_task0/step_000020.pt`. It is not the reproduction's file. The initialization is the pretrained policy with zero-initialized adapters, which is what that pilot started from. Both sides of each table receive one scenario object. Loading this checkpoint changed 491 tensors. The runs were on a shared GPU.

```bash
python scripts/matched_checkpoint_eval.py scenario \
  --base-policy /user_data/Mozhgan/Projects/RL_Project/checkpoints/HuggingFaceVLA_smolvla_libero \
  --checkpoint runs/nominal_ppo/libero10_task0/step_000020.pt \
  --scenario configs/scenarios/libero10_nominal.json \
  --output-dir runs/nominal_rl/paired_task0
```

The same command with `configs/scenarios/libero10_moderate.json` scores tasks 1, 2, and 9. Those tasks were not the ones the checkpoint trained on. The comparison asks whether the task-0 update changed them, and it records the pretrained rates under this scenario. Raw records are gitignored under `runs/nominal_rl/paired_task0/` and `runs/nominal_rl/paired_moderate/`.

### Task 0, ten initial states

Exit code 0. Loading the checkpoint changed 491 tensors. The initialization took 2631 s and the checkpoint 2777 s, on a shared GPU.

| Policy | File | Successes | Success steps | By simulator step 100, 200, 300, 400, 520 |
| --- | --- | --- | --- | --- |
| Initialization | pretrained weights, zero adapters | 2/10 | 301, 425 | 0, 0, 0, 1, 2 |
| Earlier 20-update checkpoint | `runs/nominal_ppo/libero10_task0/step_000020.pt` | 0/10 | none | 0, 0, 0, 0, 0 |

Wall clock was 2631 s and 2777 s. The step-300 column is still zero because the first success is at step 301. The checkpoint failed the two initial states the initialization solved, and the other eight. This sample does not show that the 20 updates improved task 0. Ten episodes are too few to establish that they made it worse.

### Tasks 1, 2, and 9, which this checkpoint was not trained on

The same earlier file, scored on `configs/scenarios/libero10_moderate.json`. Wall clock was 3734 s and 3891 s.

| Task | Initialization | Earlier checkpoint | Initialization success steps |
| --- | --- | --- | --- |
| 1 | 4/10 | 0/10 | 279, 304, 321, 437 |
| 2 | 4/10 | 0/10 | 217, 250, 272, 318 |
| 9 | 4/10 | 0/10 | 263, 288, 322, 351 |

Across all 30 episodes the initialization has 0 successes by step 200, 6 by step 300, 11 by step 400, and 12 by step 520. The earlier checkpoint has 0 at every one of those budgets.

Together with task 0, the initialization succeeds on 14 of 40 episodes and the earlier checkpoint on 0 of 40. The zeros include the task the update trained on and three tasks it never trained on. That is the observation. It is not a statistical finding of catastrophic forgetting, and it is not evidence that PPO learned the training task. The successes sit hundreds of simulator steps into the episode, which is the same scale the 32-step rollout rarely covers.

## The reproduction's checkpoint

This section scores `runs/nominal_rl/task0_repro/step_000020.pt`. The earlier section scores `runs/nominal_ppo/libero10_task0/step_000020.pt`. Both files are update 20 of the same configuration and the same seed. Loading the reproduction file changed 491 tensors. That count matches the earlier file and does not make the two weight files the same object.

The initialization numbers below are the pretrained run already recorded on these two scenario files. This pass scored the reproduction checkpoint only. It used `configs/scenarios/libero10_nominal.json` and `configs/scenarios/libero10_moderate.json`, with `env_seed` 1000 and `policy_seed` 2000. Task 0 took 1976 s. Tasks 1, 2, and 9 took 5021 s. Both were on a shared GPU. Records are gitignored under `runs/nominal_rl/paired_repro_task0/` and `runs/nominal_rl/paired_repro_moderate/`.

| Task | Reproduction successes | Success steps | By simulator step 100, 200, 300, 400, 520 |
| --- | --- | --- | --- |
| 0 | 0/10 | none | 0, 0, 0, 0, 0 of 10 |
| 1 | 0/10 | none | 0, 0, 0, 0, 0 of 10 |
| 2 | 0/10 | none | 0, 0, 0, 0, 0 of 10 |
| 9 | 0/10 | none | 0, 0, 0, 0, 0 of 10 |

Across the 30 episodes of tasks 1, 2, and 9, the reproduction checkpoint has 0 successes at every one of those budgets.

| Policy | File | Task 0 | Tasks 1, 2, and 9 | Combined |
| --- | --- | --- | --- | --- |
| Initialization | pretrained weights, zero adapters | 2/10 | 12/30 | 14/40 |
| Earlier 20-update checkpoint | `runs/nominal_ppo/libero10_task0/step_000020.pt` | 0/10 | 0/30 | 0/40 |
| Reproduction checkpoint | `runs/nominal_rl/task0_repro/step_000020.pt` | 0/10 | 0/30 | 0/40 |

The eight-episode in-training sample of the reproduction file was also 0/8, on initial states 0 and 1 with four noise replicates. That sample is not this table.

On these 40 episodes the reproduction checkpoint has no successes, on task 0, which the update trained, and on tasks 1, 2, and 9, which the update did not train. The earlier checkpoint has the same 0/40 on the same scenarios, where the initialization has 14/40. Each column is one paired sample. The drop covers the training task and tasks excluded from the RL task list. It does not establish catastrophic forgetting, and it does not establish that PPO learned the training task.

## Stability pilots

Each file still changes one field of the reproduction and stays on four copies of task 0, seed 0, rollout 32, noise 0.1, and one executed action. The loss line is unchanged. `configs/nominal_ppo_libero10_moderate.json` was not launched. The three-cycle commands override `--updates` and `--eval_every` because the trainer refuses an evaluation schedule longer than the run, and it also refuses an `eval_scenario` whose schedule is switched off. The PPO hyperparameters in the files are otherwise the ones that ran. Output directories for the three-cycle pilots are separate from the 20-update directories, so the longer runs start from the pretrained initialization.

| File | The one change | What ran |
| --- | --- | --- |
| `configs/ablations/actor_lr_1e-6.json` | Actor learning rate 1e-6 | Three cycles, then 20 updates |
| `configs/ablations/kl_coef_0.01.json` | `kl_coef` 0.01 | Invoked. The trainer exited before cycle 0 |
| `configs/ablations/regime_lora.json` | Regime `lora`, with `--control_arm` | Three cycles, then 20 updates |

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=1 python src/train_flow_rl.py \
  --config configs/ablations/actor_lr_1e-6.json \
  --updates 3 --eval_every 3 \
  --output_dir runs/nominal_rl/ablate_actor_lr_u3

CUDA_VISIBLE_DEVICES=2 python src/train_flow_rl.py \
  --config configs/ablations/kl_coef_0.01.json \
  --updates 3 --eval_every 3 \
  --output_dir runs/nominal_rl/ablate_kl_u3

CUDA_VISIBLE_DEVICES=0 python src/train_flow_rl.py \
  --config configs/ablations/regime_lora.json \
  --updates 3 --eval_every 3 \
  --output_dir runs/nominal_rl/ablate_lora_u3
```

The 20-update commands are the files as written: actor learning rate on GPU 2, LoRA on GPU 1. Process GPU memory during the three-cycle runs was 3714 MiB for the full-rank actor-rate job and 2246 MiB for LoRA. The 20-update jobs load the same models. Wall times below are shared-machine times.

### Three cycles against the reproduction

The reproduction's first three cycles, from the table above, had mean approximate KL 1.98, 0.39, and 0.51, clip fractions 0.77, 0.52, and 0.58, and ratio maxima 19.2, 3.86, and 5.38. Both pilots that reached an update were calmer on that same window. The first-step ratio was 1.000000 in both. `kl_to_reference` stayed 0 because `kl_coef` stayed 0. No optimizer step was skipped. Advantages are the raw GAE values. After standardization the mean is 0 and the standard deviation is 1 on these cycles too.

| Run | Cycle | Successes | Adv std | Value mean | Return mean | Expl. var | KL mean | KL max | Ratio min | Ratio max | Clip | Grad max | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Actor 1e-6 | 0 | 0 | 0.046 | −0.411 | −0.411 | −0.23 | 0.033 | 0.24 | 0.31 | 2.02 | 0.26 | 688 | 31 |
| Actor 1e-6 | 1 | 0 | 0.036 | −0.290 | −0.300 | 0.07 | 0.021 | 0.30 | 0.47 | 3.28 | 0.13 | 1127 | 29 |
| Actor 1e-6 | 2 | 0 | 0.023 | −0.182 | −0.201 | 0.44 | 0.028 | 0.31 | 0.25 | 1.93 | 0.16 | 538 | 30 |
| LoRA | 0 | 0 | 0.046 | −0.411 | −0.411 | −0.23 | 0.0046 | 0.049 | 0.58 | 1.28 | 0.047 | 37 | 49 |
| LoRA | 1 | 0 | 0.036 | −0.254 | −0.275 | 0.24 | 0.0032 | 0.026 | 0.84 | 1.34 | 0.031 | 61 | 36 |
| LoRA | 2 | 0 | 0.025 | −0.337 | −0.315 | 0.04 | 0.0042 | 0.036 | 0.74 | 1.54 | 0.078 | 57 | 40 |

The actor-rate wall time for the whole three-cycle command, including its eight-episode sample, was 626 s. The LoRA command took 1121 s. Those eight-episode samples scored 4/8 and 2/8. They use the first two initial states and are not the ten-episode comparison.

The lower rate still clips. Pre-clip gradient means on those three cycles were 161 to 201, and the applied norm stayed at 1. The smaller KL is the smaller step length after that clip. LoRA's pre-clip means were about 13 to 14, so the clip binds on the large steps and not on every step.

### The reference anchor does not run on this surface

`configs/ablations/kl_coef_0.01.json` keeps `expert_full_vlm_lora` and sets `kl_coef` to 0.01. The process exited 1 after about 20 s, with the manifest already printed at 99.40M trainable. The trainer's message is that this regime trains part of the policy in place, so switching adapters off does not recover the pretrained policy.

That check is `reference_is_available`. It returns true only when every trainable parameter of the flow model is an adapter. In `expert_full_vlm_lora` the action expert body and the action head require gradients, so the check is false and the run stops before collection. The penalty that would have been added is `mean(ref_logp.detach() - new_logp)` on the executed action dimensions, with the reference taken by disabling adapters. Disabling adapters leaves an in-place change to the action expert where it is. The anchor therefore does not constrain the action-generating policy in the prepared file. The trainer refuses that configuration rather than allocating a second copy of the weights. A regime the same check would accept is `frozen` or `lora`, which would be a second change from the reproduction. That combination was not launched. `kl_to_reference` in the other pilots is 0 because their `kl_coef` is 0. It is still not the approximate KL.

### What LoRA actually updates

The LoRA manifest is 1.42M trainable of 606.35M: 1.11M under the action-expert name, 0 on the action head, 0 on backbone adapters, and 0.31M in the critic. The surface line says 0 of 96,696,800 of the expert's own parameters carry gradients. The 1.11M figure is the rank-8 adapters injected into the expert. Their names contain `lm_expert`, so the manifest counts them in that group. The run is declared a control.

Comparing each checkpoint with a fresh initialization of the same regime:

| Group | Actor 1e-6, 3 cycles | Actor 1e-6, 20 updates | LoRA, 3 cycles | LoRA, 20 updates |
| --- | --- | --- | --- | --- |
| Expert body, RMS of the change | 1.9e-6 | 4.8e-6 | 0 | 0 |
| Expert adapters, RMS | none | none | 0.025 | 0.025 |
| Language adapters, RMS | 0.020 | 0.020 | 0 | 0 |
| Action head, RMS | 1.0e-5 | 2.8e-5 | 0 | 0 |
| Critic, RMS | 0.113 | 0.113 | 0.113 | 0.113 |
| Remaining weights | 0 | 0 | 0 | 0 |

The largest absolute change in the expert body after 20 updates at 1e-6 is 1.3e-4. About 4.3 million of its 96.7 million elements are nonzero in the difference. The language adapters move by an RMS of 0.020, and that distance is already present at cycle 3. LoRA leaves the expert body, the action head, and the vision and language weights unchanged. The adapters and the critic are what move. The critic RMS is about 0.11 in both runs, which matches the critic learning rate of 1e-3 staying put while the actor rate or the actor surface changes.

### Twenty updates of the two calmer configurations

Both of the runs that finished three stable cycles were extended to 20 updates, the same 2560 environment steps as the reproduction. The actor-rate cycles took 1289 s and the whole command 3068 s. LoRA's cycles took 474 s and the whole command 1048 s. All 640 optimizer steps were applied in each run. `kl_to_reference` remained 0.

| Run | Success decisions | KL mean, min to max | Clip, min to max | Ratio min | Ratio max | Value mean | Return mean | Expl. var | Pre-clip grad mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reproduction | 3 | 0.22 to 1.98 | 0.45 to 0.77 | ~1e-13 | 19.2 | −0.41 to 0.34 | −0.41 to 0.34 | −9.48 to 0.94 | 63 to 135 |
| Actor 1e-6 | 2, cycles 9 and 13 | 0.0085 to 0.033 | 0.094 to 0.26 | 0.25 | 3.28 | −0.41 to 0.22 | −0.41 to 0.19 | −1.34 to 0.81 | 127 to 201 |
| LoRA | 1, cycle 8 | 0.0019 to 0.0076 | 0.016 to 0.11 | 0.56 | 1.57 | −0.41 to −0.07 | −0.41 to 0.04 | −1.52 to 0.87 | 8.5 to 16 |

The actor-rate pre-clip means stay above 100, so the clip of 1.0 still sets the applied step and the learning rate sets its length. LoRA's pre-clip means are near 10, and the applied-norm mean dips to 0.96, so some of its steps are inside the clip. Raw advantage standard deviations reach 0.23 and 0.22 on the cycles that contain the sparse success, and stay near 0.01 to 0.05 otherwise. Standardization still rescales those windows to standard deviation 1.

The in-training sample at cycle 20, eight episodes on initial states 0 and 1, scored 2/8 for the actor-rate checkpoint and 1/8 for LoRA. That sample is not the table below.

### Ten-episode scores of the 20-update checkpoints

The initialization column is the pretrained run already recorded on these scenario files. These passes scored the new checkpoints only, with `env_seed` 1000 and `policy_seed` 2000. Loading the actor-rate file changed 491 tensors. Loading the LoRA file changed 261 tensors.

| Policy | File | Task 0 | Tasks 1, 2, and 9 | Combined |
| --- | --- | --- | --- | --- |
| Initialization | pretrained weights, zero adapters | 2/10 | 12/30 | 14/40 |
| Actor learning rate 1e-6 | `runs/nominal_rl/ablate_actor_lr/step_000020.pt` | 3/10 | 13/30 | 16/40 |
| LoRA control | `runs/nominal_rl/ablate_lora/step_000020.pt` | 2/10 | 15/30 | 17/40 |
| Earlier checkpoint, actor 1e-5 | `runs/nominal_ppo/libero10_task0/step_000020.pt` | 0/10 | 0/30 | 0/40 |
| Reproduction, actor 1e-5 | `runs/nominal_rl/task0_repro/step_000020.pt` | 0/10 | 0/30 | 0/40 |

Actor learning rate 1e-6, task 0, wall 4757 s. Success steps 365, 378, and 461. By simulator step 100, 200, 300, 400, and 520: 0, 0, 0, 2, 3 of 10. Tasks 1, 2, and 9 took 5515 s.

| Task | Successes | Success steps |
| --- | --- | --- |
| 1 | 4/10 | 300, 323, 326, 382 |
| 2 | 4/10 | 192, 225, 262, 279 |
| 9 | 5/10 | 265, 281, 310, 312, 432 |

Across those 30 episodes the counts by steps 100, 200, 300, 400, and 520 are 0, 1, 7, 12, and 13.

LoRA, task 0, wall 2155 s. Success steps 321 and 399. By the same budgets: 0, 0, 0, 2, 2 of 10. Tasks 1, 2, and 9 took 5548 s.

| Task | Successes | Success steps |
| --- | --- | --- |
| 1 | 4/10 | 279, 300, 307, 516 |
| 2 | 6/10 | 210, 242, 262, 265, 272, 289 |
| 9 | 5/10 | 275, 284, 292, 331, 444 |

Across those 30 episodes the counts by steps 100, 200, 300, 400, and 520 are 0, 0, 11, 13, and 15.

On this paired sample both calmer checkpoints succeed on the training task and on the three tasks the update did not train. The counts sit next to the initialization's 14/40. They do not establish that PPO improved success. They also do not repeat the 0/40 of the actor learning rate 1e-5 checkpoints. Ten episodes per task are too few to treat either difference as a measured gain or as a measured preservation of the pretrained policy.

No learning recipe is selected. LoRA kept mean approximate KL below 0.008 for all 20 cycles and is a declared control whose action expert body does not train. The lower learning rate kept the recommended surface and kept mean approximate KL below 0.034, with ratio maxima still as large as 3.28. The moderate-task configuration stays unlaunched.

## Changes that were not put into these runs

Four other levers show up in the trainer, and none of them was added to the three files above.

KL early stopping inside a cycle would break the loop `for t in range(rollout_steps)` when the logged approximate KL, which is already the PPO-ratio estimator, crosses a threshold. The loss line would stay `pg_loss + value_coef * v_loss + kl_coef * kl_penalty`. The number of the 32 sequential steps that actually run would change. The existing tripwire is a different switch: it is off in these configs, it watches the last step's ratio over whole cycles, and it stops the run rather than the inner loop.

A longer rollout is `--rollout_steps`. GAE and the reward window would grow with it, which matters because the paired successes on these tasks sit between simulator steps 192 and 516. The same flag currently also sets the number of optimizer steps, one per stored timestep. Lengthening the rollout without changing that loop would add steps of the kind the lower learning rate was shrinking. That is two changes, and it was not made.

A critic-only warm-up would leave the actor parameter group unstepped for some cycles. The optimizer already has separate actor and critic groups, and every `ppo_step_stored` call steps both. There is no cycle range that skips the actor. The loss formula could stay. The schedule would change. These logs show the critic moving by an RMS near 0.11 while the full-rank expert body at learning rate 1e-6 barely moves, which is why a warm-up remains a later experiment.

A larger batch is more copies in `--tasks`. That changes the four-row gradient inside each of the 32 steps and leaves the 32 steps in place. Gradient accumulation is already an argument of `ppo_step_stored` (`accum_index`, `accum_total`) and the trainer never passes it, so each timestep is its own optimizer step. Wiring it would backward once per timestep and step once per group of timesteps, which reduces the number of updates per rollout without changing the loss. One forward still has to belong to one timestep: the trainer's comment says a wider slice would pair one observation with several timesteps' trajectories. That wiring was not added.

## Limitations

- Twenty updates and 2560 environment steps are a contact-scale run. A handful of success decisions is not a learning curve.
- Fourteen, sixteen, and seventeen successes in forty episodes, and zero for each actor learning rate 1e-5 checkpoint, are one paired sample per file. They do not establish that PPO helped, and they do not establish catastrophic forgetting.
- The lower learning rate and the LoRA control are calmer optimizations on this sample. They are not a selected training recipe.
- The GPUs were already busy. The wall times are shared-machine times. Process memory above is one sample during the three-cycle runs.
- In-training evaluation is eight episodes. The reported comparison is the ten-episode scenario table.
- Seen, unseen, and severity scenarios are not in this branch. The perturbation package was not added.
- The old 2/8 versus 0/8 comparison was not recomputed. It did not pin the flow noise.
- `configs/nominal_ppo_libero10_moderate.json` has not been run. The reference-KL file was invoked and refused, and it was not rewritten into a second change.
