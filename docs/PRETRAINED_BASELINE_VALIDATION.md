# Pretrained baseline validation

The earlier LIBERO-Long screen scored 7/20 with the Flow RL evaluator. The [LIBERO VLA leaderboard](https://huggingface.co/spaces/HuggingFaceVLA/libero-vla-leaderboard) lists 0.60 in the Long column for `HuggingFaceVLA/smolvla_libero`. This page is the check of that checkpoint with the native LeRobot 0.4.4 evaluator, and of the Flow RL sampler on the same weights. PPO was not run, and the trainer's default of one executed action was not changed.

## What the leaderboard and the checkpoint say

The leaderboard table is hardcoded in the space. For `HuggingFaceVLA/smolvla_libero` it lists Spatial 0.90, Object 1.0, Goal 1.0, LIBERO-90 as `--`, Long 0.60, and an average of 0.87. The average is the mean of the four reported suites. The 0.60 figure is LIBERO-Long (`libero_10`), which is the suite in the 7/20 screen.

The space's contribution example runs `n_action_steps=10`, one episode, and `libero_spatial`, against a local directory named `smolvla_pipe`. It is an example command. It is not a record of the evaluation that produced the 0.60.

The downloaded checkpoint is Hugging Face revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`. Its `config.json` sets `n_action_steps` to 1, `chunk_size` to 50, and `num_steps` to 10. Images are resized with padding to 512×512. State and action normalization are mean/standard deviation. The declared cameras are `observation.images.image` and `observation.images.image2`. A LeRobot issue records that the paper's LIBERO-Long figure is 0.71 and that the authors evaluated with `n_action_steps=1` ([huggingface/lerobot#2354](https://github.com/huggingface/lerobot/issues/2354)). This run did not re-read the paper PDF.

## Stack

| Package | Version |
| --- | --- |
| Python | 3.10.22 |
| LeRobot | 0.4.4 |
| MuJoCo | 3.8.1 |
| hf-libero | 0.1.5 |
| torch | 2.10.0 |
| gymnasium | 1.4.0 |

Rendering used `MUJOCO_GL=egl`. The four cards already had other jobs on them, so the times below are wall-clock times on a shared machine. MuJoCo 3.4 and later changes how some stored LIBERO-Spatial initial states settle ([huggingface/lerobot#4390](https://github.com/huggingface/lerobot/issues/4390)). This environment is 3.8.1. That issue is about a spatial task. It was not isolated here.

## Observation and action pipeline

Both paths use LeRobot's LIBERO environment and processors. The measurement on task 0, initial state 0, reset seed 0:

| Check | Result |
| --- | --- |
| Simulator image | 256×256×3, `agentview_image` mapped to `image` and `robot0_eye_in_hand_image` mapped to `image2` |
| After the shared preprocessor | 3×256×256, float, channels first. Main and wrist differ from a second pass through the same processors by 0 |
| After `prepare_images` | 3×512×512, the checkpoint's padded resize. Pixel values are mapped to [-1, 1] |
| State | 8 values: end-effector position, quaternion converted to axis-angle, and two gripper positions. Mean/standard-deviation normalization, then zeros padded to 32. The two pipelines match exactly |
| Instruction | `put both the alphabet soup and the tomato sauce in the basket`, the same string in both pipelines |
| Control | relative, which is the environment default |
| Action | crop the model width of 32 down to 7, then the checkpoint's mean/standard-deviation unnormalizer |

`LiberoProcessorStep` flips each image on both spatial axes. That flip is inside the processor both pipelines call.

## One shared observation and one shared noise tensor

The noise is standard normal, float32, shape `(1, 50, 32)`. LeRobot's `sample_actions` and the Flow RL `flow_ode` loop consumed the same tensor. Float32 output matches exactly, including the unnormalized action that would be sent to the simulator. The maximum absolute difference is 0.

The project's `evaluate` wraps the forward in bfloat16 autocast. On this observation that changes the normalized chunk by at most 0.020 and the executed unnormalized action by at most 0.0025. The Euler step itself is the library's `x + dt * v` with `dt = -1/10`. The float32 match is the check that the wrapper is driving that step.

## Native LeRobot 0.4.4

The command is `python -m lerobot.scripts.lerobot_eval` from the installed 0.4.4 package. It does not import the Flow RL sampler. Settings: `libero_10`, 10 episodes per task, batch size 1, synchronous environments, one task at a time, `use_amp=false`, seed 0, the local checkpoint. The library renders up to 10 videos per task. Its own `set_seed(0)` feeds the global generator, and LIBERO advances the stored initial-state index on every reset, including the reset a successful step performs internally. Episode index and initial-state index are therefore not the same number after the first success.

| Executed actions | Success | Per task, out of 10 | Rollout | Per episode |
| --- | --- | --- | --- | --- |
| 1 | 53/100 | 4, 6, 5, 9, 3, 8, 9, 3, 2, 4 | 13114 s | 131 s |
| 10 | 46/100 | 2, 8, 5, 6, 1, 8, 5, 1, 5, 5 | 2571 s | 26 s |

A 95% Wilson interval for 53/100 is 43% to 63%. The leaderboard's 60% sits inside it. The interval for 46/100 is 37% to 56%, which leaves 60% outside. The checkpoint ships with one executed action. The contribution example uses ten. On this stack, one executed action scored higher.

## Flow RL, same weights, controlled seeds

Ten tasks run together. Episode `k` of every task starts at stored initial state `k`. The reset seed and the standard-normal chunk noise are functions of the task, the episode, and the environment step, so horizons 1, 5, 10, 25, and 50 share a draw whenever they replan on the same step. The episode still runs to 520 environment steps. Horizons 25 and 50 do not divide 520, and the last plan executes only the remaining steps. Bfloat16 is the precision `evaluate` uses. Float32 was repeated at 1 and 10 because that is the native evaluator's precision.

| Actions | Precision | Success | Per task, out of 10 | Inference | Wall |
| --- | --- | --- | --- | --- | --- |
| 1 | bf16 | 46/100 | 0, 6, 5, 9, 0, 9, 8, 3, 2, 4 | 3763 s | 4282 s |
| 5 | bf16 | 44/100 | 3, 4, 6, 6, 0, 9, 7, 2, 3, 4 | 749 s | 1017 s |
| 10 | bf16 | 41/100 | 4, 7, 4, 6, 0, 7, 6, 1, 4, 2 | 374 s | 610 s |
| 25 | bf16 | 47/100 | 3, 3, 6, 7, 1, 8, 7, 3, 3, 6 | 151 s | 390 s |
| 50 | bf16 | 30/100 | 0, 1, 5, 5, 0, 7, 5, 3, 1, 3 | 79 s | 277 s |
| 1 | fp32 | 42/100 | 0, 5, 5, 7, 0, 9, 8, 2, 2, 4 | 2576 s | 3047 s |
| 10 | fp32 | 40/100 | 2, 7, 4, 7, 0, 7, 6, 0, 5, 2 | 253 s | 479 s |

Task 4, the two mugs, is 0/10 at the short horizons and 1/10 at 25. Executing all 50 predicted actions drops the suite to 30/100. Horizons 1 through 25 stay between 41/100 and 47/100 under bfloat16.

## Where the 7/20 and the 0.60 come apart

The 7/20 screen was two episodes per task, one executed action, a batch of ten, bfloat16, and an unpinned global generator. Its Wilson interval is 18% to 57%. The 10-episode native result at the same horizon is 53/100. The screen was a small draw from a policy that, under the native evaluator, scores in the low fifties on this stack.

The float32 wrapper matches LeRobot's sampler on a shared observation and a shared noise tensor. A suite-level gap between 53/100 native and 42/100 or 46/100 Flow RL is a gap between two episode schedules: batch size, which stored state is selected after a success reset, and which noise is drawn. It is not a different first action from the same inputs. Bfloat16 versus float32, with the Flow RL seeds held fixed, moves horizon 1 from 42/100 to 46/100 and horizon 10 from 40/100 to 41/100.

The leaderboard Long number is 0.60. The native evaluator reproduces a result whose uncertainty includes that number when one action is executed, and a lower result when ten are executed. Executing the whole chunk is worse than either. None of these runs is the leaderboard's original job, and MuJoCo 3.8.1 was not swapped for the version that job used.

Raw outputs are under `runs/pretrained_baseline/`, which is gitignored. The harness is `scripts/pretrained_baseline_validation.py`.
