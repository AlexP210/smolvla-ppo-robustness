# Project plan: SmolVLA robustness through PPO post-training

The working branch is `flow-rl`. It is the git history of `smolvla_flow_rl`, not a merge with LeRobot. Tag `flow-rl-baseline` is the unmodified upstream commit `c161af2100610cf9f411bce23ad57180eb771d44`. The migration commit on top of that tag adds this file, the project README, `AGENTS.md`, and `.github/workflows/ci.yml`. It does not change the trainer. No LIBERO episode and no PPO update has been run on this branch. A passing syntax gate is not evidence that training works.

The LeRobot 0.7.1 history is `archive/lerobot-0.7.1` at `462e4571e1bf7a37ef15b4594746c6600e8a3621`. Tag `original-baseline` is `b07adcc5e3ce42c26a248236dbb3571f84c28c43`. Sections 1–6 describe that archived tree. They are not a map of the files in this checkout.

Remotes: `origin` is `git@github.com:AlexP210/smolvla-ppo-robustness.git`. `upstream` is `https://github.com/mturan33/smolvla_flow_rl`. `upstream-lerobot` is `https://github.com/huggingface/lerobot.git`. GitHub's default branch is still `main` until an admin changes it. See section 7.4.

Status labels used below:

| Label | Meaning |
| --- | --- |
| Implemented and covered by an in-tree test | The behavior is in code and an upstream test asserts it. Those tests are not part of `.github/workflows/ci.yml` and were not re-run here. |
| Implemented, not exercised here | The behavior is in code. The tests inspected for this plan do not run that path, or they skip the simulator. |
| Requires modification | An existing function is the integration point, but it does not do this job as written. |
| Requires new implementation | No class or function in this tree performs this step. |

## 1. There is no SmolVLA PPO implementation

The headings in sections 1–6 refer to the archived LeRobot 0.7.1 tree (`archive/lerobot-0.7.1`). This checkout is the `smolvla_flow_rl` trainer. Its PPO code is `src/flow_sde.py`, `src/ppo_step.py`, `src/gae.py`, `src/value_head.py`, and `src/train_flow_rl.py`. That code has not been executed here.

`SmolVLAPolicy` (`src/lerobot/policies/smolvla/modeling_smolvla.py`) is a flow-matching policy. `VLAFlowMatching.forward` returns an MSE loss between the flow target `u_t` and the predicted vector field `v_t`. `VLAFlowMatching.sample_actions` integrates that field with `euler_integrate` for `config.num_steps` (10 in the class default and in the checkpoints below). Neither method returns a log-probability, a value, or an advantage.

The only registered RL algorithm is SAC:

- `get_algorithm_class` in `src/lerobot/rl/algorithms/factory.py` accepts `"sac"` and raises otherwise.
- `SACAlgorithm` (`src/lerobot/rl/algorithms/sac/sac_algorithm.py`) calls `self.policy.actor(...)`, which must return actions and `log_probs`.
- That actor is `GaussianActorPolicy` (`src/lerobot/policies/gaussian_actor/modeling_gaussian_actor.py`). `SACAlgorithmConfig.from_policy_config` is typed for `GaussianActorConfig`.
- `GaussianActorPolicy` raises if asked to chunk actions. SmolVLA's default `chunk_size` and `n_action_steps` are 50.

`src/lerobot/rl/actor.py` (`act_with_policy`) and `src/lerobot/rl/gym_manipulator.py` (`RobotEnv`, `make_robot_env`) roll out that SAC stack on a real robot. `HILSerlRobotEnvConfig.sim_settings` raises `NotImplementedError` for `make_env` and `lerobot-eval`. TD-MPC (`src/lerobot/policies/tdmpc/modeling_tdmpc.py`) computes an advantage for its own latent model. It is not wired to SmolVLA or LIBERO.

| Component | Status |
| --- | --- |
| SmolVLA flow-matching loss and chunked action sampling | Implemented, not exercised here |
| `SmolVLAPolicy.prepare_images` uint8 to `[-1, 1]` | Implemented and covered by `tests/policies/smolvla/test_modeling_smolvla.py` |
| SAC actor-learner, replay buffer, critic | Implemented, not exercised here. Incompatible interface with SmolVLA |
| PPO clip objective, GAE, value baseline for SmolVLA | Requires new implementation |
| Stochastic log-probability of a SmolVLA action chunk | Requires new implementation |

## 2. What the existing training and evaluation paths actually do

These are two separate pipelines. They do not form one PPO loop.

### Supervised training

`lerobot_train.train` (`src/lerobot/scripts/lerobot_train.py`) loads a dataset and calls `update_policy`. `update_policy` runs `policy(batch)` and steps AdamW. For SmolVLA that call is `SmolVLAPolicy.forward`, which is the flow-matching MSE. `reduction="none"` exists for sample weighting. It is not an advantage.

`make_smolvla_pre_post_processors` (`src/lerobot/policies/smolvla/processor_smolvla.py`) renames observations, normalizes with dataset stats, appends a newline to the task string, tokenizes with `config.vlm_model_name`, and moves tensors to the device. The postprocessor unnormalizes actions.

Default `SmolVLAConfig` freezes the vision encoder and sets `train_expert_only=True` and `load_vlm_weights=False`. A checkpoint's own `config.json` replaces those defaults on `from_pretrained`. The LIBERO checkpoints set both freeze flags to false (section 3).

### LIBERO environment

`make_env` (`src/lerobot/envs/factory.py`) builds from `EnvConfig.sim_settings()`. `LiberoEnvConfig` (`src/lerobot/envs/configs.py`) registers `"libero"` and passes `is_libero_plus`, cameras, `init_states`, `control_mode`, and image size into `env_fns` (`src/lerobot/sim/integrations/libero.py`).

`LiberoEnv` defers `OffScreenRenderEnv` to the first `reset` via `_ensure_env`. `reset`:

1. Calls the inner `reset` and `seed`.
2. If `init_states` is true, calls `set_init_state` on a row of `get_task_init_states` and advances `init_state_id` by `n_envs`.
3. Steps `num_steps_wait` (default 10) with `get_libero_dummy_action()` (`[0, 0, 0, 0, 0, 0, -1]`).
4. Sets `robot.controller.use_delta` from `control_mode` (`"relative"` or `"absolute"`).

`step` forwards `action` to the inner env, then sets `info["is_success"]` from `self._env.check_success()`. `terminated` is the inner `done` or that success flag. The float `reward` is whatever `OffScreenRenderEnv.step` returns. This repository does not define that reward.

`canonical_observation` maps cameras through `camera_name_mapping` (default `agentview_image` to `image`, `robot0_eye_in_hand_image` to `image2`) and builds an 8-vector state: end-effector position (3), axis-angle (3), gripper qpos (2). `LiberoProcessorStep` (`src/lerobot/processor/env_processor.py`) does the same conversion on batched tensors and flips images on height and width. `tests/sim/integrations/test_libero.py` checks the numpy adapter against `LiberoProcessorStep` and checks `conventions()` without starting MuJoCo.

Action space in this file is `Box(-1, 1, shape=(7,))` (`ACTION_DIM = 7`). Suite horizons are `TASK_SUITE_MAX_STEPS`: spatial and object 280, goal 300, `libero_10` 520, `libero_90` 400. `LiberoEnvConfig` defaults to `observation_height = observation_width = 360` and `fps = 20`.

`tests/sim/integrations/test_libero.py` never constructs `OffScreenRenderEnv`. Creating a real LIBERO env is implemented and not covered by that test.

### Evaluation rollout

`rollout` in `src/lerobot/scripts/lerobot_eval.py`:

1. `policy.reset()`, which clears SmolVLA's action deque (`SmolVLAPolicy.reset`).
2. `env.reset`.
3. Each step: `prepare_canonical_observation` when the vector env sets `canonical_observations`, else `preprocess_observation`.
4. Copies `task_description` into `observation["task"]` (`LiberoEnv.task_description` is `task.language`).
5. Runs `env_preprocessor`, then the policy preprocessor (rename, normalize, tokenize).
6. Under `torch.inference_mode()`, calls `policy.select_action`. SmolVLA refills a chunk only when the deque is empty (`_check_get_actions_condition`) and pops one action. RTC is rejected inside `select_action`.
7. Postprocesses, steps the vector env, and records `reward`, `success` from `info["final_info"]["is_success"]`, and `done`.

`eval_policy` aggregates `pc_success` and `pc_success_ci95`. `return_observations` defaults to false. The returned dict has no log-probability and no value.

`select_action` is decorated with `@torch.no_grad()`. A PPO update cannot backpropagate through the action that this method returns.

### Where a PPO loop would have to be built

| Stage | In this tree | Status |
| --- | --- | --- |
| LIBERO construction | `LiberoEnv`, `env_fns`, `LiberoEnvConfig` | Implemented, not exercised here (no simulator in the unit test) |
| Language-conditioned action | `select_action` → `sample_actions` | Implemented, not exercised here |
| Rollout traces of reward and success | `lerobot_eval.rollout` | Implemented, not exercised here. Not a training buffer |
| Online transition transport | `act_with_policy`, `ReplayBuffer` | Implemented for SAC on `RobotEnv`. Requires modification to target LIBERO, and the policy type still does not match |
| Reward used for learning | Inner LIBERO `step` return | Not defined in this repository. Success flag is separate |
| Advantage / GAE | — | Requires new implementation |
| PPO parameter update | — | Requires new implementation |

## 3. Pretrained checkpoints

Hub configs were read for this plan. Weights were not loaded into `SmolVLAPolicy`.

`SmolVLMWithExpertModel.__init__` (`src/lerobot/policies/smolvla/smolvlm_with_expert.py`), when `load_vlm_weights` is true, downloads `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` in bfloat16 and always downloads its processor. Checkpoint `model.safetensors` is applied afterward by `PreTrainedPolicy.from_pretrained`. Construction therefore needs that VLM repo even when the policy repo already stores a fine-tuned state dict. The class then truncates the text stack to `num_vlm_layers` (16 in `SmolVLAConfig`).

| Hub repo | Hub sha prefix | Named by | State | Cameras | Action | `train_expert_only` / `freeze_vision_encoder` |
| --- | --- | --- | --- | --- | --- | --- |
| `lerobot/smolvla_base` | `5e8d12a6e297` | Tests such as `tests/test_yaml_policy_path.py` | `[6]` | `camera1`, `camera2`, `camera3`, each `[3, 256, 256]` | `[6]` | true / true |
| `lerobot/smolvla_libero` | `a24f1f380461` | `benchmarks/benchmarks.yaml` `libero.revision`; `scripts/ci/benchmark_matrix.py` default `lerobot/smolvla_{name}` | `[8]` | `camera1`, `camera2` | `[7]` | false / false |
| `lerobot/smolvla_libero_plus` | `bb424c8a430f` | `benchmarks/benchmarks.yaml` `libero_plus.revision` | `[8]` | `camera1`, `camera2`, plus `empty_camera_0` and `empty_camera_1` | `[7]` | false / false |

`benchmarks/benchmarks.yaml` maps LIBERO's `observation.images.image` and `observation.images.image2` to `camera1` and `camera2`. `lerobot_eval.eval_main` passes `cfg.rename_map` into the preprocessor. Without that map, `rollout/context.py` raises `Visual feature mismatch` when camera names differ.

`lerobot/smolvla_base` is not a LIBERO policy: 6-D state, 6-D action, three cameras. `LiberoEnv` steps a 7-vector. The nominal and perturbation-aware runs should load `lerobot/smolvla_libero`.

`lerobot/smolvla_libero_plus` sets `empty_cameras` to 2. `SmolVLAConfig.validate_features` inserts empty visual slots, and `prepare_images` masks cameras that are absent up to `empty_cameras`. The two real LIBERO views still need the rename map. The saved config leaves the VLM trainable.

`resize_imgs_with_padding` is `[512, 512]` on all three checkpoints, so a 360×360 LIBERO frame is resized inside `prepare_images`. That removes a hard shape conflict. It does not make a 360×360 render the same distribution as the 256×256 features stored in the checkpoint config.

## 4. Where perturbations can enter

Nothing in `LiberoEnv` takes a color, lighting, texture, dynamics, or object-offset argument.

### Visual, on the image the policy sees

`ImageTransformsConfig` (`src/lerobot/transforms/transforms.py`) defaults to `enable=False`. When enabled, the preset set is brightness, contrast, saturation, hue, sharpness, and a small affine. `ImageAugmentationProcessorStep` (`src/lerobot/processor/image_augmentation_processor.py`) applies `BatchedImageTransforms` to `observation.images.*` on a training batch. `lerobot-train` can place that step before the policy preprocessor. `rollout` does not.

That path changes pixels after rendering. It does not change LIBERO lighting or textures, and the affine is a 2D image warp, not a camera move in MuJoCo. Wiring it into `rollout` is a modification of the eval loop, or a new wrapper around the observation dict before `preprocessor`. A training-only augmentation must not be reused unchanged as an evaluation perturbation: `enable` is a training flag, and the draws are random unless a generator seed is set on `ImageAugmentationProcessorStep`.

### Spatial, initial object position

`LiberoEnv.reset` already cycles stored MuJoCo states from `get_task_init_states`. Those states are benchmark initial conditions, not a knob for "move the object by N centimeters." Offsetting an object requires either a new state edit after `set_init_state`, or a LIBERO-plus task whose layout is already baked into the BDDL and init file. The stored-state cycle is implemented. A controlled position shift is new.

### Physical

No mass, friction, or gravity field exists on `LiberoEnv` or `LiberoEnvConfig`. A physical perturbation is new, and it has to happen on the inner simulator after `_ensure_env`.

### Seen versus unseen

No split of perturbation parameters exists. `tests/robustness/` currently only checks that `perturbations`, `rollouts`, `config`, and `evaluation` directories exist (`tests/robustness/test_suite_layout.py`).

## 5. LIBERO-Plus

`LiberoPlusEnvConfig` subclasses `LiberoEnvConfig`, registers `"libero_plus"`, and sets `is_libero_plus=True`. Its docstring lists seven external dimensions: camera, layout, robot initial state, language, light, background, and sensor noise. The Python in this repo does not generate those dimensions.

`src/lerobot/sim/integrations/libero_plus.py` only calls LIBERO's `env_fns` with `is_libero_plus=True`. `get_task_init_states` then:

- loads `libero_newobj` files when the name contains `_add_` or `_level`, and reshapes them to `(1, -1)`;
- strips `_(language|view|light)_…` and `_(table|tb)_<digits>` from other filenames and loads the base `.pruned_init`.

So a view, light, or language variant reuses the base init file. The scene difference has to come from the external LIBERO-plus package's task definition, not from this function. `LiberoPlusEnvConfig` says that package is a git fork on `PYTHONPATH`, installed from the `libero_plus_native` stage of `docker/Dockerfile.sim`, and is not a `pyproject.toml` extra. `libero` is an extra (`hf-libero`, Linux only). `libero_plus` is not.

`benchmarks/benchmarks.yaml` already records a smoke entry: task `libero_spatial`, policy revision `bb424c8a430f`, and the same camera rename as LIBERO. The runner is `scripts/ci/benchmark_matrix.py` plus the archived workflow `.github/upstream-workflows/benchmark_run.yml`. That workflow is not executed by this project's CI. Reuse for the course is the in-tree env class, the rename map, and the pinned checkpoint. Reuse is not "turn the archived benchmark workflow back on."

`smolvla_libero_plus` is a policy trained under that benchmark. Evaluating it is a comparison point. It is not a substitute for training our own PPO policies.

## 6. Dependencies, memory, and a 12–24 GB GPU

Required to load SmolVLA and step LIBERO, from the code and `pyproject.toml`:

- Python `>=3.12`.
- Extra `smolvla`: `transformers` in the pinned range, `num2words`, and `accelerate`.
- Extra `libero`: `hf-libero` on Linux, `scipy`, and the `dataset` extra (video stack). LIBERO-Plus is additional and outside that extra.
- `MUJOCO_GL=egl` is the headless setting documented for this env; rendering is MuJoCo, not the policy GPU.
- First construction with `load_vlm_weights=True` downloads `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` and its processor, then the policy `model.safetensors`.

This analysis did not measure VRAM. The structure that dominates memory is visible in the config:

- VLM is the SmolVLM2-500M class, cut to 16 text layers, plus an action expert at `expert_width_multiplier=0.75`.
- Two images are resized to 512×512 (`prepare_images`).
- Inference denoises a chunk of 50 for `num_steps=10`, with `use_cache=True` so the prefix KV is computed once per chunk.
- `select_action` holds no gradient.
- `lerobot/smolvla_libero` sets `train_expert_only=False` and `freeze_vision_encoder=False`. A backward pass through `forward` or through the 10 denoising steps therefore includes the VLM unless those flags are changed after load.

Inference of one LIBERO environment (`eval.batch_size=1`, `torch.inference_mode`) is the feasible first use of a 12 GB or 24 GB GPU. It was not timed here.

A PPO update that backpropagates through the unfrozen 500M VLM and a 50-step chunk at 512×512 should not be planned for 12 GB on the archived 0.7.1 tree. That is not the trainer in this checkout. The adopted trainer stores sampling trajectories on the host and exposes explicit freeze regimes in `src/model_setup.py`. Its 12 GB claim is still unmeasured.

Other bottlenecks that are in the code, independent of that measurement:

- `rollout` runs every sub-environment until the slowest one finishes, and LIBERO horizons are 280–520 control steps.
- `return_observations=True` keeps every frame. Leave it false in the eval path. A PPO buffer should store what the update needs, not the whole rendered episode twice.
- `LiberoEnvConfig` at 360×360 is larger than the 256×256 feature shape stored on the LIBERO checkpoint. Resize still happens, but the render cost is higher.
- Several vector environments multiply MuJoCo processes. Start at one environment.
- `ImageTransformsConfig` comments already warn that unbounded GPU augmentations at 480×640 hold multiple float copies of the batch. Perturbation code should apply one seeded transform to the rendered frame, not the training stack's `max_num_transforms=3` random subset, until memory is measured.

## 7. External trainers, and the starting point

The reading below was done on clones outside the LeRobot checkout, before this branch was created. This branch now *is* the `smolvla_flow_rl` history at `c161af2100610cf9f411bce23ad57180eb771d44`, plus the migration commit. FlowVLA-RL and RLinf were not merged in. Claims below are from those files, not from READMEs alone, except where a README is quoted as a claim the code does not support.

### 7.1 What each repository actually implements

| Question | `mturan33/smolvla_flow_rl` | `BlackMirean/FlowVLA-RL` | `RLinf/RLinf` |
| --- | --- | --- | --- |
| Algorithm | PPO on a flow-matching SmolVLA policy | GRPO, critic-free, on SmolVLA | PPO and flow-SDE sampling for several VLAs. A search of `rlinf/` finds no `smolvla` symbol. |
| License | Apache-2.0. `NOTICE` names RLinf commit `832db3f5c7aa6c61d72f20aaf43b7ce0f4ad2b03` and LeRobot 0.4.4 as upstreams of `src/flow_sde.py`, `src/value_head.py`, `src/ppo_step.py`, and `src/action_model.py`. | MIT, copyright Teng Zhu, 2026. | Apache-2.0. |
| LeRobot pin | `requirements.txt`: `lerobot[smolvla]>=0.4.4,<0.5` and `hf-libero>=0.1.3,<0.2`. | `requirements-pinned.md`: LeRobot **v0.6.0**, and the README says HEAD is incompatible. | Not a LeRobot tree. Embodiment code vendors its own policies. |
| Checkpoint | Default warning path and `scripts/download_base_policy.py` use `HuggingFaceVLA/smolvla_libero`. Hub `config.json` for that id declares `observation.images.image` and `observation.images.image2`, state, and action shape `[7]`. | Pins `lerobot/smolvla_base` revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205` and trains its own SFT. The report compares against `lerobot/smolvla_libero`. | OpenPI, GR00T, LingBot, and related configs. Not the LIBERO SmolVLA checkpoint. |
| Stochastic sampling | `src/flow_sde.py` `sde_sample`: one randomly chosen integration step is Gaussian (`flow_sde`); the rest are the constant-step Euler update (`flow_ode`). Eval sets every step deterministic. | `src/flowvla_rl/sde.py` and `smolvla_sde.py`: Gaussian transition at denoising steps, with `gaussian_log_prob`. | `rlinf/models/embodiment/openpi/sampling/rl_sampler.py` implements `flow_ode`, `flow_sde`, `flow_cps`, and `flow_noise`. |
| Log-probability | `get_logprob_norm` is an elementwise Gaussian density. `sde_sample` slices to `[:, :, :action_chunk, :action_env_dim]` and keeps only the stochastic step (`denoise_inds`). `recompute_logprob_value` rescores the stored chain at that same index. | `SmolVLATrajectory.log_probs` sums element log-densities under a mask. `recompute_log_probs` rescores a stored trajectory. | The OpenPI sampler returns a transition log-density for the chosen noise method. |
| Rollout buffer | `src/rollout.py` `collect_rollout` stores, per decision, `env_obs`, `chains`, `denoise_inds`, `prev_logprobs`, `values`, rewards, dones, and a bootstrap value. Chains stay on CPU. | Group rollouts of 4 episodes from one initial scene (`libero_rollout.py`). | Ray workers and per-policy rollout workers. A framework, not a single buffer class we can lift. |
| Executed horizon | `ppo_step_stored` sums log-probabilities over `[:, :n_action_steps, :action_env_dim]` only. `collect_rollout` executes `n_action_steps` actions from one plan and folds reward and done with a max, so the stored row stays one per decision. `src/horizon_gate.py` refuses a run whose training horizon and eval horizon disagree. | Chunk masking is in `SmolVLATrajectory.masked_log_probs`. Suite step caps are hardcoded (object/spatial 280, goal 300). | OpenPI configs (the pin cited by `smolvla_flow_rl`) use a LIBERO horizon of 480 for both train and eval. |
| Critic | `src/value_head.py` `ValueHead`: MLP on the features entering `action_out_proj`, captured by a forward pre-hook. Optional LayerNorm and PopArt buffers. The head is a separate module. | None. Advantages are the group mean (and optional group standard deviation) of episode returns. | `rlinf/models/embodiment/modules/value_head.py` is the file `smolvla_flow_rl` says it adapted. Other value models exist for other policies. |
| Reward and GAE | `terminal_success` in `src/env_libero.py` is 1 on success, else 0. No shaping. `src/gae.py` `compute_gae` is the standard δ recursion with `1 - done` cutting the bootstrap, plus a caller-supplied bootstrap for a segment that ends mid-episode. Advantages are standardized over the pooled batch with population std. | Binary success, one scalar per episode, then group-relative advantages. No GAE. | RLinf's own advantage utilities. Not inspected line by line beyond the OpenPI loss names cited below. |
| PPO objective | `src/reference_ops.py` `pg_loss_reference`: clipped surrogate plus dual clip at `clip_ratio_c=3`. Value loss is clipped Huber (`value_clip=0.2`, `huber_delta=10`). `ppo_step_stored` adds `value_coef * v_loss` and an optional KL to a frozen reference. No entropy term, because the noise scale is not a parameter. Gradient accumulation divides the loss by the number of micro-batches. Non-finite steps are skipped. | `grpo_loss` in `src/flowvla_rl/grpo.py` weights the log-probability by the group advantage. It is not the clipped PPO surrogate. | `rlinf/algorithms/losses.py` `compute_ppo_actor_loss` is the reference `smolvla_flow_rl` quotes. |
| Trainable pieces | `src/model_setup.py` regimes: `expert_full_vlm_lora` (recommended in the README: full-rank action expert, LoRA on the language backbone, action-head projections open), `full`, `lora`, `frozen`. The README's own parameter counts are 96.70M (`full`) and larger for the LoRA-on-VLM regime. `frozen` and `lora` do not train the action-head projections. | Trains the action expert. The frozen SFT policy is the KL reference. | Per-policy configs. Adopting this means taking the framework's worker and config system. |
| Memory | The README states a 12 GB RTX 5070 Ti laptop run in bf16. This analysis did not reproduce that measurement. The code that makes the claim plausible is real: bf16 autocast in `collect_rollout`, prefix KV computed once per chunk, and chains stored on CPU. | README states peak 20.4 GiB on a 32 GB RTX 5090, and that rollouts are CPU-bound. Not remeasured here. | Distributed workers. Not a 12 GB single-process recipe. |
| Evaluation | `src/evaluate.py` scores deterministic (`mode="eval"`) episodes and can reread one log at several budgets that do not exceed the horizon. `eval_cli.py` checks that the eval flags match the recorded training run. | `configs/eval_protocol.md` and `REPORT.md`: ODE sampling, 20 episodes per task, eval seed 1000. | Example YAMLs per embodiment. |
| Tests and results | The clone contains `tests/test_invariants.py` only. `reference_ops.py` names `tests/test_reference_parity.py`, which is not in this tree. The README says no success rates are published. Do not cite a number from this repository. | `tests/` exists and the README links a passing Actions badge. Reported LIBERO-Object change 58.5% to 64.8% is a README/REPORT claim. The loss code is GRPO, and the report itself says the 300-update budget was chosen using a seed that is inside the mean. Treat those percentages as their result under their protocol, not as a number this project has reproduced. | Large test tree for the framework. No SmolVLA LIBERO number to adopt. |

Two facts ruled out porting this trainer into the archived LeRobot 0.7.1 tree:

1. `src/env_libero.py` imports `from lerobot.envs.configs import LiberoEnv` and `from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, _make_env_fns`. On `archive/lerobot-0.7.1` the class is `LiberoEnvConfig` in `lerobot.envs.configs`, and the factory is `lerobot.sim.integrations.libero._make_env_fns`. FlowVLA-RL's `libero_rollout.py` imports `from lerobot.envs.libero import LiberoEnv`, the same missing module.
2. `src/action_model.py` calls `vlm_with_expert.forward(..., fill_kv_cache=True)`. On `archive/lerobot-0.7.1`, `SmolVLMWithExpertModel.forward` does not take `fill_kv_cache`. That call matches LeRobot 0.4.4, which is the pin in `requirements.txt`, and it does not match 0.7.1.

`HuggingFaceVLA/smolvla_libero` and `lerobot/smolvla_libero` are different checkpoints. Both have action shape `[7]`. The first declares `observation.images.image` and `observation.images.image2`, which is what `REQUIRED_IMAGE_KEYS` in `src/action_model.py` demands. The second declares `camera1` and `camera2`, which is what `benchmarks/benchmarks.yaml` on `archive/lerobot-0.7.1` renames. This trainer refuses the second id.

### 7.2 Decision

**Option C is adopted.** This branch is the `smolvla_flow_rl` history. It is not a vendor of those files inside LeRobot 0.7.1, and it is not a dependency of that environment.

- **A, integrate into LeRobot 0.7.1**, was a port. The PPO math does not import LeRobot, but `action_model.py` and `env_libero.py` call LeRobot 0.4.4 private APIs that 0.7.1 renamed or removed.
- **B, external dependency**, cannot share an environment with LeRobot 0.7.1. The pin is `lerobot<0.5`.
- **C** keeps the upstream sampling, executed-chunk mask, GAE, and PPO step. Robustness work, when it starts, goes on `adapt_observation` / `collect_rollout`. `NOTICE` and the RLinf and LeRobot copyright lines stay with those files. They were not rewritten in the migration.

What this adoption does not buy:

- No published success rate. The first training run is still an experiment. It has not been started.
- No visual or spatial perturbation API. Seen/unseen robustness is still new code.
- No LIBERO-Plus. `LiberoPlusEnv` exists only on `archive/lerobot-0.7.1`.
- The upstream README's 12 GB claim was not measured here.
- The checkpoint id is `HuggingFaceVLA/smolvla_libero` (keys `image` / `image2`). `lerobot/smolvla_libero` is a different Hub repo and is refused by `src/action_model.py`.

FlowVLA-RL is GRPO, not PPO, and pins LeRobot 0.6.0. RLinf is the mathematical upstream named in `NOTICE` and does not contain a SmolVLA trainer. Neither was merged.

### 7.3 Branches

Branch A is the migration commit on `flow-rl`. Do not open B–F until a later task asks for them. None of them edits the PPO files to satisfy CI. CPU tests that land later must not import MuJoCo or a checkpoint. The active CI runs `scripts/check_syntax.py` and `scripts/check_import_closure.py` only. `tests/test_invariants.py` needs LeRobot and LIBERO and is outside that job.

### Branch A — migration onto `flow-rl` (done in the migration commit)

`flow-rl` is `smolvla_flow_rl` `main` at `c161af2100610cf9f411bce23ad57180eb771d44`, tagged `flow-rl-baseline`, plus documentation and CI. `NOTICE` is the upstream file. `requirements.txt` still pins `lerobot[smolvla]>=0.4.4,<0.5`. The LeRobot 0.7.1 history is `archive/lerobot-0.7.1`, not a parent of this branch.

Acceptance for this commit: `git rev-parse flow-rl-baseline^{}` equals `upstream/main` at the time of the tag; `git diff flow-rl-baseline -- src NOTICE LICENSE requirements.txt environment.yml` is empty; `archive/lerobot-0.7.1` still points at `462e4571e1bf7a37ef15b4594746c6600e8a3621`.

### Branch B — `feat/checkpoint-and-horizon`

Depends on A.

Responsibilities: pin `HuggingFaceVLA/smolvla_libero` and the suite horizon in one config. Training, in-training eval, and the final eval read that horizon through `horizon_gate.assert_single_horizon`. Document that `lerobot/smolvla_libero` is a different checkpoint and is not loaded by this trainer.

Acceptance: a unit test of the gate rejects disagreeing horizons; a dry import of the harness check rejects a config whose image keys are `camera1`/`camera2`. No GPU.

### Branch C — `feat/nominal-ppo-smoke`

Depends on B. First GPU run. Not CI.

Responsibilities: one task, one environment, bf16, the README's `expert_full_vlm_lora` regime or the `frozen` regime if memory fails, a short segment, one PPO update via `ppo_step_stored`. Record peak GPU memory, whether the step was skipped as non-finite, and that eval sampling uses `mode="eval"`.

Acceptance: the process finishes one update; the log contains `pg_loss`, `v_loss`, `clip_frac`, and `step_skipped`; peak memory is written into the run notes. No success-rate claim from a handful of steps.

### Branch D — `feat/perturbation-spec`

Depends on A, which is done. Not started. Do not put this under `src/lerobot/robustness/`; that path exists only on the archived branch.

Responsibilities: seen and unseen specs for color jitter and for an initial-position delta. Identity spec is a no-op. The position delta is data only.

Acceptance: a fixed seed reproduces the jittered tensor; a spec cannot be both seen and unseen; the position applicator is not imported from any LIBERO module. CPU only.

### Branch E — `feat/perturbation-in-rollout`

Depends on A, C, and D.

Responsibilities: apply the seen visual spec inside the adopted `collect_rollout` path, after the simulator returns the frame and before `adapt_observation`. Do not apply the unseen spec during training. Leave `terminal_success` and `compute_gae` unchanged so the reward definition stays the one in their `env_libero.py`.

Acceptance: a fake observation dict changes under the seen spec and does not change under the unseen spec; the stored `chains` row still lines up with one decision, not with each executed action.

### Branch F — `feat/robustness-eval`

Depends on B and E.

Responsibilities: three eval passes on the same checkpoint, all deterministic: nominal, seen, unseen. Same horizon as training. Report success counts and the number of episodes. Do not compare against a number from FlowVLA-RL or from `smolvla_flow_rl`, because the first has no PPO and the second publishes none.

Acceptance: the eval log records the checkpoint id, horizon, `n_action_steps`, and which spec was applied. A budget larger than the horizon is rejected by their existing `parse_budgets`.

### Dependencies that stay pinned

| Item | Pin |
| --- | --- |
| Training code | `flow-rl-baseline` `c161af2100610cf9f411bce23ad57180eb771d44`, Apache-2.0 plus `NOTICE` |
| LeRobot | `lerobot[smolvla]>=0.4.4,<0.5` in the conda env `smolvla-flow-rl`, not in the env `lerobot` |
| LIBERO package | `hf-libero>=0.1.3,<0.2` |
| Policy weights | `HuggingFaceVLA/smolvla_libero` |
| Archived eval path | `lerobot/smolvla_libero` and LIBERO-Plus remain on `archive/lerobot-0.7.1` only |

### 7.4 Default branch, not done here

`origin/main` stays the LeRobot 0.7.1 history. Making `flow-rl` the default branch is a GitHub setting: Settings → General → Default branch → `flow-rl`. That does not require a force-push, and it leaves `main` in place. `archive/lerobot-0.7.1` is the same commit as `main`, so the archive remains if an admin later renames `main`.

Renaming `flow-rl` to `main` while `main` still exists, or deleting `main`, rewrites or removes the existing branch. That needs an explicit team decision. It was not done. Do not force-push `main` to this history.

## 8. Superseded plan

The previous branch list (experiment config, visual perturbations, spatial interface, rollout record, nominal eval, a from-scratch PPO update, a LIBERO loop, perturbation-aware PPO, a report, and LIBERO-Plus eval, all under `src/lerobot/robustness/` on `archive/lerobot-0.7.1`) is withdrawn. It assumed the policy gradient had to be written on that tree. LIBERO-Plus and the spatial applicator remain future work on this trainer, not branches on `LiberoEnv` in the archive.

## 9. Parallel work and conflicts

Do not open research branches in this migration. After it, B and D can proceed together. C is the first GPU job and should be the only edit to `src/train_flow_rl.py` and `src/model_setup.py` in that task. E edits `src/rollout.py` and should not start while C is changing that loop. F waits for E.

Do not edit `src/flow_sde.py`, `src/gae.py`, `src/ppo_step.py`, or `src/reference_ops.py` to add perturbations or to satisfy a formatter. Perturbations enter only at the observation, before the policy preprocessor. `lerobot/smolvla_base` is the wrong initialization: `scripts/download_base_policy.py` says its cameras and widths do not match this harness.

The archived branch still contains `modeling_smolvla.py` and the SAC factory. Do not register `"ppo"` there, and do not merge that branch into `flow-rl`.

## 10. Suggested next assignment

Not this commit. The next research task is branch B (checkpoint id and a single horizon), then branch C (one PPO update, with peak memory written down). Branch C is the measurement the 12 GB claim still needs. Do not treat that run as a success rate.
