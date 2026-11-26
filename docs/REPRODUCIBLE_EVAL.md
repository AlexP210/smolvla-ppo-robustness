# Reproducible evaluation

The matched comparison in [NOMINAL_PPO_PILOT.md](./NOMINAL_PPO_PILOT.md) gave the initialization 2/8 successes and the 20-update checkpoint 0/8. Those two runs did not share a pinned sample of the flow's starting noise. This note is the mechanism that pins it. It does not replace that comparison, and it does not change the action execution horizon or the Euler step.

## Where the randomness is

Two generators are easy to conflate.

The simulator is seeded when the vector environment resets. Historically `evaluate` uses `seed + 7000 + batch * n_envs + slot`, and `set_initial_states` with `per_task=1` gives every slot in a batch the same `init_state_id`, because `slot % 1` is 0. The reset seed therefore changes when the batch gets wider, even for a slot whose task did not change. `build_env` also resets once, with `seed + slot`, before evaluation resets again. Training rollouts use `train_reset_seeds` in `src/seeding.py`, a different derivation, and only after an in-training evaluation.

The action sample starts from `torch.normal(0, 1)` of shape `(batch, chunk, max_action_dim)` inside `sample_actions`, on the global generator. A different batch shape is a different draw, including for the first row. `sde_sample` then integrates that noise. In `flow_ode` the integration scale is zero, so the Euler update does not add further noise. The starting noise is the random part. Evaluation mode does not make that draw a function of the episode.

`eval_cli` also seeds PyTorch, NumPy, and Python once, before the model is built. Adapter and value-head initialization consume that stream. Two evaluations that both forget to do this, which is what the seed-7000 probe did, do not even share a global stream.

## Separate seeds

`src/eval_noise.py` derives both values with `numpy.random.SeedSequence`. The environment domain and the policy domain use different tags, so the same integers do not land on both.

- `env_reset_seed(env_seed, episode_id, batch)` is the integer passed to `ve.reset`. It does not take the batch width.
- `initial_flow_noise(policy_seed, episode_ids, decision, chunk, action_dim, batch=...)` stacks one standard-normal row per episode id. Row `i` does not depend on `i` or on the other ids. The draw is float32, from a private CPU generator, which is the same distribution the policy's `sample_noise` requests.

`evaluate(..., env_seed=None, policy_seed=None, episode_ids=None)` leaves both unset by default, so the historical reset and the global noise draw stay what existing runs used. Set either seed and each worker uses `episode_ids`, defaulting to the slot index. The reset id is `episode_id % 50`. The noise for decision `t` is passed into `sample_actions`, and the integrator is still `sde_sample` with `flow_ode`.

The same `episode_id`, `batch`, and `decision` therefore match across a pretrained policy and a later checkpoint. That is the common-random-number pair. `src/eval_cli.py` accepts `--env_seed` and `--policy_seed`. Leaving either flag unset keeps that half of the historical path. `--action_steps` still defaults to 1, and the Euler step in `src/flow_sde.py` is not edited.

`scripts/matched_checkpoint_eval.py` is unchanged. It still calls `evaluate` without these seeds, so it still follows the historical reset and the global noise draw. The recorded 2/8 and 0/8 comparison was produced that way and was not recomputed. A reported paired comparison needs ten episodes per task and one shared scenario. That is a later feature, and eight episodes are not that comparison.

`episodes_per_task` still repeats the episode ids the caller passed. The batch index changes the reset seed and the noise tag, and the LIBERO initial-state id is `episode_id % 50`. A file of ten distinct initial states is the scenario feature, which is not in this branch.

## What the tests separate

`tests/test_eval_noise.py` is CPU-only.

- The same episode id produces the same reset seed, and a different seed, id, or batch ordinal does not.
- An episode id drawn alone matches that id's row inside a larger, reordered batch. The tensors are equal, so this is the sampling rule and not a tolerance.
- A few thousand elements stay near mean 0 and standard deviation 1.
- `sde_sample` in `flow_ode`, with a velocity that is a per-element scale, turns that shared noise into bitwise-equal actions for the solo row and the matching row of the batch. A neural forward can still disagree in the last bits when the batch width changes. That residual is numerical. It is not this noise draw, and this test does not call the network.
- The noise helper does not advance the global generator. `evaluate` still defaults `env_seed` and `policy_seed` to `None`, and the historical reset expression is still the branch taken when `env_seed` is unset.

Fixed scenarios, severity ladders, robustness curves, and the ten-episode paired comparison are not implemented here. Perturbations are not implemented here. In-training evaluation does not pass the new seeds, so a training run's own eval stays on the historical path until a later change opts in.
