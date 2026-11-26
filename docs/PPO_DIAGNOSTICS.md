# PPO diagnostics

The 20-update pilot log stored the last optimizer step of each cycle. Advantage statistics, the reward counts inside a window, stored values, returns, explained variance, and an approximate KL of the probability ratio were not in that file. This page is the logging that records them. The pilot itself was not run again.

The objective is unchanged. `pg_loss_reference`, the value loss, the optional reference penalty, and the optimizer step are the same calls as before. The new numbers are means of detached tensors. No generator is drawn.

## What one cycle record contains

`metrics.jsonl` still has the last step's fields at the top level, including `ratio_mean`, `clip_frac`, `pg_loss`, `v_loss`, `grad_norm`, and `grad_norm_applied`. `grad_norm` is the norm before clipping. `grad_norm_applied` is the norm after it. A new `diagnostics` object sits beside those fields.

| Field | Source |
| --- | --- |
| `advantage_before_normalization`, `advantage_after_normalization` | The GAE tensor before and after `normalize_advantages`. Each has count, mean, population standard deviation, min, and max. |
| `reward` | Counts of positive, zero, and other decisions, the reward sum, and one sum per environment. A positive entry is one successful decision of the sparse reward. It is not a count of distinct episode identifiers. |
| `stored_values` | Value predictions saved at collection time. |
| `returns` | GAE value targets. |
| `explained_variance` | `1 - Var(returns - stored values) / Var(returns)`, population variances. `null` when the returns do not vary. This compares the values from the rollout with the returns, not the values recomputed inside the update. |
| `optimizer_steps` | Means and extremes over every optimizer step in the cycle. |

`optimizer_steps` includes `approx_kl_mean`, ratio mean, standard deviation, median, min, and max, `clip_frac_mean`, `pg_loss_mean`, `v_loss_mean`, and the gradient norm before and after clipping as both a mean and a maximum. The approximate KL is Schulman's estimator `mean((ratio - 1) - log(ratio))`, with `ratio = pi_new / pi_old`. It is not `kl_to_reference`. That field is still the optional anchor against a frozen copy, and it stays zero when `--kl_coef` is 0.

Non-finite step values are left out of the means. A skipped step is counted in `n_skipped`.

The line printed after the existing cycle line reports the success-decision count, the cycle-mean approximate KL, the mean clip fraction, and the explained variance. The existing cycle line is still the last step. The first-update ratio check still reads that step's `ratio_mean`.

## What this branch does not do

The summaries read tensors the update already has. They do not enter the loss, the backward pass, `optimizer.step`, or a random generator. `tests/test_ppo_diagnostics.py` checks the KL estimator, explained variance, the cycle mean across every step, and that those helpers leave the global generator where they found it. It also checks that the loss line in `src/ppo_step.py` is still `pg_loss + value_coef * v_loss + kl_coef * kl_penalty`. It does not call the optimizer and it does not import LIBERO.

The 20-update pilot log was not recomputed, and this page does not invent advantage vectors for it. Reward-and-advantage reporting over a stored rollout, a longer nominal training run, and a checked-in hyperparameter file are later branches. Perturbation-aware training is not part of this change.
