# Copyright 2026 the course project authors.
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Seeds for evaluation, split into the simulator and the flow's starting noise.

The environment seed and the policy seed are different inputs. A shared counter is how an
evaluation that adds one more environment changes the initial state of every other
environment, and how a wider policy batch changes the Gaussian draw of the first row.

Each draw is a function of the seed, the episode id, and the decision. The row a sample
occupies in a batch is not part of that function. The same id therefore produces the same
reset seed and the same initial noise whether it is evaluated alone or beside other ids,
and whether those other ids are listed before it or after it.

The noise is standard normal, the same distribution `sample_noise` asks PyTorch for. It is
drawn on the CPU from a private generator so the global random stream, and the CUDA
generator, are not what decides the sample. The flow integration that consumes this noise
is unchanged.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

MAX_ENV_SEED = 2 ** 31 - 1
_ENV_DOMAIN = 1
_POLICY_DOMAIN = 2


def _seed_sequence(parts: Sequence[int]) -> np.random.SeedSequence:
    return np.random.SeedSequence([int(part) for part in parts])


def env_reset_seed(env_seed: int, episode_id: int, batch: int = 0) -> int:
    """Simulator reset seed for one episode. Independent of how many episodes share the batch."""
    words = _seed_sequence((env_seed, episode_id, batch, _ENV_DOMAIN)).generate_state(1)
    return int(words[0] % MAX_ENV_SEED)


def _policy_generator_seed(policy_seed: int, episode_id: int, decision: int, batch: int) -> int:
    words = _seed_sequence((policy_seed, episode_id, batch, decision, _POLICY_DOMAIN)).generate_state(2)
    return int(words[0]) | (int(words[1]) << 32)


def initial_flow_noise(policy_seed: int, episode_ids: Sequence[int], decision: int,
                       chunk: int, action_dim: int, *, batch: int = 0,
                       device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Standard normal noise, one independent draw per episode id.

    Shape is `(len(episode_ids), chunk, action_dim)`. Row `i` depends on
    `(policy_seed, episode_ids[i], batch, decision)` and not on `i` or on the other ids.
    """
    if chunk < 1 or action_dim < 1:
        raise ValueError("chunk and action_dim must be positive, got %r and %r" % (chunk, action_dim))
    rows = []
    for episode_id in episode_ids:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_policy_generator_seed(policy_seed, int(episode_id), int(decision), int(batch)))
        rows.append(torch.normal(mean=0.0, std=1.0, size=(chunk, action_dim),
                                  dtype=dtype, generator=generator))
    if not rows:
        noise = torch.empty((0, chunk, action_dim), dtype=dtype)
    else:
        noise = torch.stack(rows, dim=0)
    if device is not None:
        noise = noise.to(device)
    return noise


def resolve_episode_ids(episode_ids: Sequence[int] | None, n_envs: int) -> list[int]:
    """Slot identities for one vector batch. The default is the slot index, which the caller can replace."""
    if n_envs < 1:
        raise ValueError("n_envs must be positive, got %r" % (n_envs,))
    if episode_ids is None:
        return list(range(n_envs))
    ids = [int(episode_id) for episode_id in episode_ids]
    if len(ids) != n_envs:
        raise ValueError("episode_ids has length %d and the vector environment has %d workers"
                         % (len(ids), n_envs))
    return ids
