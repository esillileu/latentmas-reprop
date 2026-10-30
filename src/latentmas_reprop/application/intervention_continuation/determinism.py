"""Guard shared inference against consumption of process random generators."""

import random
from contextlib import contextmanager

import numpy as np
import torch


class StochasticForwardError(ValueError):
    """Shared forward consumed a process random generator."""


@contextmanager
def deterministic_forward(device: torch.device):
    """Reject common latent operations that consume the process RNGs."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    py = random.getstate()
    numpy_state = np.random.get_state()
    yield
    after = np.random.get_state()
    if (
        not torch.equal(cpu, torch.get_rng_state())
        or (
            cuda is not None and not torch.equal(cuda, torch.cuda.get_rng_state(device))
        )
        or py != random.getstate()
        or numpy_state[0] != after[0]
        or not np.array_equal(numpy_state[1], after[1])
        or numpy_state[2:] != after[2:]
    ):
        raise StochasticForwardError(
            "Stochastic forward cannot create or extend reusable state"
        )
