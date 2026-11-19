from __future__ import annotations

import random
import numpy as np
import torch


def set_reproducibility(seed: int) -> None:
    """Seed Python, NumPy, Torch and available CUDA devices."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a seeded torch.Generator used for deterministic shuffling."""
    return torch.Generator().manual_seed(seed)
