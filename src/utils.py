"""Seeding and checkpoint helpers (seeding follows class `code/L14/helper_utils.py`)."""

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU, CUDA and MPS)."""
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds mps and cuda generators
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_deterministic() -> None:
    """Prefer deterministic kernels. warn_only=True because some MPS ops have no deterministic version."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_checkpoint(path: Path, **state) -> None:
    """Save models/optimizers/epoch with torch.save, writing to a temp file first so a crash can't corrupt it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> dict:
    """Load a checkpoint written by save_checkpoint."""
    return torch.load(Path(path), map_location=device, weights_only=False)
