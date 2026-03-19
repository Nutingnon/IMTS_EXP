from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class ValueNormStats:
    mean: Tensor
    std: Tensor


def _safe_std(std: Tensor) -> Tensor:
    return torch.where((std == 0) | torch.isnan(std), torch.ones_like(std), std)


def compute_standardization_stats_from_tensors(values: list[Tensor]) -> ValueNormStats:
    if len(values) == 0:
        raise ValueError("Cannot compute normalization stats from an empty value list.")

    concatenated = torch.cat(values, dim=0)
    mean = torch.nanmean(concatenated, dim=0)
    std = torch.sqrt(torch.nanmean((concatenated - mean) ** 2, dim=0))
    return ValueNormStats(mean=mean, std=_safe_std(std))


def apply_standardization(values: Tensor, stats: ValueNormStats) -> Tensor:
    return (values - stats.mean) / stats.std


def compute_standardization_stats_from_arrays(values: list[np.ndarray]) -> ValueNormStats:
    if len(values) == 0:
        raise ValueError("Cannot compute normalization stats from an empty value list.")

    concatenated = np.concatenate(values, axis=0)
    mean = np.nanmean(concatenated, axis=0)
    std = np.nanstd(concatenated, axis=0)
    std = np.where((std == 0) | np.isnan(std), 1.0, std)
    return ValueNormStats(
        mean=torch.as_tensor(mean, dtype=torch.float32),
        std=torch.as_tensor(std, dtype=torch.float32),
    )