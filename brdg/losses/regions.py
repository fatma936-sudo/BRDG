"""Derive per-region supervision from the pixel-level ground truth."""

import torch
import torch.nn.functional as F

from brdg.config import IGNORE_INDEX


def _region_class_counts(targets_down: torch.Tensor, assignment_map: torch.Tensor,
                         num_classes: int):
    """(B, C, K) counts of each class inside each hard-assigned region."""
    B, HW, K = assignment_map.shape
    flat = targets_down.view(B, HW)
    valid = flat != IGNORE_INDEX
    safe = flat.masked_fill(~valid, 0)

    one_hot = F.one_hot(safe, num_classes=num_classes).float()
    one_hot = one_hot * valid.unsqueeze(-1)
    hard = assignment_map.argmax(dim=2)
    hard_one_hot = F.one_hot(hard, num_classes=K).float()
    return torch.bmm(one_hot.transpose(1, 2), hard_one_hot)


def get_superpixel_boundary_labels(targets_down: torch.Tensor, assignment_map: torch.Tensor,
                                   num_classes: int) -> torch.Tensor:
    """(B, K, 1) in {0, 1}: 1 where a region straddles more than one class."""
    counts = _region_class_counts(targets_down, assignment_map, num_classes)
    n_classes_per_sp = (counts > 0).float().sum(dim=1)
    return (n_classes_per_sp > 1).float().unsqueeze(-1)


def get_region_class_labels(targets_down: torch.Tensor, assignment_map: torch.Tensor,
                            num_classes: int) -> torch.Tensor:
    """(B, K): majority class inside each region."""
    return _region_class_counts(targets_down, assignment_map, num_classes).argmax(dim=1)
