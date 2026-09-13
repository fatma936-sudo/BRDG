"""
Multi-stage training schedule from the paper.

    Warmup (epochs 1-5)          encoder frozen, only the segmentation loss active
    Unfreeze & ramp (6-10)       encoder unfrozen at 0.1x LR, auxiliary weights ramp in
    Full (11+)                   all terms at their final weights

The assignment temperature tau anneals from start_tau to end_tau over the first
`anneal_epochs` epochs, sharpening the soft superpixel assignment over time.
"""

from typing import Tuple


def loss_weights_for_epoch(ep: int,
                           warmup_epochs: int = 5,
                           ramp_epochs: int = 5,
                           final_coarse_w: float = 0.3,
                           final_sp_bnd_w: float = 1.0,
                           final_compact_w: float = 1e-3,
                           final_contrastive_class_w: float = 0.05,
                           final_contrastive_boundary_w: float = 0.02,
                           final_gate_entropy_w: float = 1e-4) -> Tuple[float, ...]:
    if ep <= warmup_epochs:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    t = min(1.0, (ep - warmup_epochs) / max(1, ramp_epochs))
    return (final_coarse_w * t,
            final_sp_bnd_w * t,
            final_compact_w * t,
            final_contrastive_class_w * t,
            final_contrastive_boundary_w * t,
            final_gate_entropy_w * t)


def tau_for_epoch(ep: int, start_tau: float = 1.0, end_tau: float = 0.5,
                  anneal_epochs: int = 10) -> float:
    if ep <= anneal_epochs:
        a = (ep - 1) / max(1, anneal_epochs - 1)
        return start_tau * (1 - a) + end_tau * a
    return end_tau
