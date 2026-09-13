"""
Segmentation metrics.

Two mIoU/Dice conventions are provided:

* ``miou_score`` / ``dice_score`` average per image over all classes. This is the
  convention the published results use, and it is the default everywhere in this
  repository.
* ``update_confusion`` / ``metrics_from_confusion`` accumulate a dataset-level
  confusion matrix and average only over classes with non-zero ground-truth
  support in the split. Pass ``--dataset-level-metrics`` to train.py to report
  these alongside the defaults.

The two differ whenever a class is absent from an individual image, so choose one
and state which when reporting numbers.
"""

import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from brdg.config import IGNORE_INDEX


# --------------------------------------------------------------------------
# per-image metrics
# --------------------------------------------------------------------------
@torch.no_grad()
def miou_score(preds, targets, num_classes, return_per_class=False):
    miou = 0.0
    per = [] if return_per_class else None
    for c in range(num_classes):
        valid = targets != IGNORE_INDEX
        p = ((preds == c) & valid).float()
        t = (targets == c).float()
        inter = (p * t).sum(dim=(1, 2))
        union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2)) - inter
        iou = (inter + 1e-7) / (union + 1e-7)
        miou += iou
        if return_per_class:
            per.append(iou.mean().item())
    mean = (miou / num_classes).mean().item()
    return (per, mean) if return_per_class else mean


@torch.no_grad()
def dice_score(preds, targets, num_classes, return_per_class=False):
    ds = 0.0
    per = [] if return_per_class else None
    for c in range(num_classes):
        valid = targets != IGNORE_INDEX
        p = ((preds == c) & valid).float()
        t = (targets == c).float()
        inter = (p * t).sum(dim=(1, 2))
        union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        d = (2 * inter + 1e-7) / (union + 1e-7)
        ds += d
        if return_per_class:
            per.append(d.mean().item())
    mean = (ds / num_classes).mean().item()
    return (per, mean) if return_per_class else mean


# --------------------------------------------------------------------------
# dataset-level metrics
# --------------------------------------------------------------------------
@torch.no_grad()
def update_confusion(conf: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor,
                     num_classes: int) -> torch.Tensor:
    """Accumulate conf[ground_truth, prediction], dropping ignore pixels."""
    valid = targets != IGNORE_INDEX
    t, p = targets[valid].reshape(-1), preds[valid].reshape(-1)
    keep = (t >= 0) & (t < num_classes) & (p >= 0) & (p < num_classes)
    t, p = t[keep], p[keep]
    if t.numel() == 0:
        return conf
    binc = torch.bincount(t * num_classes + p, minlength=num_classes * num_classes)
    return conf + binc.reshape(num_classes, num_classes).to(conf.dtype)


def metrics_from_confusion(conf: torch.Tensor) -> Dict:
    """IoU/Dice over the whole split; classes with no ground truth are excluded."""
    conf = conf.double()
    tp = conf.diag()
    gt_support = conf.sum(dim=1)
    fp = conf.sum(dim=0) - tp
    fn = gt_support - tp

    iou = tp / (tp + fp + fn).clamp_min(1.0)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn).clamp_min(1.0)
    present = gt_support > 0
    n_present = int(present.sum().item())

    per_iou, per_dice = iou.tolist(), dice.tolist()
    support = [int(v) for v in gt_support.tolist()]
    for c in range(conf.shape[0]):
        if support[c] == 0:
            per_iou[c] = float("nan")
            per_dice[c] = float("nan")

    return {
        "mean_iou": float(iou[present].mean().item()) if n_present else float("nan"),
        "mean_dice": float(dice[present].mean().item()) if n_present else float("nan"),
        "mean_class_acc": float((tp / gt_support.clamp_min(1.0))[present].mean().item())
                          if n_present else float("nan"),
        "pixel_acc": float((tp.sum() / conf.sum().clamp_min(1.0)).item()),
        "per_class_iou": per_iou,
        "per_class_dice": per_dice,
        "class_support": support,
        "num_classes_present": n_present,
    }


# --------------------------------------------------------------------------
# boundary F1
# --------------------------------------------------------------------------
def fast_boundary_map(lbl: torch.Tensor) -> torch.Tensor:
    b = torch.zeros_like(lbl, dtype=torch.bool)
    b[:, :-1] |= (lbl[:, :-1] != lbl[:, 1:])
    b[:-1, :] |= (lbl[:-1, :] != lbl[1:, :])
    return b


def dilate(mask: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return mask
    t = mask.float().unsqueeze(0).unsqueeze(0)
    out = F.max_pool2d(t, kernel_size=2 * r + 1, stride=1, padding=r)
    return out[0, 0] > 0.5


@torch.no_grad()
def boundary_f1_score(preds, targets, tolerance: int = 2) -> float:
    """BF1 at a pixel tolerance, averaged over the batch."""
    scores = []
    for i in range(preds.shape[0]):
        pb, gb = fast_boundary_map(preds[i]), fast_boundary_map(targets[i])
        if pb.sum() == 0 and gb.sum() == 0:
            scores.append(1.0)
            continue
        pb_d, gb_d = dilate(pb, tolerance), dilate(gb, tolerance)
        prec = (pb & gb_d).sum().item() / (pb.sum().item() or 1)
        rec = (gb & pb_d).sum().item() / (gb.sum().item() or 1)
        scores.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(scores))


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------
def measure_latency_fps_mem(model, H: int, W: int, device, iters_warmup=10, iters_meas=100):
    """Latency (ms), throughput (FPS) and peak memory (GB) at batch size 1."""
    model.eval()
    x = torch.randn(1, 3, H, W, device=device)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(iters_warmup):
            model(x)
            if use_cuda:
                torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(iters_meas):
            model(x)
            if use_cuda:
                torch.cuda.synchronize()
    dur = time.time() - start

    avg_ms = (dur / max(1, iters_meas)) * 1000.0
    fps = (iters_meas / dur) if dur > 0 else 0.0
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if use_cuda else None
    return float(avg_ms), float(fps), (None if peak_gb is None else float(peak_gb))
