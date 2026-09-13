"""Training and evaluation loops."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from brdg.losses import (boundary_contrastive_loss, get_region_class_labels,
                         get_superpixel_boundary_labels, supervised_contrastive_loss)
from brdg.metrics import (boundary_f1_score, dice_score, measure_latency_fps_mem,
                          metrics_from_confusion, miou_score, update_confusion)


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    for name, p in model.named_parameters():
        if name.startswith("backbone.enc"):
            p.requires_grad = trainable


def train_one_epoch(loader, model, optimizer, seg_loss_fn, bce_boundary, device,
                    num_classes: int,
                    contrastive_class_w: float = 0.0,
                    contrastive_boundary_w: float = 0.0,
                    coarse_w: float = 0.0, sp_bnd_w: float = 0.0, compact_w: float = 0.0,
                    gate_entropy_w: float = 0.0,
                    adj_boost: float = 2.0,
                    accum_steps: int = 4, epoch: int = 0, tau: float = 1.0) -> None:
    model.train()
    model.set_tau(tau)
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Train {epoch:03d}", leave=False)

    ema = 0.0
    for step, (imgs, targets, _) in enumerate(pbar):
        imgs, targets = imgs.to(device), targets.to(device)
        out = model(imgs)

        H, W = out["feature_shape"]
        A_hwk = out["assignment_map"]
        hard_spatial = out["assignment_map_spatial"].argmax(dim=1)

        targets_down = F.interpolate(
            targets.unsqueeze(1).float(), size=(H, W), mode="nearest").squeeze(1).long()
        bnd_targets = get_superpixel_boundary_labels(targets_down, A_hwk, num_classes)
        reg_labels = get_region_class_labels(targets_down, A_hwk, num_classes)

        loss_main = seg_loss_fn(out["final_logits"], targets)
        loss_coarse = seg_loss_fn(out["coarse_logits"], targets)

        with torch.no_grad():
            pos = bnd_targets.sum()
            neg = bnd_targets.numel() - pos
            pw = (neg / pos).clamp(min=1.0) if pos > 0 else torch.tensor(1.0, device=device)
        bce_boundary.pos_weight = pw
        loss_bnd = bce_boundary(out["superpixel_boundary_logits"], bnd_targets)

        A_bhwk = A_hwk.view(imgs.size(0), H, W, -1)
        compact = (out["dist_spatial"].permute(0, 2, 3, 1) * A_bhwk).mean()

        loss_contra_cls = supervised_contrastive_loss(
            out["region_feats"], reg_labels, temperature=0.07, max_regions=64)
        loss_contra_bnd = boundary_contrastive_loss(
            out["region_feats"], bnd_targets, hard_spatial,
            temperature=0.07, max_regions=64, adjacency_boost=adj_boost)

        g = out["gate"].clamp(1e-6, 1 - 1e-6)
        gate_entropy = -(g * g.log() + (1 - g) * (1 - g).log()).mean()

        total = (loss_main
                 + coarse_w * loss_coarse
                 + sp_bnd_w * loss_bnd
                 + compact_w * compact
                 + contrastive_class_w * loss_contra_cls
                 + contrastive_boundary_w * loss_contra_bnd
                 + gate_entropy_w * gate_entropy)

        (total / accum_steps).backward()
        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        ema = total.item() if step == 0 else 0.98 * ema + 0.02 * total.item()
        pbar.set_postfix(loss=f"{ema:.3f}", mi=f"{loss_main.item():.3f}",
                         bnd=f"{loss_bnd.item():.3f}", ctrb=f"{loss_contra_bnd.item():.3f}",
                         tau=f"{tau:.2f}", adj=f"{adj_boost:.2f}")


@torch.no_grad()
def validate(loader, model, device, num_classes: int, dataset_level: bool = False):
    """Return (per-image mIoU, dataset-level mIoU or None)."""
    model.eval()
    conf = (torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
            if dataset_level else None)
    total, n = 0.0, 0
    for imgs, targets, _ in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)["final_logits"].argmax(1)
        total += miou_score(preds, targets, num_classes)
        if dataset_level:
            conf = update_confusion(conf, preds, targets, num_classes)
        n += 1
    model.train()
    per_image = float(total / max(1, n))
    ds_level = float(metrics_from_confusion(conf.cpu())["mean_iou"]) if dataset_level else None
    return per_image, ds_level


@torch.no_grad()
def evaluate_full(loader, model, device, num_classes: int, H: int, W: int,
                  tolerance_px: int = 2, dataset_level: bool = False) -> Dict:
    model.eval()
    class_iou = torch.zeros(num_classes, dtype=torch.float64)
    class_dice = torch.zeros(num_classes, dtype=torch.float64)
    conf = (torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
            if dataset_level else None)
    miou_sum = mdice_sum = bf1_sum = 0.0
    steps = 0

    for imgs, targets, _ in tqdm(loader, desc="Eval", leave=False):
        imgs, targets = imgs.to(device), targets.to(device).long()
        preds = model(imgs)["final_logits"].argmax(1)

        per_iou, batch_miou = miou_score(preds, targets, num_classes, return_per_class=True)
        per_dice, batch_mdice = dice_score(preds, targets, num_classes, return_per_class=True)
        class_iou += torch.tensor(per_iou, dtype=torch.float64)
        class_dice += torch.tensor(per_dice, dtype=torch.float64)
        miou_sum += batch_miou
        mdice_sum += batch_mdice
        bf1_sum += boundary_f1_score(preds, targets, tolerance=tolerance_px)
        if dataset_level:
            conf = update_confusion(conf, preds, targets, num_classes)
        steps += 1

    infer_ms, fps, peak_gb = measure_latency_fps_mem(model, H, W, device)
    result = {
        "mean_iou": float(miou_sum / max(1, steps)),
        "mean_dice": float(mdice_sum / max(1, steps)),
        "bf1": float(bf1_sum / max(1, steps)),
        "per_class_iou": (class_iou / max(1, steps)).tolist(),
        "per_class_dice": (class_dice / max(1, steps)).tolist(),
        "inference_ms": infer_ms,
        "fps": fps,
        "peak_mem_gb": peak_gb,
    }
    if dataset_level:
        ds = metrics_from_confusion(conf.cpu())
        result.update({f"dataset_level_{k}": v for k, v in ds.items()})
    return result
