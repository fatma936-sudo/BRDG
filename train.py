#!/usr/bin/env python3
"""
Train BRDG.

Defaults follow the paper: ResNet-34 U-Net backbone, K = 100 superpixels,
512 x 640 input, AdamW at 1e-4 with weight decay 1e-4 and a 0.1x encoder rate,
100 epochs on a warmup -> ramp -> full schedule, loss = 0.5 CE + 0.5 Tversky
plus the boundary BCE and the adjacency-boosted contrastive terms.

    python train.py --dataset endovis2018 --data_root /path/to/endovis2018_tools \
                    --save_root runs/endovis2018
"""

import argparse
import csv
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brdg.config import (DEFAULT_FEAT_CH, DEFAULT_IMAGE_SIZE, DEFAULT_NUM_SUPERPIXELS,
                         get_classes)
from brdg.data import build_datasets
from brdg.engine import evaluate_full, set_encoder_trainable, train_one_epoch, validate
from brdg.losses import SegCombinedLoss
from brdg.models import BACKBONES, DifferentiableSuperpixelNet, ResNetUNetBackbone
from brdg.schedules import loss_weights_for_epoch, tau_for_epoch
from brdg.utils import count_parameters, get_device, seed_everything


def build_argparser():
    p = argparse.ArgumentParser("Train BRDG")
    p.add_argument("--dataset", required=True,
                   choices=("endovis2017", "endovis2018", "cityscapes", "ade20k", "bsds500"))
    p.add_argument("--data_root", required=True)
    p.add_argument("--save_root", default="runs/brdg")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")

    p.add_argument("--backbone", default="resnet34", choices=sorted(BACKBONES))
    p.add_argument("--num_superpixels", type=int, default=DEFAULT_NUM_SUPERPIXELS)
    p.add_argument("--feat_ch", type=int, default=DEFAULT_FEAT_CH)

    p.add_argument("--image_height", type=int, default=DEFAULT_IMAGE_SIZE[0])
    p.add_argument("--image_width", type=int, default=DEFAULT_IMAGE_SIZE[1])
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--accum_steps", type=int, default=4)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--encoder_lr_mult", type=float, default=0.1)

    p.add_argument("--w_ce", type=float, default=0.5)
    p.add_argument("--w_tv", type=float, default=0.5)
    p.add_argument("--tv_alpha", type=float, default=0.4)
    p.add_argument("--tv_beta", type=float, default=0.6)

    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--ramp_epochs", type=int, default=5)
    p.add_argument("--anneal_epochs", type=int, default=10)
    p.add_argument("--start_tau", type=float, default=1.0)
    p.add_argument("--end_tau", type=float, default=0.5)

    p.add_argument("--coarse_w", type=float, default=0.3)
    p.add_argument("--sp_bnd_w", type=float, default=1.0)
    p.add_argument("--compact_w", type=float, default=1e-3)
    p.add_argument("--contrastive_class_w", type=float, default=0.05)
    p.add_argument("--contrastive_boundary_w", type=float, default=0.02)
    p.add_argument("--gate_entropy_w", type=float, default=1e-4)
    p.add_argument("--adj_boost", type=float, default=2.0,
                   help="w_ik = 1 + (adj_boost - 1) for adjacent negatives; 1.0 disables it")

    p.add_argument("--tolerance_px", type=int, default=2)
    p.add_argument("--dataset-level-metrics", dest="dataset_level", action="store_true",
                   help="also report confusion-matrix mIoU/Dice over classes with GT support")
    return p


def main():
    args = build_argparser().parse_args()
    classes, num_classes = get_classes(args.dataset)
    seed_everything(args.seed, deterministic=args.deterministic)
    device = get_device()

    H, W = args.image_height, args.image_width
    os.makedirs(args.save_root, exist_ok=True)
    ckpt_dir = os.path.join(args.save_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_ds, val_ds = build_datasets(args.dataset, args.data_root, num_classes, (H, W))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    backbone = ResNetUNetBackbone(arch=args.backbone, feat_ch=args.feat_ch,
                                  pretrained=True, freeze_bn=True)
    model = DifferentiableSuperpixelNet(
        num_classes=num_classes, num_superpixels=args.num_superpixels,
        feat_ch=args.feat_ch, tau=args.start_tau, backbone=backbone).to(device)

    print(f"BRDG | {args.backbone} | {count_parameters(model) / 1e6:.2f} M trainable params "
          f"| {num_classes} classes | K = {args.num_superpixels}", flush=True)

    enc_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if name.startswith("backbone.enc") else other_params).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.lr * args.encoder_lr_mult},
         {"params": other_params, "lr": args.lr}],
        weight_decay=args.weight_decay)

    seg_loss_fn = SegCombinedLoss(w_ce=args.w_ce, w_tv=args.w_tv,
                                  tv_alpha=args.tv_alpha, tv_beta=args.tv_beta)
    bce_boundary = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0, device=device))

    set_encoder_trainable(model, False)          # frozen through warmup
    best_miou, best_path = -1.0, os.path.join(ckpt_dir, "best_model.pth")

    for ep in range(1, args.epochs + 1):
        if ep == args.warmup_epochs + 1:
            set_encoder_trainable(model, True)

        cw, bw, cmpw, ctr_cls, ctr_bnd, gew = loss_weights_for_epoch(
            ep, warmup_epochs=args.warmup_epochs, ramp_epochs=args.ramp_epochs,
            final_coarse_w=args.coarse_w, final_sp_bnd_w=args.sp_bnd_w,
            final_compact_w=args.compact_w,
            final_contrastive_class_w=args.contrastive_class_w,
            final_contrastive_boundary_w=args.contrastive_boundary_w,
            final_gate_entropy_w=args.gate_entropy_w)
        tau = tau_for_epoch(ep, args.start_tau, args.end_tau, args.anneal_epochs)

        train_one_epoch(train_loader, model, optimizer, seg_loss_fn, bce_boundary, device,
                        num_classes, contrastive_class_w=ctr_cls,
                        contrastive_boundary_w=ctr_bnd, coarse_w=cw, sp_bnd_w=bw,
                        compact_w=cmpw, gate_entropy_w=gew, adj_boost=args.adj_boost,
                        accum_steps=args.accum_steps, epoch=ep, tau=tau)

        miou, ds_miou = validate(val_loader, model, device, num_classes,
                                 dataset_level=args.dataset_level)
        extra = f"  (dataset-level: {ds_miou:.4f})" if ds_miou is not None else ""
        print(f"Epoch {ep:03d}/{args.epochs}  Val mIoU: {miou:.4f}{extra}", flush=True)

        if miou > best_miou:
            best_miou = miou
            torch.save({"model": model.state_dict(),
                        "args": vars(args),
                        "num_classes": num_classes}, best_path)
            print(f"  saved best to {best_path}", flush=True)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    metrics = evaluate_full(val_loader, model, device, num_classes, H, W,
                            tolerance_px=args.tolerance_px, dataset_level=args.dataset_level)

    print("\n=== Final evaluation (best checkpoint) ===")
    print(f"mIoU  {metrics['mean_iou']:.4f} | Dice {metrics['mean_dice']:.4f} "
          f"| BF1 {metrics['bf1']:.4f}")
    print(f"{metrics['inference_ms']:.2f} ms | {metrics['fps']:.1f} FPS "
          f"| {count_parameters(model) / 1e6:.2f} M params")
    if args.dataset_level:
        print(f"dataset-level mIoU {metrics['dataset_level_mean_iou']:.4f} "
              f"over {metrics['dataset_level_num_classes_present']}/{num_classes} classes")

    with open(os.path.join(args.save_root, "metrics.json"), "w") as f:
        json.dump({"args": vars(args), "best_val_miou": best_miou, **metrics}, f, indent=2)

    per_class_csv = os.path.join(args.save_root, "per_class_metrics.csv")
    with open(per_class_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "IoU", "Dice"])
        for cid in range(num_classes):
            w.writerow([cid, classes[cid]["name"],
                        round(metrics["per_class_iou"][cid], 6),
                        round(metrics["per_class_dice"][cid], 6)])

    print(f"\nSaved to {args.save_root}  ({datetime.now().isoformat(timespec='seconds')})")


if __name__ == "__main__":
    main()
