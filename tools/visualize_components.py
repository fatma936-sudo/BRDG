#!/usr/bin/env python3
"""
Render each BRDG component on one frame: the superpixel assignment, the gate,
both pathways, the gated output, and which pixels gating changed.

    python -m tools.visualize_components --ckpt runs/.../best_model.pth \
        --dataset endovis2018 --data_root /path/to/data --index 230 --outdir out/
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from brdg.config import get_classes
from tools.common import (boundaries_of, denormalize, heatmap, label_to_rgb,
                          load_model, load_val_frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="endovis2018")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    classes, num_classes = get_classes(args.dataset)
    model, _, device = load_model(args.ckpt, args.dataset)
    img_t, tgt, _ = load_val_frame(args.dataset, args.data_root, args.index, num_classes)

    with torch.no_grad():
        out = model(img_t.unsqueeze(0).to(device))

    rgb = denormalize(img_t)
    gt = tgt.numpy()
    gt_edge = boundaries_of(gt)
    A = out["assignment_map_spatial"][0]
    hard = A.argmax(0).cpu().numpy()
    coarse = out["coarse_logits"][0].argmax(0).cpu().numpy()
    refined = out["refined_logits"][0].argmax(0).cpu().numpy()
    final = out["final_logits"][0].argmax(0).cpu().numpy()
    gate = out["gate"][0, 0].cpu().numpy()

    def overlay(pred):
        lab = label_to_rgb(pred, classes).astype(np.float32)
        base = rgb.astype(np.float32)
        o = np.where((pred > 0)[..., None], 0.68 * lab + 0.32 * base, 0.45 * base)
        o[gt_edge] = (255, 255, 255)
        return o.clip(0, 255).astype(np.uint8)

    rng = np.random.default_rng(0)
    sp = rng.integers(40, 255, size=(A.shape[0], 3), dtype=np.uint8)[hard]
    sp[boundaries_of(hard)] = 255

    fixed = (coarse != gt) & (final == gt)
    broke = (coarse == gt) & (final != gt)
    impact = (rgb * 0.22).astype(np.uint8)
    impact[fixed] = (60, 230, 110)
    impact[broke] = (235, 70, 70)
    impact[gt_edge] = (255, 255, 255)

    panels = {"input": rgb, "gt": overlay(gt), "assign": sp, "gate": heatmap(gate),
              "coarse": overlay(coarse), "refined": overlay(refined),
              "final": overlay(final), "impact": impact}
    for name, arr in panels.items():
        Image.fromarray(arr).save(os.path.join(args.outdir, f"comp_{name}.png"), optimize=True)

    print(f"frame {args.index}: gating repaired {100 * fixed.mean():.2f}% of pixels, "
          f"damaged {100 * broke.mean():.2f}%, changed {100 * (final != coarse).mean():.2f}%")
    print(f"wrote {len(panels)} panels to {args.outdir}")


if __name__ == "__main__":
    main()
