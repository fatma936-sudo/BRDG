#!/usr/bin/env python3
"""
Capture the activation at every encoder and decoder stage with forward hooks and
project it from its channels onto three principal components, so the structure
each stage encodes is visible.

    python -m tools.visualize_layers --ckpt runs/.../best_model.pth \
        --dataset endovis2018 --data_root /path/to/data --index 230 --outdir out/
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from brdg.config import get_classes
from tools.common import load_model, load_val_frame

STAGES = ["stem", "layer1", "layer2", "layer3", "layer4",
          "up4", "up3", "up2", "up1", "feat"]


def pca_rgb(t: torch.Tensor) -> np.ndarray:
    """(C,H,W) -> (H,W,3) uint8 via the first three principal components."""
    C, H, W = t.shape
    X = t.reshape(C, H * W).t()
    X = X - X.mean(0, keepdim=True)
    q = min(3, C)
    _, _, V = torch.pca_lowrank(X, q=q, center=False)
    Y = (X @ V[:, :q]).t().reshape(q, H, W)
    if q < 3:
        Y = torch.cat([Y] * 3, 0)[:3]
    Y = Y.cpu().numpy()

    out = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(3):
        c = Y[i]
        lo, hi = np.percentile(c, 2), np.percentile(c, 98)
        out[..., i] = np.clip((c - lo) / max(hi - lo, 1e-8), 0, 1)
    return (out * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="endovis2018")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    _, num_classes = get_classes(args.dataset)
    model, _, device = load_model(args.ckpt, args.dataset)
    img_t, _, _ = load_val_frame(args.dataset, args.data_root, args.index, num_classes)

    acts = {}

    def hook(name):
        def fn(_m, _i, out):
            acts[name] = out.detach()[0].float()
        return fn

    B = model.backbone
    handles = [B.enc["relu"].register_forward_hook(hook("stem")),
               B.enc["layer1"].register_forward_hook(hook("layer1")),
               B.enc["layer2"].register_forward_hook(hook("layer2")),
               B.enc["layer3"].register_forward_hook(hook("layer3")),
               B.enc["layer4"].register_forward_hook(hook("layer4")),
               B.up4.register_forward_hook(hook("up4")),
               B.up3.register_forward_hook(hook("up3")),
               B.up2.register_forward_hook(hook("up2")),
               B.up1.register_forward_hook(hook("up1")),
               B.out_conv.register_forward_hook(hook("feat"))]
    with torch.no_grad():
        model(img_t.unsqueeze(0).to(device))
    for h in handles:
        h.remove()

    H0 = img_t.shape[-2]
    meta = {}
    for name in STAGES:
        t = acts[name]
        C, H, W = t.shape
        # NEAREST keeps each stage's true spatial resolution visible
        Image.fromarray(pca_rgb(t)).resize((img_t.shape[-1], H0), Image.NEAREST).save(
            os.path.join(args.outdir, f"layer_{name}.png"), optimize=True)
        meta[name] = {"C": int(C), "H": int(H), "W": int(W), "stride": int(round(H0 / H))}
        print(f"{name:<8} ({C:>4}, {H:>3}, {W:>3})  stride 1/{meta[name]['stride']}")

    with open(os.path.join(args.outdir, "shapes.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {len(STAGES)} layer maps to {args.outdir}")


if __name__ == "__main__":
    main()
