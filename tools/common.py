"""Shared helpers for the visualisation tools."""

import numpy as np
import torch

from brdg.config import get_classes
from brdg.data import build_datasets
from brdg.models import DifferentiableSuperpixelNet, ResNetUNetBackbone
from brdg.utils import get_device

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def load_model(ckpt_path, dataset, arch="resnet34", num_superpixels=100,
               feat_ch=96, tau=0.5):
    device = get_device()
    _, num_classes = get_classes(dataset)
    ck = torch.load(ckpt_path, map_location=device)
    state = ck.get("model", ck)
    saved = ck.get("args") or {}
    arch = saved.get("backbone", arch)
    num_superpixels = saved.get("num_superpixels", num_superpixels)
    feat_ch = saved.get("feat_ch", feat_ch)

    backbone = ResNetUNetBackbone(arch=arch, feat_ch=feat_ch, pretrained=False)
    model = DifferentiableSuperpixelNet(
        num_classes=num_classes, num_superpixels=num_superpixels,
        feat_ch=feat_ch, tau=tau, backbone=backbone).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, num_classes, device


def load_val_frame(dataset, data_root, index, num_classes, image_size=(512, 640)):
    _, val_ds = build_datasets(dataset, data_root, num_classes, image_size)
    return val_ds[index]


def denormalize(img_t):
    rgb = img_t.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def label_to_rgb(lbl, classes):
    out = np.zeros(lbl.shape + (3,), dtype=np.uint8)
    for cid, meta in classes.items():
        out[lbl == cid] = tuple(meta["color"])
    return out


def boundaries_of(lbl):
    b = np.zeros_like(lbl, dtype=bool)
    b[:, :-1] |= lbl[:, :-1] != lbl[:, 1:]
    b[:-1, :] |= lbl[:-1, :] != lbl[1:, :]
    return b


def heatmap(v):
    """Scalar field in [0,1] -> inferno-like RGB."""
    anchors = [(0.0, (0, 0, 4)), (0.25, (60, 15, 110)), (0.5, (140, 41, 129)),
               (0.75, (222, 96, 77)), (1.0, (252, 255, 164))]
    v = np.clip(v, 0, 1)
    out = np.zeros(v.shape + (3,), dtype=np.float32)
    for i in range(len(anchors) - 1):
        x0, c0 = anchors[i]
        x1, c1 = anchors[i + 1]
        m = (v >= x0) & (v <= x1)
        t = np.zeros_like(v)
        t[m] = (v[m] - x0) / (x1 - x0)
        for ch in range(3):
            out[..., ch][m] = c0[ch] + t[m] * (c1[ch] - c0[ch])
    return out.astype(np.uint8)
