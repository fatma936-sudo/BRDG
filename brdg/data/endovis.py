"""EndoVis 2017 / 2018 surgical instrument segmentation."""

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from brdg.config import DEFAULT_IMAGE_SIZE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class EndovisDataset(Dataset):
    """
    Expects either
        <root>/<split>/left_frames/*.png  +  <root>/<split>/labels/*.png     (2018)
    or
        <root>/<split>/image/*.bmp        +  <root>/<split>/label/*.bmp      (2017)

    Labels must stay indexed (mode "P" or "L"); converting them to RGB destroys
    the class ids.
    """

    def __init__(self, data_root: str, split: str, num_classes: int,
                 image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
                 augment: bool = False):
        self.root = Path(data_root)
        self.split = split
        self.num_classes = num_classes
        self.image_size = image_size
        self.augment = augment
        self.samples = []

        if not self.root.exists():
            raise FileNotFoundError(f"Data root not found: {self.root}")

        if (self.root / split / "left_frames").exists():
            img_dir, lbl_dir, pattern = (self.root / split / "left_frames",
                                         self.root / split / "labels", "*.png")
        else:
            img_dir, lbl_dir, pattern = (self.root / split / "image",
                                         self.root / split / "label", "*.bmp")
        if not (img_dir.exists() and lbl_dir.exists()):
            raise FileNotFoundError(f"Expected EndoVis folders {img_dir} and {lbl_dir}")

        for img_path in sorted(img_dir.rglob(pattern)):
            lbl_path = lbl_dir / img_path.relative_to(img_dir)
            if lbl_path.exists():
                self.samples.append((str(img_path), str(lbl_path)))
        if not self.samples:
            raise RuntimeError(f"No paired images and labels found in {img_dir}")

        print(f"Loaded {len(self.samples)} {split} samples")
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_p, lbl_p = self.samples[idx]
        img = Image.open(img_p).convert("RGB")
        lbl_img = Image.open(lbl_p)
        if lbl_img.mode not in ("P", "L"):
            raise ValueError(f"Expected an indexed label mask, got {lbl_img.mode} in {lbl_p}")

        img = transforms.functional.resize(
            img, self.image_size, interpolation=transforms.InterpolationMode.BILINEAR)
        lbl_img = transforms.functional.resize(
            lbl_img, self.image_size, interpolation=transforms.InterpolationMode.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                img = transforms.functional.hflip(img)
                lbl_img = transforms.functional.hflip(lbl_img)
            if random.random() < 0.3:
                ang = random.uniform(-7, 7)
                img = transforms.functional.rotate(
                    img, ang, interpolation=transforms.InterpolationMode.BILINEAR, fill=(0, 0, 0))
                lbl_img = transforms.functional.rotate(
                    lbl_img, ang, interpolation=transforms.InterpolationMode.NEAREST, fill=0)

        img_t = self.normalize(transforms.ToTensor()(img))
        lbl_t = torch.from_numpy(np.array(lbl_img)).long()
        if lbl_t.min() < 0 or lbl_t.max() >= self.num_classes:
            raise ValueError(
                f"Label ids outside [0, {self.num_classes - 1}] in {lbl_p}: "
                f"min={lbl_t.min().item()}, max={lbl_t.max().item()}")
        return img_t, lbl_t, img_p
