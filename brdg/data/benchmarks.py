"""General-domain benchmarks: Cityscapes-19, ADE20K-150 and BSDS500."""

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from brdg.config import DEFAULT_IMAGE_SIZE, IGNORE_INDEX
from brdg.data.endovis import IMAGENET_MEAN, IMAGENET_STD


class BenchmarkSegmentationDataset(Dataset):
    """Cityscapes (19 train ids) and ADE20K (150 classes)."""

    def __init__(self, data_root: str, dataset: str, split: str, num_classes: int,
                 image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE, augment: bool = False):
        self.root = Path(data_root)
        self.dataset = dataset
        self.num_classes = num_classes
        self.image_size = image_size
        self.augment = augment
        self.samples = []
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        if dataset == "cityscapes":
            img_dir = self.root / "leftImg8bit" / split
            if not img_dir.exists():
                img_dir = self.root / "images" / split
            lbl_dir = self.root / "gtFine" / split
            for img_path in sorted(img_dir.rglob("*_leftImg8bit.png")):
                rel = img_path.relative_to(img_dir)
                name = img_path.name.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
                lbl_path = lbl_dir / rel.parent / name
                if lbl_path.exists():
                    self.samples.append((str(img_path), str(lbl_path)))
        elif dataset == "ade20k":
            ade_split = "training" if split == "train" else "validation"
            img_dir = self.root / "images" / ade_split
            lbl_dir = self.root / "annotations" / ade_split
            for img_path in sorted(img_dir.rglob("*.jpg")):
                lbl_path = lbl_dir / img_path.relative_to(img_dir).with_suffix(".png")
                if lbl_path.exists():
                    self.samples.append((str(img_path), str(lbl_path)))
        else:
            raise ValueError(f"Unsupported benchmark dataset: {dataset}")

        if not self.samples:
            raise RuntimeError(f"No paired {dataset} samples found under {self.root}")
        print(f"Loaded {len(self.samples)} {dataset} {split} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, lbl_p = self.samples[idx]
        img = Image.open(img_p).convert("RGB")
        lbl = Image.open(lbl_p)
        img = transforms.functional.resize(
            img, self.image_size, interpolation=transforms.InterpolationMode.BILINEAR)
        lbl = transforms.functional.resize(
            lbl, self.image_size, interpolation=transforms.InterpolationMode.NEAREST)
        if self.augment and random.random() < 0.5:
            img = transforms.functional.hflip(img)
            lbl = transforms.functional.hflip(lbl)

        img_t = self.normalize(transforms.ToTensor()(img))
        lbl_t = torch.from_numpy(np.array(lbl, dtype=np.int64)).long()
        if self.dataset == "ade20k":
            # ADE20K ships 0 as "unlabelled"; shift so classes are 0..149.
            lbl_t = torch.where(lbl_t == 0, torch.full_like(lbl_t, IGNORE_INDEX), lbl_t - 1)
        valid = lbl_t != IGNORE_INDEX
        if valid.any() and (lbl_t[valid].min() < 0 or lbl_t[valid].max() >= self.num_classes):
            raise ValueError(f"Invalid {self.dataset} labels in {lbl_p}")
        return img_t, lbl_t, img_p


class BSDS500Dataset(Dataset):
    """Binary consensus-boundary view of the BSDS500 multi-annotator labels."""

    def __init__(self, data_root: str, split: str, num_classes: int = 2,
                 image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
                 augment: bool = False, consensus: float = 0.5):
        from scipy.io import loadmat          # only BSDS500 needs scipy.io

        self.loadmat = loadmat
        self.root = Path(data_root)
        self.image_size = image_size
        self.augment = augment
        self.consensus = consensus
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        img_dir = self.root / "images" / split
        gt_dir = self.root / "groundTruth" / split
        self.samples = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            gt_path = gt_dir / f"{img_path.stem}.mat"
            if gt_path.exists():
                self.samples.append((str(img_path), str(gt_path)))
        if not self.samples:
            raise RuntimeError(f"No BSDS500 pairs found in {img_dir} and {gt_dir}")
        print(f"Loaded {len(self.samples)} BSDS500 {split} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, gt_p = self.samples[idx]
        img = Image.open(img_p).convert("RGB")
        annotations = self.loadmat(gt_p)["groundTruth"]
        boundaries = [annotations[0, i]["Boundaries"][0, 0].astype(np.float32)
                      for i in range(annotations.shape[1])]
        consensus = np.mean(boundaries, axis=0)
        lbl = Image.fromarray((consensus >= self.consensus).astype(np.uint8))

        img = transforms.functional.resize(
            img, self.image_size, interpolation=transforms.InterpolationMode.BILINEAR)
        lbl = transforms.functional.resize(
            lbl, self.image_size, interpolation=transforms.InterpolationMode.NEAREST)
        if self.augment and random.random() < 0.5:
            img = transforms.functional.hflip(img)
            lbl = transforms.functional.hflip(lbl)

        img_t = self.normalize(transforms.ToTensor()(img))
        lbl_t = torch.from_numpy(np.array(lbl, dtype=np.int64)).long()
        return img_t, lbl_t, img_p
