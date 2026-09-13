"""Build the train/val pair for any supported dataset name."""

from typing import Tuple

from brdg.config import DEFAULT_IMAGE_SIZE
from brdg.data.benchmarks import BenchmarkSegmentationDataset, BSDS500Dataset
from brdg.data.endovis import EndovisDataset


def build_datasets(dataset: str, data_root: str, num_classes: int,
                   image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE):
    """Return (train_dataset, val_dataset)."""
    if dataset.startswith("endovis"):
        return (EndovisDataset(data_root, "train", num_classes, image_size, augment=True),
                EndovisDataset(data_root, "val", num_classes, image_size, augment=False))
    if dataset in ("cityscapes", "ade20k"):
        return (BenchmarkSegmentationDataset(data_root, dataset, "train", num_classes,
                                             image_size, augment=True),
                BenchmarkSegmentationDataset(data_root, dataset, "val", num_classes,
                                             image_size, augment=False))
    if dataset == "bsds500":
        return (BSDS500Dataset(data_root, "train", num_classes, image_size, augment=True),
                BSDS500Dataset(data_root, "val", num_classes, image_size, augment=False))
    raise ValueError(f"unknown dataset {dataset!r}")
