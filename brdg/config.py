"""Dataset class definitions and shared constants."""

from typing import Dict, List

IGNORE_INDEX = 255
DEFAULT_IMAGE_SIZE = (512, 640)          # (H, W), as used throughout the paper
DEFAULT_NUM_SUPERPIXELS = 100
DEFAULT_FEAT_CH = 96                     # C_f in the paper

ENDOVIS2018_CLASSES: Dict[int, Dict[str, List[int]]] = {
    0: {"name": "background", "color": [0, 0, 0]},
    1: {"name": "bipolar-forceps", "color": [0, 255, 0]},
    2: {"name": "prograsp-forceps", "color": [0, 255, 255]},
    3: {"name": "large-needle-driver", "color": [125, 255, 12]},
    4: {"name": "monopolar-curved-scissors", "color": [255, 55, 0]},
    5: {"name": "ultrasound-probe", "color": [24, 55, 125]},
    6: {"name": "suction-instrument", "color": [187, 155, 25]},
    7: {"name": "clip-applier", "color": [0, 255, 125]},
}

ENDOVIS2017_CLASSES: Dict[int, Dict[str, List[int]]] = {
    0: {"name": "background", "color": [0, 0, 0]},
    1: {"name": "bipolar-forceps", "color": [0, 255, 0]},
    2: {"name": "prograsp-forceps", "color": [0, 255, 255]},
    3: {"name": "large-needle-driver", "color": [125, 255, 12]},
    4: {"name": "vessel-sealer", "color": [255, 55, 0]},
    5: {"name": "grasping-retractor", "color": [24, 55, 125]},
    6: {"name": "monopolar-curved-scissors", "color": [187, 155, 25]},
    7: {"name": "ultrasound-probe", "color": [0, 255, 125]},
}

CITYSCAPES_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic-light", "traffic-sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]
CITYSCAPES_CLASSES = {i: {"name": n, "color": [0, 0, 0]} for i, n in enumerate(CITYSCAPES_NAMES)}
ADE20K_CLASSES = {i: {"name": f"ade20k-{i:03d}", "color": [0, 0, 0]} for i in range(150)}
BSDS500_CLASSES = {
    0: {"name": "non-boundary", "color": [0, 0, 0]},
    1: {"name": "boundary", "color": [255, 255, 255]},
}

CLASS_SETS = {
    "endovis2017": ENDOVIS2017_CLASSES,
    "endovis2018": ENDOVIS2018_CLASSES,
    "cityscapes": CITYSCAPES_CLASSES,
    "ade20k": ADE20K_CLASSES,
    "bsds500": BSDS500_CLASSES,
}


def get_classes(dataset: str):
    """Return (classes dict, num_classes) for a dataset name."""
    if dataset not in CLASS_SETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(CLASS_SETS)}")
    classes = CLASS_SETS[dataset]
    return classes, len(classes)
