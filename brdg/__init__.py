"""BRDG: Boundary-Responsive Differentiable Gating for Superpixel-Based Segmentation."""

__version__ = "0.1.0"

from brdg.models import DifferentiableSuperpixelNet, ResNetUNetBackbone

__all__ = ["DifferentiableSuperpixelNet", "ResNetUNetBackbone", "__version__"]
