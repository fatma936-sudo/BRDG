# BRDG: Boundary-Responsive Differentiable Gating for Superpixel-Based Segmentation

Official repository for:

**Boundary-Responsive Differentiable Gating for Superpixel-Based Segmentation**

> CVPR 2026

BRDG introduces a boundary-responsive differentiable gating mechanism for superpixel-based segmentation. The method selectively activates high-fidelity refinement around ambiguous semantic boundaries while preserving the efficiency of superpixel-level reasoning.

---

## 🌐 Project Page

The project page includes the paper, poster, video, Medium tutorial, model variants, qualitative results, pretrained weights, and citation.

👉 **Website:** https://fatma936-sudo.github.io/BRDG/

---

## 🔗 Resources

| Resource | Link |
|---|---|
| Paper | Coming soon |
| Project Page | https://fatma936-sudo.github.io/BRDG/ |
| Code | https://github.com/fatma936-sudo/BRDG |
| Hugging Face Weights | Coming soon |
| Poster | Coming soon |
| YouTube Video | Coming soon |
| Medium Tutorial | Coming soon |

---

## 🧠 Method Overview

Superpixel-based segmentation improves efficiency by grouping visually coherent pixels into compact regions. However, this often reduces boundary precision, especially around thin structures, small objects, and complex semantic transitions.

BRDG addresses this issue by learning a differentiable gate that identifies boundary-sensitive regions and selectively applies high-resolution refinement only where needed.

<p align="center">
  <img src="docs/assets/teaser.png" width="90%">
</p>

---

## 🧩 Model Variants

| Variant | Backbone | Input Size | Use Case | Weights |
|---|---|---:|---|---|
| BRDG-Tiny | SegFormer-B0 | 512 × 512 | Fast inference / ablation | Coming soon |
| BRDG-Base | SegFormer-B2 | 512 × 512 | Balanced accuracy and speed | Coming soon |
| BRDG-Large | SegFormer-B5 | 512 × 512 | Best segmentation quality | Coming soon |
| BRDG-Mobile | MobileNet / EfficientNet | 512 × 512 | Lightweight deployment | Coming soon |

---

## 🚀 Installation

```bash
git clone https://github.com/fatma936-sudo/BRDG.git
cd BRDG

conda create -n brdg python=3.10 -y
conda activate brdg

pip install -r requirements.txt
