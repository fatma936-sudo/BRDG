"""The three cooperative agents of BRDG in one differentiable module."""

import torch
import torch.nn as nn

from brdg.config import DEFAULT_FEAT_CH, DEFAULT_NUM_SUPERPIXELS
from brdg.models.backbone import ResNetUNetBackbone


class DifferentiableSuperpixelNet(nn.Module):
    """
    Agent 1 (Region and Feature Creator)
        backbone -> F (B, C_f, H, W); two 1x1 heads emit the coarse class logits
        Y_c and the assignment logits, softmaxed at temperature tau into A;
        soft-pooling F through A gives K region descriptors r_k.

    Agent 2 (Boundary Detector)
        an MLP maps each r_k to a boundary probability p_k; re-projecting through
        A yields the dense pixel-wise gate g.

    Agent 3 (Refinement)
        a second MLP classifies each region; scattering those logits back through
        A gives Y_r, and the output is Y = (1 - g) * Y_c + g * Y_r.
    """

    def __init__(self, num_classes: int,
                 num_superpixels: int = DEFAULT_NUM_SUPERPIXELS,
                 feat_ch: int = DEFAULT_FEAT_CH,
                 tau: float = 1.0,
                 backbone: nn.Module = None,
                 arch: str = "resnet34"):
        super().__init__()
        self.num_classes = int(num_classes)
        self.K = int(num_superpixels)
        self.tau = float(tau)

        self.backbone = backbone if backbone is not None else ResNetUNetBackbone(
            arch=arch, feat_ch=feat_ch, pretrained=True)

        self.coarse_head = nn.Conv2d(feat_ch, num_classes, 1)
        self.assign_head = nn.Conv2d(feat_ch, self.K, 1)

        self.boundary_mlp = nn.Sequential(
            nn.Linear(feat_ch, feat_ch), nn.ReLU(True), nn.Linear(feat_ch, 1))
        self.region_cls_mlp = nn.Sequential(
            nn.Linear(feat_ch, feat_ch), nn.ReLU(True), nn.Linear(feat_ch, num_classes))

    def set_tau(self, tau: float) -> None:
        self.tau = float(tau)

    def _soft_assign(self, feat):
        logits = self.assign_head(feat)                 # (B,K,H,W)
        return torch.softmax(logits / self.tau, dim=1), logits

    def _soft_centers(self, A):
        B, K, H, W = A.shape
        y = torch.linspace(0, 1, H, device=A.device).view(1, 1, H, 1).expand(B, K, H, 1)
        x = torch.linspace(0, 1, W, device=A.device).view(1, 1, 1, W).expand(B, K, 1, W)
        mass = A.sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)
        ybar = (A * y).sum(dim=(2, 3), keepdim=True) / mass
        xbar = (A * x).sum(dim=(2, 3), keepdim=True) / mass
        dist2 = (x - xbar).pow(2) + (y - ybar).pow(2)
        return xbar.squeeze(-1).squeeze(-1), ybar.squeeze(-1).squeeze(-1), dist2

    def _soft_pool_features(self, feat, A):
        mass = A.sum(dim=(2, 3)).clamp_min(1e-8)             # (B,K)
        pooled = torch.einsum("bkij,bfij->bkf", A, feat)     # (B,K,C_f)
        return pooled / mass.unsqueeze(-1)

    def forward(self, x):
        feat = self.backbone(x)
        B, Fd, H, W = feat.shape

        coarse_logits = self.coarse_head(feat)               # (B,C,H,W)
        A, A_logits = self._soft_assign(feat)                # (B,K,H,W)
        xbar, ybar, dist2 = self._soft_centers(A)
        region_feats = self._soft_pool_features(feat, A)     # (B,K,C_f)

        boundary_logits = self.boundary_mlp(region_feats)    # (B,K,1)
        boundary_prob = torch.sigmoid(boundary_logits)

        region_class_logits = self.region_cls_mlp(region_feats)             # (B,K,C)
        refined = torch.einsum("bkij,bkc->bcij", A, region_class_logits)    # (B,C,H,W)

        gate = torch.einsum("bkij,bkq->bij", A, boundary_prob).unsqueeze(1) # (B,1,H,W)
        gate = gate.clamp(0, 1)
        final_logits = (1.0 - gate) * coarse_logits + gate * refined

        return {
            "final_logits": final_logits,
            "coarse_logits": coarse_logits,
            "refined_logits": refined,
            "superpixel_boundary_logits": boundary_logits,
            "assignment_map": A.permute(0, 2, 3, 1).reshape(B, H * W, self.K),
            "assignment_map_spatial": A,
            "dist_spatial": dist2,
            "feature_shape": (H, W),
            "region_feats": region_feats,
            "gate": gate,
            "mlp_output": region_class_logits,
        }
