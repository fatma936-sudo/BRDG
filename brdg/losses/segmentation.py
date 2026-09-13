"""Primary segmentation objective: 0.5 cross-entropy + 0.5 Tversky."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from brdg.config import IGNORE_INDEX


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.4, beta: float = 0.6,
                 smooth: float = 1e-6, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.softmax(logits, dim=1)
        valid = targets != self.ignore_index
        safe_targets = targets.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_targets, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()

        valid = valid.unsqueeze(1).float()
        probs = probs * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        tp = (probs * one_hot).sum(dims)
        fp = (probs * (1 - one_hot)).sum(dims)
        fn = ((1 - probs) * one_hot).sum(dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class SegCombinedLoss(nn.Module):
    def __init__(self, w_ce: float = 0.5, w_tv: float = 0.5,
                 tv_alpha: float = 0.4, tv_beta: float = 0.6):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.tv = TverskyLoss(alpha=tv_alpha, beta=tv_beta)
        self.w_ce, self.w_tv = w_ce, w_tv

    def forward(self, logits, targets):
        return self.w_ce * self.ce(logits, targets) + self.w_tv * self.tv(logits, targets)
