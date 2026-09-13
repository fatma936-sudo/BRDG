"""Region-level contrastive objectives, including the adjacency boost."""

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(feats: torch.Tensor, labels: torch.Tensor,
                                temperature: float = 0.07,
                                max_regions: int = 64) -> torch.Tensor:
    """Supervised InfoNCE over region descriptors grouped by their majority class."""
    B, K, _ = feats.shape
    dev = feats.device
    losses = []
    for b in range(B):
        idx = torch.randperm(K, device=dev)[:min(K, max_regions)]
        fb, lb = feats[b][idx], labels[b][idx]
        if torch.unique(lb).numel() < 2 or fb.size(0) < 2:
            continue
        fb = F.normalize(fb, dim=1)
        sim = torch.mm(fb, fb.t()) / temperature
        logits = sim - torch.eye(sim.size(0), device=dev) * 1e9

        pos_mask = (lb.view(-1, 1) == lb.view(1, -1)).float()
        pos_mask.fill_diagonal_(0)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        denom = pos_mask.sum(dim=1).clamp_min(1)
        losses.append((-(pos_mask * log_prob).sum(dim=1) / denom).mean())
    if not losses:
        return torch.tensor(0.0, device=dev)
    return torch.stack(losses).mean()


def adjacency_from_assignments(hard_assign_spatial: torch.Tensor, K: int) -> torch.Tensor:
    """(K, K) boolean region-adjacency graph from a hard assignment map (H, W)."""
    adj = torch.zeros((K, K), dtype=torch.bool, device=hard_assign_spatial.device)
    for a, b in ((hard_assign_spatial[:, :-1], hard_assign_spatial[:, 1:]),
                 (hard_assign_spatial[:-1, :], hard_assign_spatial[1:, :])):
        diff = a != b
        if diff.any():
            ai, bi = a[diff].view(-1), b[diff].view(-1)
            adj[ai, bi] = True
            adj[bi, ai] = True
    adj.fill_diagonal_(False)
    return adj


def boundary_contrastive_loss(feats: torch.Tensor, boundary_labels: torch.Tensor,
                              hard_assign_spatial: torch.Tensor,
                              temperature: float = 0.07, max_regions: int = 64,
                              adjacency_boost: float = 2.0) -> torch.Tensor:
    """
    Supervised InfoNCE over boundary-vs-interior regions with the adjacency boost
    w_ik = 1 + alpha * 1[i, k adjacent], applied as a log-weight on the negatives
    so spatially neighbouring regions on opposite sides of a semantic boundary are
    pushed apart hardest. adjacency_boost = 1.0 disables it (alpha = 0).
    """
    B, K, _ = feats.shape
    dev = feats.device
    eps = 1e-12
    losses = []

    for b in range(B):
        idx = torch.randperm(K, device=dev)[:min(K, max_regions)]
        fb = feats[b][idx]
        bl = boundary_labels[b].squeeze(-1).long()[idx]
        if torch.unique(bl).numel() < 2 or fb.size(0) < 2:
            continue

        fb = F.normalize(fb, dim=1)
        sim = torch.mm(fb, fb.t()) / temperature
        M = sim.size(0)
        eye = torch.eye(M, device=dev, dtype=torch.bool)
        logits = sim.masked_fill(eye, -1e9)

        pos_mask = (bl.view(-1, 1) == bl.view(1, -1)) & (~eye)
        neg_mask = (~pos_mask) & (~eye)

        adj = adjacency_from_assignments(hard_assign_spatial[b], K)[idx][:, idx]
        weight = torch.ones_like(logits) + (adj & neg_mask).float() * (adjacency_boost - 1.0)
        logits = logits + torch.log(weight.clamp_min(eps))

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        denom = pos_mask.float().sum(dim=1).clamp_min(1.0)
        losses.append((-(pos_mask.float() * log_prob).sum(dim=1) / denom).mean())

    if not losses:
        return torch.tensor(0.0, device=dev)
    return torch.stack(losses).mean()
