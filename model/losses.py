"""
Supervised contrastive loss (SupConLoss).
Reference: https://github.com/HobbitLong/SupContrast
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning loss.

    Args:
        temperature: temperature of the contrastive loss (default: 0.07)
        base_temperature: base temperature (default: 0.07)
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels, exclude_labels=None):
        """SupCon loss.

        Args:
            features: [B, D] normalised features
            labels: [B] class labels
            exclude_labels: set or list of class ids to isolate (e.g. the good class);
                            samples of these classes take part in neither attraction nor repulsion

        Returns:
            scalar loss
        """
        device = features.device
        batch_size = features.shape[0]

        # normalise the features
        features = F.normalize(features, dim=-1)

        # participation mask: exclude the classes in exclude_labels
        if exclude_labels is not None:
            exclude_set = set(exclude_labels)
            participate = torch.tensor(
                [l.item() not in exclude_set for l in labels],
                dtype=torch.bool, device=device
            )  # [B]
        else:
            participate = torch.ones(batch_size, dtype=torch.bool, device=device)

        # no participating sample in the batch: return 0
        if not participate.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # keep only the participating samples
        features = features[participate]  # [B', D]
        labels_f = labels[participate]    # [B']
        batch_size_f = features.shape[0]

        # cosine similarity matrix [B', B']
        sim_matrix = features @ features.T

        # subtract the maximum for numerical stability
        sim_matrix = sim_matrix / self.temperature
        sim_matrix = sim_matrix - sim_matrix.max(dim=-1, keepdim=True).values.detach()

        # positive mask: 1 for equal labels
        labels_f = labels_f.unsqueeze(1)  # [B', 1]
        mask = torch.eq(labels_f, labels_f.T).float().to(device)  # [B', B']

        # remove self-pairs (diagonal)
        logits_mask = ~torch.eye(batch_size_f, dtype=torch.bool, device=device)
        mask = mask * logits_mask

        # denominator: all negatives + positives (except self)
        exp_logits = torch.exp(sim_matrix) * logits_mask  # [B', B']
        sum_exp = exp_logits.sum(dim=-1, keepdim=True)  # [B', 1]

        # loss per sample
        pos_mask = mask  # [B', B']
        pos_exp = (exp_logits * pos_mask).sum(dim=-1)  # [B']
        num_pos = pos_mask.sum(dim=-1)  # [B']

        # avoid division by zero
        loss = -torch.log(pos_exp / (sum_exp.squeeze(-1) + 1e-8) + 1e-8)
        loss = loss * (num_pos > 0).float()
        loss = loss.sum() / (num_pos.sum() + 1e-8)

        loss = loss * self.base_temperature
        return loss
