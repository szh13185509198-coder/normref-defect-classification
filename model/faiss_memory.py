"""Learnable equal-capacity class-prototype memory with FAISS retrieval."""

from __future__ import annotations

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FaissPrototypeMemory(nn.Module):
    """Store one or more prototypes for every class.

    ``prototypes_per_class=1`` deliberately keeps the historical ``[K, D]``
    parameter shape, so old checkpoints remain loadable without conversion.
    Multi-prototype checkpoints use ``[K, M, D]``.  In both training and
    inference, a class score is aggregated over its exactly ``M`` prototypes.
    ``max`` preserves P5.0 and all historical single-prototype behavior;
    ``logmeanexp`` is the normalized smooth mixture registered for P5.1.
    Consequently, no class receives an index-cardinality advantage.
    """

    def __init__(
        self,
        num_classes,
        feat_dim=1152,
        orth_margin=0.3,
        init_logit_scale=10.0,
        prototype_init_mode="zero",
        prototypes_per_class=1,
        prototype_aggregation="max",
        prototype_mixture_temperature=0.2,
    ):
        super().__init__()
        if int(prototypes_per_class) < 1:
            raise ValueError("prototypes_per_class must be positive")
        if prototype_aggregation not in {"max", "logmeanexp"}:
            raise ValueError(
                "prototype_aggregation must be 'max' or 'logmeanexp'; got "
                f"{prototype_aggregation!r}"
            )
        if float(prototype_mixture_temperature) <= 0:
            raise ValueError("prototype_mixture_temperature must be positive")

        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.prototypes_per_class = int(prototypes_per_class)
        self.prototype_aggregation = prototype_aggregation
        self.prototype_mixture_temperature = float(prototype_mixture_temperature)
        self.prototype_init_mode = prototype_init_mode
        self.orth_margin = orth_margin
        self.faiss_index = None
        self._faiss_class_ids = None
        self._text_refined_prototypes = None

        shape = (
            (self.num_classes, self.feat_dim)
            if self.prototypes_per_class == 1
            else (self.num_classes, self.prototypes_per_class, self.feat_dim)
        )
        if prototype_init_mode == "zero":
            initial_prototypes = torch.zeros(*shape)
        elif prototype_init_mode == "legacy_random":
            initial_prototypes = torch.randn(*shape)
        else:
            raise ValueError(
                f"Unsupported prototype_init_mode: {prototype_init_mode}"
            )
        self.prototypes = nn.Parameter(initial_prototypes)
        self.logit_scale = nn.Parameter(
            torch.ones(1) * np.log(float(init_logit_scale))
        )

    def prototype_sets(self):
        """Return the learnable memory in the canonical ``[K, M, D]`` shape."""
        if self.prototypes.ndim == 2:
            return self.prototypes.unsqueeze(1)
        return self.prototypes

    def normalized_prototype_sets(self):
        return F.normalize(self.prototype_sets(), dim=-1)

    def class_representatives(self):
        """Return normalized class centroids for diagnostics/visualization."""
        return F.normalize(self.normalized_prototype_sets().mean(dim=1), dim=-1)

    def aggregate_prototype_scores(self, prototype_scores):
        """Collapse ``[..., class, prototype]`` scores without count bias."""
        if prototype_scores.shape[-1] != self.prototypes_per_class:
            raise ValueError(
                "prototype score count does not match prototypes_per_class: "
                f"{prototype_scores.shape[-1]} != {self.prototypes_per_class}"
            )
        if self.prototype_aggregation == "max":
            return prototype_scores.max(dim=-1).values
        temperature = self.prototype_mixture_temperature
        return temperature * (
            torch.logsumexp(prototype_scores / temperature, dim=-1)
            - np.log(float(self.prototypes_per_class))
        )

    def true_class_responsibilities(
        self, features, labels, assignment_temperature=0.2
    ):
        """Return soft assignments to the prototypes of each sample's class."""
        if float(assignment_temperature) <= 0:
            raise ValueError("assignment_temperature must be positive")
        features = F.normalize(features, dim=-1)
        sets = self.normalized_prototype_sets()
        labels = labels.to(device=features.device, dtype=torch.long)
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("labels must be a [batch] tensor aligned with features")
        selected = sets.index_select(0, labels)
        scores = torch.einsum("bd,bmd->bm", features, selected)
        return torch.softmax(scores / float(assignment_temperature), dim=-1)

    def prototype_balance_loss(
        self, features, labels, assignment_temperature=0.2
    ):
        """Encourage equal effective capacity without collapsing each sample.

        Only classes represented by at least two samples in the current batch
        participate.  The loss combines low per-sample assignment entropy with
        a uniform class-average responsibility.  This mutual-information form
        rejects both monopoly and the trivial all-prototypes-identical solution.
        """
        if self.prototypes_per_class == 1:
            return features.sum() * 0.0
        responsibilities = self.true_class_responsibilities(
            features, labels, assignment_temperature
        )
        losses = []
        for class_id in labels.unique(sorted=True):
            mask = labels == class_id
            if int(mask.sum()) < 2:
                continue
            mean_responsibility = responsibilities[mask].mean(dim=0)
            mean_responsibility = mean_responsibility.clamp_min(1e-8)
            class_responsibilities = responsibilities[mask].clamp_min(1e-8)
            sample_entropy = -torch.sum(
                class_responsibilities * torch.log(class_responsibilities),
                dim=-1,
            ).mean()
            marginal_kl = torch.sum(
                mean_responsibility
                * torch.log(
                    mean_responsibility * float(self.prototypes_per_class)
                )
            )
            losses.append(
                sample_entropy + marginal_kl
            )
        if not losses:
            return features.sum() * 0.0
        return torch.stack(losses).mean()

    def forward(self, features):
        features = F.normalize(features, dim=-1)
        if self._text_refined_prototypes is not None:
            if self.prototypes_per_class != 1:
                raise RuntimeError(
                    "Text-refined prototypes are not defined for P5.0 multi-prototype memory"
                )
            sets = self._text_refined_prototypes.unsqueeze(1)
        else:
            sets = self.normalized_prototype_sets()
        prototype_scores = torch.einsum("bd,kmd->bkm", features, sets)
        class_scores = self.aggregate_prototype_scores(prototype_scores)
        return self.logit_scale.exp().clamp(max=33.0) * class_scores

    def orthogonal_loss(self):
        protos = self.class_representatives()
        sim = protos @ protos.T
        mask = ~torch.eye(self.num_classes, device=sim.device).bool()
        return torch.clamp(sim[mask] - self.orth_margin, min=0).mean()

    def set_text_refined_prototypes(self, refined_prototypes):
        if self.prototypes_per_class != 1:
            raise RuntimeError(
                "Text refinement is incompatible with multi-prototype P5.0"
            )
        self._text_refined_prototypes = refined_prototypes

    def clear_text_refined_prototypes(self):
        self._text_refined_prototypes = None

    def build_faiss_index(self, use_text_refined=False, model=None, text_prototypes=None):
        if use_text_refined:
            if self.prototypes_per_class != 1:
                raise RuntimeError(
                    "Text-refined FAISS indexing is incompatible with multi-prototype P5.0"
                )
            if model is None or text_prototypes is None:
                raise ValueError(
                    "model and text_prototypes are required for text-refined indexing"
                )
            with torch.no_grad():
                refined = model.refine_prototypes_with_text(
                    self.prototypes, text_prototypes
                )
            sets = refined.detach().unsqueeze(1)
        else:
            sets = self.prototype_sets().detach()

        protos_np = sets.reshape(-1, self.feat_dim).cpu().numpy().astype(np.float32)
        faiss.normalize_L2(protos_np)
        self.faiss_index = faiss.IndexFlatIP(self.feat_dim)
        self.faiss_index.add(protos_np)
        self._faiss_class_ids = np.repeat(
            np.arange(self.num_classes, dtype=np.int64),
            self.prototypes_per_class,
        )

    def search(self, features, k=1):
        """Return top-k classes after the configured within-class aggregation."""
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not built. Call build_faiss_index() first.")
        if not 1 <= int(k) <= self.num_classes:
            raise ValueError(f"k must be in [1, {self.num_classes}]")
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        features = np.asarray(features, dtype=np.float32).copy()
        faiss.normalize_L2(features)

        # Query every prototype, then collapse to one score per class before
        # ranking.  Returning flat prototype IDs would silently bias retrieval.
        flat_d, flat_i = self.faiss_index.search(
            features, self.num_classes * self.prototypes_per_class
        )
        prototype_scores = np.full(
            (
                features.shape[0],
                self.num_classes,
                self.prototypes_per_class,
            ),
            -np.inf,
            dtype=np.float32,
        )
        for row in range(features.shape[0]):
            flat_ids = flat_i[row]
            class_ids = flat_ids // self.prototypes_per_class
            prototype_ids = flat_ids % self.prototypes_per_class
            prototype_scores[row, class_ids, prototype_ids] = flat_d[row]
        if self.prototype_aggregation == "max":
            class_scores = prototype_scores.max(axis=-1)
        else:
            temperature = self.prototype_mixture_temperature
            scaled = prototype_scores / temperature
            maxima = scaled.max(axis=-1, keepdims=True)
            class_scores = temperature * (
                maxima.squeeze(-1)
                + np.log(np.exp(scaled - maxima).mean(axis=-1))
            )
        order = np.argsort(-class_scores, axis=1, kind="stable")[:, : int(k)]
        scores = np.take_along_axis(class_scores, order, axis=1)
        return scores, order.astype(np.int64, copy=False)

    @torch.no_grad()
    def predict(self, features):
        logits = self.forward(features)
        return logits.argmax(dim=-1), logits.max(dim=-1).values
