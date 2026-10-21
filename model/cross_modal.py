"""
Simplified ADCI interaction module: CrossModalAttention + strip path + scale path + gated fusion

Idea:
1. The CLIP text anchor feature (768-d) is the query and the DINOv2/CLIP visual patch features are keys and
   values; cross-attention returns the visual features aggregated by their relevance to the text.
2. Strip path: text-guided aggregation along rows and columns, which emphasises strip-like regions.
3. Scale path: text-guided aggregation at several scales, for defects of different sizes.
4. The paths are combined by learnable gates into a visual feature refined by the text semantics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Cross-attention with the text feature as query and the visual patch features as keys/values.

    Returns the visual feature vector aggregated by relevance to the text.

    Args:
        text_dim: width of the text feature (CLIP text = 768)
        visual_dim: width of the visual patch features (DINOv2/CLIP patch tokens)
        out_dim: output width (defaults to visual_dim)
        num_heads: number of attention heads
    """
    def __init__(self, text_dim=768, visual_dim=768, out_dim=None, num_heads=8, dropout=0.0):
        super().__init__()
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.out_dim = out_dim if out_dim is not None else visual_dim
        self.num_heads = num_heads

        # query projection: text feature -> multi-head queries
        self.q_proj = nn.Linear(text_dim, visual_dim)
        # key/value projections: visual patch features
        self.k_proj = nn.Linear(visual_dim, visual_dim)
        self.v_proj = nn.Linear(visual_dim, visual_dim)

        # multi-head attention
        head_dim = visual_dim // num_heads
        assert head_dim * num_heads == visual_dim, \
            f"visual_dim({visual_dim}) must be divisible by num_heads({num_heads})"
        self.scale = head_dim ** -0.5

        # output projection
        self.out_proj = nn.Linear(visual_dim, self.out_dim)
        self.norm = nn.LayerNorm(self.out_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, text_feat, visual_patches, return_weights=False):
        """Forward pass.

        Args:
            text_feat: [B, text_dim] CLIP text anchor feature
            visual_patches: [B, N, visual_dim] visual patch features
            return_weights: also return the attention weights

        Returns:
            out: [B, out_dim] text-refined visual feature
            (optional) attn_weights: [B, num_heads, N] attention weights
        """
        B, N, D = visual_patches.shape

        # query: text feature -> [B, 1, D]
        q = self.q_proj(text_feat).unsqueeze(1)  # [B, 1, D]

        # keys/values: visual patch features
        k = self.k_proj(visual_patches)  # [B, N, D]
        v = self.v_proj(visual_patches)  # [B, N, D]

        # split into heads
        q = q.reshape(B, 1, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)  # [B, H, 1, head_dim]
        k = k.reshape(B, N, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)  # [B, H, N, head_dim]
        v = v.reshape(B, N, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)  # [B, H, N, head_dim]

        # attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, 1, N]
        attn_weights = F.softmax(attn, dim=-1)  # [B, H, 1, N]

        # weighted aggregation
        out = (attn_weights @ v).squeeze(2)  # [B, H, head_dim]
        out = out.transpose(1, 2).reshape(B, D)  # [B, D]

        # output projection + LayerNorm (no text residual, so the output contains visual information only)
        out = self.out_proj(out)
        out = self.dropout(out)
        # the text residual was removed so that no text information leaks into the visual output
        out = self.norm(out)

        if return_weights:
            # attention weights averaged over heads [B, N]
            avg_attn = attn_weights.mean(dim=1).squeeze(1)  # [B, N]
            return out, avg_attn
        return out


class StripPath(nn.Module):
    """Strip path: text-guided attention aggregated along rows and columns.

    Following strip pooling, the feature map is pooled along the horizontal and the vertical direction, and text
    attention emphasises the strips that match the description (e.g. a thin elongated scratch).

    Args:
        text_dim: width of the text feature (768)
        visual_dim: width of the visual patch features
        out_dim: output width
        img_hw: side of the patch grid (e.g. 37 for 37 x 37 = 1369 patches)
        gate_type: gate type, 'heavy' (nn.Sequential) or 'scalar' (nn.Parameter)
    """
    def __init__(self, text_dim=768, visual_dim=768, out_dim=None, img_hw=37, gate_type='heavy'):
        super().__init__()
        self.visual_dim = visual_dim
        self.out_dim = out_dim if out_dim is not None else visual_dim
        self.img_hw = img_hw
        self.gate_type = gate_type

        # text-guided aggregation of horizontal strips
        self.horizontal_attn = CrossModalAttention(
            text_dim=text_dim, visual_dim=visual_dim, out_dim=visual_dim, num_heads=4
        )
        # text-guided aggregation of vertical strips
        self.vertical_attn = CrossModalAttention(
            text_dim=text_dim, visual_dim=visual_dim, out_dim=visual_dim, num_heads=4
        )

        # fuse horizontal and vertical strips (residual form: base + wh*(h - base) + wv*(v - base));
        # separate h/v gates instead of a single fusion gate, so either direction can be switched off;
        # with both gates at 0 the output is the mean and adds no spurious signal
        if gate_type == 'scalar':
            self.hv_weights = nn.Parameter(torch.full((2,), -2.0))  # two independent sigmoids, nearly closed at start
        else:
            self.h_gate = nn.Sequential(
                nn.Linear(visual_dim * 2, visual_dim),
                nn.Sigmoid()
            )
            self.v_gate = nn.Sequential(
                nn.Linear(visual_dim * 2, visual_dim),
                nn.Sigmoid()
            )
        self.fusion_proj = nn.Linear(visual_dim, self.out_dim)
        self.norm = nn.LayerNorm(self.out_dim)
        self._init_weights()

    def _init_weights(self):
        for m in [self.fusion_proj]:
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.gate_type != 'scalar':
            for gate in (self.h_gate, self.v_gate):
                for m in gate:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight, gain=0.1)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, -2.0)  # sigmoid(-2) ~ 0.12, nearly closed at start

    def _patches_to_map(self, patches):
        """[B, N, D] → [B, D, H, W]
        
        Uses self.img_hw for the spatial size, so that a token count that is not a perfect square does not break int(sqrt(N)).
        """
        B, N, D = patches.shape
        H = W = self.img_hw
        target_N = H * W
        if N != target_N:
            # N differs from H*W: pad or truncate first, then reshape
            patches_grid = patches.permute(0, 2, 1)  # [B, D, N]
            if N < target_N:
                pad = torch.zeros(B, D, target_N - N, device=patches.device, dtype=patches.dtype)
                patches_grid = torch.cat([patches_grid, pad], dim=2)
            else:
                patches_grid = patches_grid[:, :, :target_N]
            patches_grid = patches_grid.reshape(B, D, H, W)
            return patches_grid
        return patches.permute(0, 2, 1).reshape(B, D, H, W)

    def _map_to_patches(self, feat_map):
        """[B, D, H, W] → [B, N, D]"""
        B, D, H, W = feat_map.shape
        return feat_map.reshape(B, D, H * W).permute(0, 2, 1)

    def forward(self, text_feat, visual_patches):
        """Forward pass.

        Args:
            text_feat: [B, text_dim] text feature
            visual_patches: [B, N, visual_dim] visual patch features (N = H*W)

        Returns:
            strip_feat: [B, out_dim] strip-path feature
        """
        B, N, D = visual_patches.shape
        H = W = self.img_hw

        # reshape into a 2-D feature map
        feat_map = self._patches_to_map(visual_patches)  # [B, D, H, W]

        # --- horizontal: mean over each row -> horizontal strip features ---
        # average each row over W
        h_pooled = feat_map.mean(dim=3)  # [B, D, H], one feature per row
        h_pooled = h_pooled.permute(0, 2, 1)  # [B, H, D], H horizontal strips

        # text-guided selection among the H horizontal strips
        h_strip, h_attn = self.horizontal_attn(text_feat, h_pooled, return_weights=True)  # [B, D]

        # --- vertical: mean over each column -> vertical strip features ---
        v_pooled = feat_map.mean(dim=2)  # [B, D, W], one feature per column
        v_pooled = v_pooled.permute(0, 2, 1)  # [B, W, D], W vertical strips

        # text-guided selection among the W vertical strips
        v_strip, v_attn = self.vertical_attn(text_feat, v_pooled, return_weights=True)  # [B, D]

        # --- fuse horizontal and vertical strips (residual form) ---
        # no gate_sum normalisation: independent sigmoids, residual 0 when both are 0 (output = base);
        #    normalising would turn w = [0.01, 0.01] into [0.5, 0.5] and remove the identity mapping
        base = (h_strip + v_strip) / 2  # mean as the base
        if self.gate_type == 'scalar':
            w = torch.sigmoid(self.hv_weights)  # [2] independent sigmoids
            fused = base + w[0] * (h_strip - base) + w[1] * (v_strip - base)  # [B, D]
        else:
            cat_feat = torch.cat([h_strip, v_strip], dim=-1)  # [B, 2D]
            gate_h = self.h_gate(cat_feat)  # [B, D] Sigmoid
            gate_v = self.v_gate(cat_feat)  # [B, D] Sigmoid
            fused = base + gate_h * (h_strip - base) + gate_v * (v_strip - base)  # [B, D]

        # output projection
        strip_feat = self.fusion_proj(fused)
        strip_feat = self.norm(strip_feat)

        return strip_feat, (h_attn, v_attn)


class ScalePath(nn.Module):
    """Scale path: text-guided aggregation of the visual features at several scales.

    Pooling with kernels of different sizes captures defect regions of different sizes, and every scale is
    refined by text attention.

    Args:
        text_dim: width of the text feature (768)
        visual_dim: width of the visual patch features
        out_dim: output width
        img_hw: side of the patch grid
        scales: pooling kernel sizes
        gate_type: gate type, 'heavy' (nn.Sequential) or 'scalar' (nn.Parameter)
    """
    def __init__(self, text_dim=768, visual_dim=768, out_dim=None, img_hw=37, scales=None, gate_type='heavy'):
        super().__init__()
        self.visual_dim = visual_dim
        self.out_dim = out_dim if out_dim is not None else visual_dim
        self.img_hw = img_hw
        self.gate_type = gate_type

        if scales is None:
            scales = [1, 2, 4, 8]  # pooling kernel sizes
        self.scales = scales

        # one cross-attention module per scale
        self.scale_attns = nn.ModuleList([
            CrossModalAttention(
                text_dim=text_dim, visual_dim=visual_dim, out_dim=visual_dim, num_heads=4
            )
            for _ in scales
        ])

        # multi-scale fusion (residual form: base + sum_i w_i * (scale_i - base));
        # sigmoid instead of softmax: each w_i is independent and all can be 0 (output = mean)
        if gate_type == 'scalar':
            self.scale_weights = nn.Parameter(torch.full((len(scales),), -2.0))  # Sigmoid(-2)≈0.12
        else:
            self.scale_gate = nn.Sequential(
                nn.Linear(visual_dim * len(scales), visual_dim * len(scales)),
                nn.ReLU(),
                nn.Linear(visual_dim * len(scales), len(scales)),
                nn.Sigmoid()  # independent weights; all 0 gives the mean
            )
        self.fusion_proj = nn.Linear(visual_dim, self.out_dim)
        self.norm = nn.LayerNorm(self.out_dim)
        self._init_weights()

    def _init_weights(self):
        for m in [self.fusion_proj]:
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.gate_type != 'scalar':
            for m in self.scale_gate:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, -2.0)  # sigmoid(-2) ~ 0.12, nearly closed at start

    def _patches_to_map(self, patches):
        """[B, N, D] → [B, D, H, W]
        
        Uses self.img_hw for the spatial size.
        """
        B, N, D = patches.shape
        H = W = self.img_hw
        target_N = H * W
        if N != target_N:
            patches_grid = patches.permute(0, 2, 1)  # [B, D, N]
            if N < target_N:
                pad = torch.zeros(B, D, target_N - N, device=patches.device, dtype=patches.dtype)
                patches_grid = torch.cat([patches_grid, pad], dim=2)
            else:
                patches_grid = patches_grid[:, :, :target_N]
            patches_grid = patches_grid.reshape(B, D, H, W)
            return patches_grid
        return patches.permute(0, 2, 1).reshape(B, D, H, W)

    def _map_to_patches(self, feat_map):
        """[B, D, H, W] → [B, N, D]"""
        B, D, H, W = feat_map.shape
        return feat_map.reshape(B, D, H * W).permute(0, 2, 1)

    def forward(self, text_feat, visual_patches):
        """Forward pass.

        Args:
            text_feat: [B, text_dim] text feature
            visual_patches: [B, N, visual_dim] visual patch features

        Returns:
            scale_feat: [B, out_dim] scale-path feature
        """
        B, N, D = visual_patches.shape
        H = W = self.img_hw

        # reshape into a 2-D feature map
        feat_map = self._patches_to_map(visual_patches)  # [B, D, H, W]

        # multi-scale pooling + text attention
        scale_feats = []
        for i, kernel_size in enumerate(self.scales):
            # average pooling with a fixed kernel (avoids the non-deterministic backward of adaptive_avg_pool2d)
            ph = max(1, H // kernel_size)
            pw = max(1, W // kernel_size)
            pooled = F.avg_pool2d(feat_map, kernel_size, stride=kernel_size)
            # interpolate when the size does not match exactly
            if pooled.shape[-2] != ph or pooled.shape[-1] != pw:
                pooled = F.interpolate(pooled, size=(ph, pw), mode='bilinear', align_corners=False)

            # back to a patch sequence
            pooled_patches = self._map_to_patches(pooled)  # [B, ph*pw, D]

            # text-attention refinement
            scale_feat = self.scale_attns[i](text_feat, pooled_patches)  # [B, D]
            scale_feats.append(scale_feat)

        # stack the features of all scales
        scale_stack = torch.stack(scale_feats, dim=1)  # [B, num_scales, D]

        # learnable gated fusion (residual form: base + sum_i w_i * (scale_i - base));
        # no gate_sum normalisation: independent sigmoids, residual 0 when all are 0 (output = base)
        base = scale_stack.mean(dim=1)  # [B, D], mean as the base
        if self.gate_type == 'scalar':
            gate_weights = torch.sigmoid(self.scale_weights)  # [num_scales] independent sigmoids
            residuals = scale_stack - base.unsqueeze(1)  # [B, num_scales, D]
            fused = base + (residuals * gate_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        else:
            scale_cat = scale_stack.reshape(B, -1)  # [B, num_scales * D]
            gate_weights = self.scale_gate(scale_cat)  # [B, num_scales] Sigmoid
            residuals = scale_stack - base.unsqueeze(1)  # [B, num_scales, D]
            fused = base + (residuals * gate_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]

        # output projection
        scale_feat = self.fusion_proj(fused)
        scale_feat = self.norm(scale_feat)

        return scale_feat, gate_weights


class TextGuidedFusion(nn.Module):
    """Cross-modal gate activated by image-text cosine similarity (few-shot variant).

    Routes the paths by the image-text cosine similarity; with almost no parameters it is robust to over-fitting.

    Idea:
    - the gate does not learn when to open: the CLIP cosine similarity is the signal
    - sim_score = cos(clip_cls, text_feat) is a reliable proxy of image-text agreement
    - path_scales and path_biases only rescale this signal linearly
    - 6 learnable parameters, so over-fitting is very unlikely

    Gate: W_i = sigmoid(scale_i * sim_score + bias_i)
    - scale starts at 10.0: separates high and low image-text agreement
    - bias starts at -1.0: together with the small residual gate of the adapter it limits the initial perturbation
    - high sim_score -> large W_i -> path active
    - low sim_score -> W_i ~ 0 -> path silent, output reduces to the base

    Args:
        gate_type: gate type (backward compatible)
            'cosine'  - gate activated by cosine similarity (recommended for few-shot)
            'heavy'   - nn.Sequential gate network (earlier version, many parameters, prone to over-fitting)
            'scalar'  - nn.Parameter scalar weights (earlier version, no dynamic routing)
    """
    def __init__(self, text_dim=768, visual_dim=768, out_dim=768, img_hw=37, gate_type='cosine'):
        super().__init__()
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.out_dim = out_dim
        self.gate_type = gate_type

        # 1. direct cross-attention (baseline)
        self.direct_attn = CrossModalAttention(
            text_dim=text_dim, visual_dim=visual_dim, out_dim=visual_dim, num_heads=8
        )

        # 2. strip path (always scalar gates inside; sub-path gates do not need 'heavy')
        self.strip_path = StripPath(
            text_dim=text_dim, visual_dim=visual_dim, out_dim=visual_dim, img_hw=img_hw,
            gate_type='scalar',
        )

        # 3. scale path (always scalar gates inside; sub-path gates do not need 'heavy')
        self.scale_path = ScalePath(
            text_dim=text_dim, visual_dim=visual_dim, out_dim=visual_dim, img_hw=img_hw,
            gate_type='scalar',
        )

        # ========== gate parameters (by gate_type) ==========
        if gate_type == 'cosine':
            # few-shot variant: cosine-activated gate with only 6 learnable parameters.
            # gamma (residual_gate = 0.01) is already a first safeguard, so path_biases need not be aggressive:
            #   two stacked safeguards make the gradient so sparse that the gates never open,
            #   hence path_biases = -1.0 (not -2.0) and gamma carries the main safeguard
            self.path_scales = nn.Parameter(torch.ones(3) * 10.0)  # scale, separates high and low agreement
            self.path_biases = nn.Parameter(torch.ones(3) * -1.0)  # mild bias; gamma already safeguards
        elif gate_type == 'scalar':
            self.path_weights = nn.Parameter(torch.full((3,), -2.0))  # sigmoid(-2) ~ 0.12, nearly closed at start
            self.dual_weights = nn.Parameter(torch.full((2,), -2.0))
        else:  # heavy (earlier version, kept for backward compatibility)
            self.path_gate = nn.Sequential(
                nn.Linear(visual_dim * 4, visual_dim * 3),
                nn.ReLU(),
                nn.Linear(visual_dim * 3, 3),
                nn.Sigmoid()
            )
            self.dual_path_gate = nn.Sequential(
                nn.Linear(visual_dim * 3, visual_dim),
                nn.ReLU(),
                nn.Linear(visual_dim, 2),
                nn.Sigmoid()
            )

        # final output projection
        self.final_proj = nn.Linear(visual_dim, out_dim)
        self.final_norm = nn.LayerNorm(out_dim)

        self._init_weights()

    def _init_weights(self):
        for m in [self.final_proj]:
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.gate_type == 'heavy':
            for gate in (self.path_gate, self.dual_path_gate):
                for m in gate:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight, gain=0.1)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, -2.0)

    def forward(self, text_feat, visual_patches, clip_cls=None, mode='all'):
        """Forward pass with selectable paths.

        Args:
            text_feat: [B, text_dim] CLIP text feature
            visual_patches: [B, N, visual_dim] visual patch features
            clip_cls: [B, visual_dim] CLIP global feature (the input of the cosine gate)
            mode: path selection
                'all'      - all three paths with gated fusion (default)
                'direct'   - direct cross-attention only
                'strip'    - strip path only
                'scale'    - scale path only
                'ds'       - direct + strip (no scale)
                'dsc'      - direct + scale (no strip)
                'ss'       - strip + scale (no direct)
                'none'     - plain mean over patches (no text guidance; for testing)

        Returns:
            final_feat: [B, out_dim] text-refined visual feature
            aux: dict of auxiliary outputs (attention weights, gate weights, ...)
        """
        B = text_feat.shape[0]
        aux = {}

        if clip_cls is None:
            clip_cls = visual_patches.mean(dim=1)

        # ========== cosine similarity (cosine gate only) ==========
        if self.gate_type == 'cosine':
            clip_cls_norm = F.normalize(clip_cls, dim=-1)
            text_feat_norm = F.normalize(text_feat, dim=-1)
            sim_score = (clip_cls_norm * text_feat_norm).sum(dim=-1)  # [B]

        # ========== paths (as needed) ==========
        direct_feat = None
        strip_feat = None
        scale_feat = None

        if mode in ('all', 'direct', 'ds', 'dsc'):
            direct_feat, attn_weights = self.direct_attn(
                text_feat, visual_patches, return_weights=True
            )
            aux['attn_weights'] = attn_weights

        if mode in ('all', 'strip', 'ds', 'ss'):
            strip_feat, (h_attn, v_attn) = self.strip_path(text_feat, visual_patches)
            aux['h_attn'] = h_attn
            aux['v_attn'] = v_attn

        if mode in ('all', 'scale', 'dsc', 'ss'):
            scale_feat, scale_weights = self.scale_path(text_feat, visual_patches)
            aux['scale_weights'] = scale_weights

        # ========== fusion ==========
        if mode == 'none':
            fused = visual_patches.mean(dim=1)

        elif mode == 'direct':
            fused = direct_feat

        elif mode == 'strip':
            fused = strip_feat

        elif mode == 'scale':
            fused = scale_feat

        elif mode in ('ds', 'dsc', 'ss'):
            # two-path modes
            if mode == 'ds':
                f1, f2 = direct_feat, strip_feat
            elif mode == 'dsc':
                f1, f2 = direct_feat, scale_feat
            else:  # ss
                f1, f2 = strip_feat, scale_feat

            base = clip_cls if clip_cls is not None else (f1 + f2) / 2

            if self.gate_type == 'cosine':
                w1 = torch.sigmoid(self.path_scales[0] * sim_score + self.path_biases[0]).unsqueeze(-1)
                w2 = torch.sigmoid(self.path_scales[1] * sim_score + self.path_biases[1]).unsqueeze(-1)
                fused = base + w1 * (f1 - base) + w2 * (f2 - base)
                aux['path_gate'] = torch.cat([w1, w2], dim=-1)
            elif self.gate_type == 'scalar':
                w = torch.sigmoid(self.dual_weights)
                fused = base + w[0] * (f1 - base) + w[1] * (f2 - base)
            else:
                gate_2 = self.dual_path_gate(torch.cat([f1, f2, base], dim=-1))
                fused = base + gate_2[:, 0:1] * (f1 - base) + gate_2[:, 1:2] * (f2 - base)

        else:  # 'all' (default)
            base = clip_cls if clip_cls is not None else (direct_feat + strip_feat + scale_feat) / 3

            if self.gate_type == 'cosine':
                # image-text cosine similarity mapped directly to the gate weights of the 3 paths:
                # W_i = sigmoid(scale_i * sim_score + bias_i);
                # a path is strongly active only when the image-text similarity is high enough
                w_direct = torch.sigmoid(self.path_scales[0] * sim_score + self.path_biases[0]).unsqueeze(-1)
                w_strip  = torch.sigmoid(self.path_scales[1] * sim_score + self.path_biases[1]).unsqueeze(-1)
                w_scale  = torch.sigmoid(self.path_scales[2] * sim_score + self.path_biases[2]).unsqueeze(-1)
                aux['path_gate'] = torch.cat([w_direct, w_strip, w_scale], dim=-1)
                fused = base + (
                    w_direct * (direct_feat - base)
                    + w_strip * (strip_feat - base)
                    + w_scale * (scale_feat - base)
                )
            elif self.gate_type == 'scalar':
                gate = torch.sigmoid(self.path_weights)  # [3]
                aux['path_gate'] = gate.unsqueeze(0).expand(B, -1)
                fused = base + (
                    gate[0] * (direct_feat - base)
                    + gate[1] * (strip_feat - base)
                    + gate[2] * (scale_feat - base)
                )
            else:  # heavy
                path_cat = torch.cat([direct_feat, strip_feat, scale_feat, base], dim=-1)
                gate = self.path_gate(path_cat)
                aux['path_gate'] = gate
                fused = base + (
                    gate[:, 0:1] * (direct_feat - base)
                    + gate[:, 1:2] * (strip_feat - base)
                    + gate[:, 2:3] * (scale_feat - base)
                )

        # final projection
        final_feat = self.final_proj(fused)
        final_feat = self.final_norm(final_feat)

        return final_feat, aux



# =============================================================================
# Utility: farthest point sampling (FPS) for patch selection
# =============================================================================

def farthest_point_sampling(points, coords, n_samples):
    """Farthest point sampling: the n_samples points of `points` that are spread most evenly.

    Args:
        points: [N, D] candidate patch features (N candidates)
        coords: [N, 2] patch coordinates (x, y) normalised to [0, 1]
        n_samples: number of points to sample

    Returns:
        selected_indices: [n_samples] indices of the selected patches
    """
    N = points.shape[0]
    if n_samples >= N:
        return torch.arange(N, device=points.device)

    device = points.device
    # initialisation: random first point
    selected = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.ones(N, device=device) * 1e10

    # random starting point
    selected[0] = torch.randint(0, N, (1,), device=device)

    for i in range(1, n_samples):
        # distance of every point to the nearest selected point
        last = selected[i - 1]
        dist = torch.norm(coords - coords[last], dim=-1)  # [N]
        distances = torch.min(distances, dist)  # [N]
        # take the farthest point
        selected[i] = torch.argmax(distances)

    return selected


def farthest_point_sampling_from_topk(patch_features, patch_coords, attn_scores, topk_ratio=0.3, fps_ratio=0.5):
    """From the top-K patches, keep the K/2 that are spread most evenly.

    Args:
        patch_features: [N, D] all patch features (N = number of patches in the grid)
        patch_coords: [N, 2] patch coordinates
        attn_scores: [N] attention scores
        topk_ratio: share of patches kept by the top-K step
        fps_ratio: share kept by farthest point sampling (relative to K)

    Returns:
        selected_features: [n_selected, D] selected patch features
        selected_indices: [n_selected] indices of the selected patches
    """
    N = patch_features.shape[0]

    # 1. top-K by attention score
    K = max(1, int(N * topk_ratio))
    _, topk_indices = torch.topk(attn_scores, K)

    topk_features = patch_features[topk_indices]  # [K, D]
    topk_coords = patch_coords[topk_indices]  # [K, 2]

    # 2. farthest point sampling: K/2 of the K
    n_fps = max(1, int(K * fps_ratio))
    fps_indices = farthest_point_sampling(topk_features, topk_coords, n_fps)

    selected_features = topk_features[fps_indices]  # [n_fps, D]
    selected_indices = topk_indices[fps_indices]  # [n_fps]

    return selected_features, selected_indices
