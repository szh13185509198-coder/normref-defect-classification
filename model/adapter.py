import torch
from torch import nn
import torch.nn.functional as F
from .adapter_modules import SimpleAdapter, SimpleProj
from .dino_extractor import Dinov2Extractor
from .cross_modal import (
    TextGuidedFusion,
    CrossModalAttention,
    farthest_point_sampling,
    farthest_point_sampling_from_topk,
)


class AdaptedCLIP(nn.Module):
    def __init__(
        self,
        clip_model,
        text_adapt_weight: float = 0.1,
        image_adapt_weight: float = 0.2,
        text_adapt_until: int = 3,
        image_adapt_until: int = 6,
        levels: list = [6, 12, 18, 24],
        relu: bool = True,
        dino_use_hub_first: bool = None,
        cross_modal_dim: int = 768,
        img_hw: int = None,  # side of the patch grid; computed from img_size when None
        img_size: int = 518,  # input image size, used to compute img_hw
        gate_type: str = 'cosine',  # gate type: 'cosine' (recommended), 'heavy' (nn.Sequential), 'scalar' (nn.Parameter)
        proto_text_refine: str = 'none',  # text refinement: 'none' / 'logit_boost' / 'proto_refine'
        dino_mask_ratio: float = 0.0,     # share of DINOv2 patches used by the masked Gram pooling (0.0 = off)
        clip_pool: str = 'topk',
        dino_pool: str = 'gram',
        dino_patch_policy: str = 'full',
        dino_strict_backend: bool = False,
        dino_hub_repo: str = None,
        dino_hub_checkpoint: str = None,
        rng_protocol: str = 'isolated_rng',
        adapter_train_mode: str = 'legacy_random_fixed',
        normal_aware_head: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.text_adapt_until = text_adapt_until
        self.image_adapt_until = image_adapt_until
        self.t_w = text_adapt_weight
        self.i_w = image_adapt_weight
        self.levels = levels
        # img_hw = img_size / patch_size (14)
        if img_hw is None:
            img_hw = img_size // 14
        self.img_hw = img_hw
        self.cross_modal_dim = cross_modal_dim
        self.dino_mask_ratio = dino_mask_ratio
        if clip_pool not in {'mean', 'topk'}:
            raise ValueError(f"Unsupported clip_pool: {clip_pool}")
        if dino_pool not in {'cls', 'mean', 'gram'}:
            raise ValueError(f"Unsupported dino_pool: {dino_pool}")
        self.clip_pool = clip_pool
        self.dino_pool = dino_pool
        self.fusion_dim = 768 + (1152 if dino_pool == 'gram' else 384)
        if adapter_train_mode not in {
            'legacy_random_fixed', 'terminal_only', 'frozen_random_early',
            'learned_early'
        }:
            raise ValueError(f"Unsupported adapter_train_mode: {adapter_train_mode}")
        self.adapter_train_mode = adapter_train_mode
        self.normal_aware_head_enabled = bool(normal_aware_head)

        # DINOv2 branch (frozen). Building the DINO backend consumes a backend-dependent number of CPU RNG draws;
        # isolated_rng restores the RNG state from before the construction, so the adapter initialisation that
        # follows does not depend on Hub/Transformers internals.
        if rng_protocol not in {'isolated_rng', 'legacy_rng'}:
            raise ValueError(f"Unsupported RNG protocol: {rng_protocol}")
        self.rng_protocol = rng_protocol

        def build_dino_extractor():
            return Dinov2Extractor(
                use_hub_first=dino_use_hub_first,
                patch_policy=dino_patch_policy,
                strict_backend=dino_strict_backend,
                hub_repo_path=dino_hub_repo,
                hub_checkpoint_path=dino_hub_checkpoint,
            )

        if rng_protocol == 'isolated_rng':
            # DINO currently constructs on CPU. If future construction moves to
            # CUDA, the corresponding CUDA device IDs must be added here.
            with torch.random.fork_rng(devices=[]):
                self.dino_extractor = build_dino_extractor()
        else:
            self.dino_extractor = build_dino_extractor()
        self.vis_dim = 768 + 384  # CLIP(768) + DINOv2(384)

        # CLIP patch token width = 1024 (ViT-L)
        # DINOv2 patch width = 384 (dinov2_vits14)
        self.clip_patch_dim = 1024  # CLIP ViT-L hidden dim
        self.dino_patch_dim = 384   # DINOv2 ViT-S/14

        # ============================================================
        # Text-guided cross-modal fusion (simplified ADCI)
        # Not created for proto_text_refine, which does not use it (saves GPU memory)
        # ============================================================
        self.proto_text_refine = proto_text_refine
        if not proto_text_refine or proto_text_refine == 'none':
            # clip_patch_feats are projected to 768, so the concatenated width is 768 + dino_dim
            self.patch_fusion_proj = nn.Linear(
                768 + self.dino_patch_dim, cross_modal_dim
            )
            self.text_guided_fusion = TextGuidedFusion(
                text_dim=768,
                visual_dim=cross_modal_dim,
                out_dim=cross_modal_dim,
                img_hw=img_hw,
                gate_type=gate_type,
            )

            # ============================================================
            # ControlNet-style gate gamma
            # Initialised at 0.01 rather than 0.0, so that the classification gradient reaches text_guided_fusion:
            # ============================================================
            # with gamma = 0, dL/d(residual) = gamma * dL/dZ = 0 and the text module receives no classification
            # gradient; it would be driven by the ITC loss alone and drift away from the visual feature manifold.
            # With gamma = 0.01 a weak classification gradient trains the text module towards useful refinement.
            self.residual_gate = nn.Parameter(torch.tensor([0.01]))

        # ============================================================
        # Text refinement (two modes; the visual features are never modified by text)
        # logit_boost: temperature softmax over image-text similarities, added in logit space
        #     final_logits = vision_logits + β * logit_scale * softmax(text_sim / τ)
        # proto_refine: refinement of the class prototypes in feature space
        #     refined_proto = L2_norm(visual_proto + β * text_proj(text_proto))
        # ============================================================
        self.proto_text_refine = proto_text_refine
        if proto_text_refine == 'logit_boost':
            # tau: temperature, initialised at 0.07 (the original CLIP logit scale)
            # smaller tau -> sharper softmax -> stronger winner-take-all
            self.text_temp = nn.Parameter(torch.tensor(0.07))
            # beta: residual gate of the text logits, starts at 0.01
            self.text_logit_gate = nn.Parameter(torch.tensor([0.01]))
        elif proto_text_refine == 'proto_refine':
            # Text prototype projection 768 -> 1920 (cross-modal mapping; LayerNorm + GELU stabilise the variance)
            self.text_proto_proj = nn.Sequential(
                nn.Linear(768, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Linear(1024, 1920),  # 768 (CLIP) + 1152 (DINOv2 Gram) = 1920
            )
            # ControlNet-style zero initialisation of the last layer only, so the initial residual is close to 0
            nn.init.xavier_uniform_(self.text_proto_proj[-1].weight, gain=0.01)
            nn.init.zeros_(self.text_proto_proj[-1].bias)
            # beta: residual gate of the prototypes, starts at 0.01
            self.proto_text_gate = nn.Parameter(torch.tensor([0.01]))

        # ============================================================
        # Original adapter layers (kept for backward compatibility)
        # ============================================================
        layer_adapters = nn.ModuleList(
            [SimpleAdapter(1024, 1024) for _ in range(image_adapt_until)]
        )
        seg_proj = nn.ModuleList(
            [SimpleProj(1024, 768, relu) for _ in range(len(levels))]
        )
        det_proj = SimpleProj(1024, 768, relu)
        self.image_adapter = nn.ModuleDict(
            {
                "layer_adapters": layer_adapters,
                "seg_proj": seg_proj,
                "det_proj": det_proj,
            }
        )
        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(768, 768) for _ in range(text_adapt_until)]
            + [SimpleProj(768, 768, relu=True)]
        )

        # ============================================================
        # Patch projection: CLIP patch tokens -> 768 (for cross-attention)
        # ============================================================
        self.clip_patch_proj = nn.Linear(1024, 768)

        self._init_weights_()

        # P2.3 hierarchical normal-aware branch.  Construct this after all
        # historical modules have been initialized so enabling the branch does
        # not perturb their RNG trajectory.  A zero start removes another
        # initialization lottery: the initial binary logit is exactly 0 and the
        # auxiliary loss must learn every deviation from the neutral boundary.
        self.normal_aware_head = None
        if self.normal_aware_head_enabled:
            # nn.Linear normally consumes CPU RNG even when its values are
            # overwritten immediately.  Preserve the caller's RNG state so
            # enabling this optional branch cannot move any downstream legacy
            # initialization or data trajectory.
            with torch.random.fork_rng(devices=[]):
                self.normal_aware_head = nn.Linear(self.fusion_dim, 1)
                nn.init.zeros_(self.normal_aware_head.weight)
                nn.init.zeros_(self.normal_aware_head.bias)

    def train(self, mode: bool = True):
        """Keep frozen backbones deterministic for clean adapter protocols."""
        super().train(mode)
        if mode and self.adapter_train_mode != 'legacy_random_fixed':
            self.clipmodel.eval()
            self.dino_extractor.eval()
        return self

    def _apply_layer_adapter(self, tokens, layer_index):
        adapt_out = self.image_adapter["layer_adapters"][layer_index](tokens)
        adapt_out = (
            adapt_out
            * tokens.norm(dim=-1, keepdim=True)
            / adapt_out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        )
        return self.i_w * adapt_out + (1 - self.i_w) * tokens

    def _run_clip_blocks(self, tokens):
        """Run CLIP blocks according to the explicit adapter gradient mode."""
        if self.adapter_train_mode == 'terminal_only':
            with torch.no_grad():
                for i in range(24):
                    tokens, _ = self.image_encoder.transformer.resblocks[i](
                        tokens, attn_mask=None
                    )
            return tokens

        if self.adapter_train_mode == 'frozen_random_early':
            # Clean causal control: use the same isolated-RNG random adapters
            # as the corresponding terminal_only seed, but keep the entire
            # CLIP + early-adapter feature extractor frozen. Only downstream
            # trainable modules (for example det_proj) receive gradients.
            with torch.no_grad():
                for i in range(24):
                    tokens, _ = self.image_encoder.transformer.resblocks[i](
                        tokens, attn_mask=None
                    )
                    if i < self.image_adapt_until:
                        tokens = self._apply_layer_adapter(tokens, i)
            return tokens

        for i in range(self.image_adapt_until):
            tokens, _ = self.image_encoder.transformer.resblocks[i](
                tokens, attn_mask=None
            )
            tokens = self._apply_layer_adapter(tokens, i)

        if self.adapter_train_mode == 'legacy_random_fixed':
            with torch.no_grad():
                for i in range(self.image_adapt_until, 24):
                    tokens, _ = self.image_encoder.transformer.resblocks[i](
                        tokens, attn_mask=None
                    )
        else:
            for i in range(self.image_adapt_until, 24):
                tokens, _ = self.image_encoder.transformer.resblocks[i](
                    tokens, attn_mask=None
                )
        return tokens

    def _init_weights_(self):
        for p in self.image_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.text_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for proj in [self.clip_patch_proj]:
            for p in proj.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.proto_text_refine in ('none', '', None) and hasattr(self, 'patch_fusion_proj'):
            for p in self.patch_fusion_proj.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward_normal_aware(self, fusion_features):
        """Return one defect-vs-good logit per fused visual feature.

        Positive logits mean defect and non-positive logits mean good.  The
        input is the same L2-normalized terminal feature used by the multiclass
        prototype branch, so the auxiliary loss can shape only the already
        authorized terminal adapter graph.
        """
        if self.normal_aware_head is None:
            raise RuntimeError("normal-aware head is disabled for this model")
        if fusion_features.ndim != 2 or fusion_features.shape[-1] != self.fusion_dim:
            raise ValueError(
                "normal-aware head expects [B, fusion_dim] features, got "
                f"{tuple(fusion_features.shape)}"
            )
        return self.normal_aware_head(fusion_features).squeeze(-1)


    def encode_image(self, image):
        """Fused CLIP + DINOv2 features (no gradient: the backbones are frozen)."""
        with torch.no_grad():
            clip_vis_feat, _ = self.clipmodel.visual(image, [])
            dino_feat = self.dino_extractor(image)
        fusion_feat = torch.cat([clip_vis_feat, dino_feat], dim=-1)
        return fusion_feat

    def forward(self, x):
        """Full forward pass (inference); returns seg_tokens and det_token."""
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)
        tokens = []
        for i in range(24):
            x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            if (
                i < self.image_adapt_until
                and self.adapter_train_mode != 'terminal_only'
            ):
                x = self._apply_layer_adapter(x, i)
            if i + 1 in self.levels:
                tokens.append(x[1:, :, :])
        x = x.permute(1, 0, 2)
        tokens = [t.permute(1, 0, 2) for t in tokens]
        tokens = [self.image_encoder.ln_post(t) for t in tokens]
        seg_tokens = [self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)]
        seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens]
        det_token = self.image_adapter["det_proj"](tokens[-1])
        det_token = F.normalize(det_token, dim=-1).mean(1)
        return seg_tokens, det_token

    def forward_det(self, x):
        """Memory-efficient training forward pass that returns only det_token.

        The forward and gradient semantics of the adapters are set explicitly by adapter_train_mode;
        terminal_only and frozen_random_early keep no backbone activations,
        learned_early keeps the graph that leads to the early adapters.
        """
        # --- patch embedding + positional encoding (no gradient) ---
        with torch.no_grad():
            x = self.image_encoder.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            x = torch.cat(
                [
                    self.image_encoder.class_embedding.to(x.dtype)
                    + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x,
                ],
                dim=1,
            )
            x = x + self.image_encoder.positional_embedding.to(x.dtype)
            x = self.image_encoder.patch_dropout(x)
            x = self.image_encoder.ln_pre(x)
            x = x.permute(1, 0, 2)

        # CLIP blocks and early adapters, run in the explicit mode
        x = self._run_clip_blocks(x)

        # det_token
        det_token = x.permute(1, 0, 2)  # [B, N, D]
        det_token = self.image_encoder.ln_post(det_token)
        det_token = self.image_adapter["det_proj"](det_token)
        det_token = F.normalize(det_token, dim=-1).mean(1)  # [B, 768]
        return det_token

    def _extract_clip_patch_tokens(self, x):
        """CLIP patch tokens (without the CLS token).

        Reuses the logic of forward_det and returns all patch tokens.

        Args:
            x: [B, 3, H, W] input images

        Returns:
            clip_patch_feats: [B, N, 768] projected patch features
            all_patch_feats: [B, N, 1024] original 1024-d patch features
        """
        # --- patch embedding + positional encoding (no gradient) ---
        with torch.no_grad():
            x_embed = self.image_encoder.conv1(x)
            x_embed = x_embed.reshape(x_embed.shape[0], x_embed.shape[1], -1)
            x_embed = x_embed.permute(0, 2, 1)
            x_embed = torch.cat(
                [
                    self.image_encoder.class_embedding.to(x_embed.dtype)
                    + torch.zeros(x_embed.shape[0], 1, x_embed.shape[-1], dtype=x_embed.dtype, device=x_embed.device),
                    x_embed,
                ],
                dim=1,
            )
            x_embed = x_embed + self.image_encoder.positional_embedding.to(x_embed.dtype)
            x_embed = self.image_encoder.patch_dropout(x_embed)
            x_embed = self.image_encoder.ln_pre(x_embed)
            x_embed = x_embed.permute(1, 0, 2)

        # CLIP blocks and early adapters, run in the explicit mode
        x_embed = self._run_clip_blocks(x_embed)

        # all tokens [B, N+1, 1024]
        all_tokens = x_embed.permute(1, 0, 2)  # [B, N+1, 1024]
        all_tokens = self.image_encoder.ln_post(all_tokens)  # LayerNorm

        # drop the CLS token, keep the patch tokens [B, N, 1024]
        patch_tokens = all_tokens[:, 1:, :]  # [B, N, 1024]

        # project to 768 for cross-attention
        clip_patch_feats = self.clip_patch_proj(patch_tokens)  # [B, N, 768]

        return clip_patch_feats, patch_tokens

    def _extract_dino_patch_tokens(self, x):
        """DINOv2 patch tokens (without the CLS token).

        Args:
            x: [B, 3, H, W] input images

        Returns:
            dino_patch_feats: [B, N, 384] DINOv2 patch features
        """
        with torch.no_grad():
            dino_patches = self.dino_extractor.forward_patches(x)
        return dino_patches

    def _get_dino_gram_pool(self, dino_map):
        """Second-order (Gram matrix) pooling.

        Captures the spatial co-occurrence of feature channels rather than only their mean.
        - cable_swap: changed wire order -> changed channel co-occurrence -> different Gram response
        - combined: several defects -> several abnormal co-occurrence patterns -> stronger off-diagonal response
        - good: stable co-occurrence of normal cables -> consistent Gram features
        Analogous to bilinear pooling in texture recognition and fine-grained classification.

        Args:
            dino_map: [B, D, H, W] DINOv2 feature map
        Returns:
            dino_spatial_feat: [B, D*3] global mean + strongest co-occurrence + mean co-occurrence
        """
        B, D, H, W = dino_map.shape
        N = H * W

        # 1. Global mean pooling (base semantics; covers missing_cable, cut, ...)
        pool_avg = F.adaptive_avg_pool2d(dino_map, (1, 1)).view(B, D)  # [B, D]

        # 2. Gram matrix: spatial co-occurrence of the channels
        patches = dino_map.reshape(B, D, N).permute(0, 2, 1)  # [B, N, D]
        patches_norm = F.normalize(patches, dim=-1)
        gram = torch.bmm(patches_norm.transpose(1, 2), patches_norm) / N  # [B, D, D]

        # 3. Off-diagonal statistics (the diagonal is the per-channel energy; the off-diagonal is the co-occurrence)
        mask = ~torch.eye(D, dtype=torch.bool, device=gram.device)
        off_diag = gram[:, mask].reshape(B, D, D - 1)  # [B, D, D-1]

        # strongest co-occurrence of each channel: the channel it is most often active with
        gram_max = off_diag.max(dim=-1).values  # [B, D]
        # mean co-occurrence of each channel: overall co-occurrence strength
        gram_mean = off_diag.mean(dim=-1)  # [B, D]

        # 4. Concatenate: global semantics (384) + strongest (384) + mean co-occurrence (384) = 1152
        dino_spatial_feat = torch.cat([pool_avg, gram_max, gram_mean], dim=-1)
        dino_spatial_feat = F.normalize(dino_spatial_feat, dim=-1)

        return dino_spatial_feat

    def get_dino_gram_pool_masked(self, dino_tokens, mask_indices):
        """Local DINOv2 Gram features over a given set of patches.
        
        Pure tensor operations (torch.gather, mean, bmm); no learnable parameters.
        
        Args:
            dino_tokens: [B, N, 384] 
            mask_indices: [B, K] indices of the selected top-K patches
        Returns:
            dino_spatial_feat: [B, 1152]
        """
        B, N, D = dino_tokens.shape
        K = mask_indices.shape[1]
        
        # Clamp mask_indices (CLIP and DINO patch counts can differ slightly because of register tokens)
        mask_indices = torch.clamp(mask_indices, max=N - 1)
        
        # mask_indices is [B, K]; expand to [B, K, D] for gather
        expanded_indices = mask_indices.unsqueeze(-1).expand(B, K, D)
        selected_patches = torch.gather(dino_tokens, dim=1, index=expanded_indices)  # [B, K, D]
        
        # 1. Mean pooling over the K selected patches
        pool_avg = selected_patches.mean(dim=1)  # [B, D]
        
        # 2. Gram matrix over the K patches
        patches_norm = F.normalize(selected_patches, dim=-1)  # [B, K, D]
        # [B, D, K] bmm [B, K, D] -> [B, D, D]
        gram = torch.bmm(patches_norm.transpose(1, 2), patches_norm) / K  # [B, D, D]
        
        # 3. Off-diagonal statistics
        mask = ~torch.eye(D, dtype=torch.bool, device=gram.device)
        off_diag = gram[:, mask].reshape(B, D, D - 1)  # [B, D, D-1]
        
        # strongest and mean co-occurrence
        gram_max = off_diag.max(dim=-1).values  # [B, D]
        gram_mean = off_diag.mean(dim=-1)       # [B, D]
        
        # 4. Concatenate (384 + 384 + 384 = 1152) and normalise
        dino_spatial_feat = torch.cat([pool_avg, gram_max, gram_mean], dim=-1)
        dino_spatial_feat = F.normalize(dino_spatial_feat, dim=-1)
        
        return dino_spatial_feat

    def forward_patch_fusion(self, x):
        """CLIP and DINOv2 patch tokens, fused.

        Both patch sequences are padded or truncated to N = img_hw * img_hw and reshaped into feature maps,
        so that a token count that is not a perfect square (torch.hub DINOv2 with register tokens gives
        N = 1368) does not break the view.

        Args:
            x: [B, 3, H, W] input images

        Returns:
            fused_patches: [B, img_hw*img_hw, cross_modal_dim] fused patch features
        """
        # CLIP patch tokens [B, N_clip, 768]
        clip_patch_feats, _ = self._extract_clip_patch_tokens(x)
        B, N_clip, clip_dim = clip_patch_feats.shape

        # DINOv2 patch tokens [B, N_dino, 384]
        dino_patches = self._extract_dino_patch_tokens(x)

        target_N = self.img_hw * self.img_hw  # 37*37 = 1369

        # --- CLIP: padding/truncate → reshape → interpolate ---
        if N_clip < target_N:
            pad = torch.zeros(B, target_N - N_clip, clip_dim, 
                              device=clip_patch_feats.device, dtype=clip_patch_feats.dtype)
            clip_t = torch.cat([clip_patch_feats, pad], dim=1)
        else:
            clip_t = clip_patch_feats[:, :target_N, :]
        clip_img = clip_t.permute(0, 2, 1).view(B, clip_dim, self.img_hw, self.img_hw)
        clip_img = F.interpolate(clip_img, size=(self.img_hw, self.img_hw),
                                 mode='bilinear', align_corners=False)
        clip_resized = clip_img.view(B, clip_dim, target_N).permute(0, 2, 1)

        # --- DINOv2: padding/truncate → reshape → interpolate ---
        if dino_patches is not None:
            dino_dim = dino_patches.shape[-1]
            N_dino = dino_patches.shape[1]
            if N_dino < target_N:
                pad = torch.zeros(B, target_N - N_dino, dino_dim,
                                  device=dino_patches.device, dtype=dino_patches.dtype)
                dino_t = torch.cat([dino_patches, pad], dim=1)
            else:
                dino_t = dino_patches[:, :target_N, :]
            dino_img = dino_t.permute(0, 2, 1).view(B, dino_dim, self.img_hw, self.img_hw)
            dino_img = F.interpolate(dino_img, size=(self.img_hw, self.img_hw),
                                     mode='bilinear', align_corners=False)
            dino_resized = dino_img.view(B, dino_dim, target_N).permute(0, 2, 1)
        else:
            dino_dim = self.dino_extractor.feature_dim
            dino_resized = torch.zeros(B, target_N, dino_dim, 
                                       device=clip_patch_feats.device)

        cat_patches = torch.cat([clip_resized, dino_resized], dim=-1)  # [B, N_tgt, 1152]
        fused_patches = self.patch_fusion_proj(cat_patches)  # [B, N_tgt, cross_modal_dim]
        return fused_patches

    def forward_text_guided(self, x, text_feat=None, mode='all'):
        """Forward pass returning the text-refined visual feature.

        Args:
            x: [B, 3, H, W] input images
            text_feat: [B, 768] CLIP text feature (optional; random if omitted, for testing)
            mode: path selection
                'all'      - all three paths with gated fusion (default)
                'direct'   - direct cross-attention only
                'strip'    - strip path only
                'scale'    - scale path only
                'ds'       - Direct + Strip
                'dsc'      - Direct + Scale
                'ss'       - Strip + Scale
                'none'     - plain mean over patches (no text guidance; for testing)

        Returns:
            purified_feat: [B, out_dim] text-refined visual feature
            aux: dict of auxiliary outputs
        """
        # 1. Fused patch features [B, N, 768]
        fused_patches = self.forward_patch_fusion(x)

        # 2. Placeholder text feature if none is given
        if text_feat is None:
            B = fused_patches.shape[0]
            text_feat = torch.zeros(B, 768, device=fused_patches.device)

        # 3. Text-guided cross-modal fusion (with the selected mode)
        purified_feat, aux = self.text_guided_fusion(text_feat, fused_patches, mode=mode)

        return purified_feat, aux


    def get_anomaly_region_feature(
        self, x, text_feat, topk_ratio=0.3, fps_ratio=0.5
    ):
        """Defect-region feature from patches selected by attention and farthest-point sampling (FPS).

        Steps:
        1. fused patch features [B, N, D];
        2. cross-attention with the text feature gives an attention score per patch;
        3. the top-K patches by attention are kept;
        4. FPS keeps the K/2 most evenly spread of them;
        5. weighted pooling over the kept patches gives the defect-region feature.

        Args:
            x: [B, 3, H, W] input images
            text_feat: [B, 768] text feature
            topk_ratio: share of patches kept by the top-K step
            fps_ratio: share kept by FPS (relative to K)

        Returns:
            region_feat: [B, D] defect-region feature
            aux: dict with intermediate outputs
        """
        B, _, H, W = x.shape

        # 1. Fused patch features [B, N, 768]
        fused_patches = self.forward_patch_fusion(x)
        N = fused_patches.shape[1]
        D = fused_patches.shape[-1]

        # 2. Reuse self.text_guided_fusion.direct_attn instead of creating a new module;
        #    attention averaged over heads [B, N]
        direct_attn = self.text_guided_fusion.direct_attn
        _, attn_weights = direct_attn(text_feat, fused_patches, return_weights=True)

        # 3. Patch grid coordinates [N, 2]
        grid_h = grid_w = self.img_hw
        coords_h = torch.arange(grid_h, device=x.device).float() / grid_h
        coords_w = torch.arange(grid_w, device=x.device).float() / grid_w
        gy, gx = torch.meshgrid(coords_h, coords_w, indexing='ij')
        patch_coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # [N, 2]

        # 4. FPS selection per sample
        topk = max(1, int(N * topk_ratio))
        n_fps = max(1, int(topk * fps_ratio))

        selected_features_list = []
        selected_indices_list = []
        for b in range(B):
            feat_b = fused_patches[b]  # [N, D]
            attn_b = attn_weights[b]  # [N]

            # Top-K
            _, topk_idx = torch.topk(attn_b, topk)
            topk_feat = feat_b[topk_idx]  # [topk, D]
            topk_coord = patch_coords[topk_idx]  # [topk, 2]

            # FPS
            fps_idx = farthest_point_sampling(topk_feat, topk_coord, n_fps)
            selected_feat = topk_feat[fps_idx]  # [n_fps, D]

            selected_features_list.append(selected_feat)
            selected_indices_list.append(topk_idx[fps_idx])

        # 5. Attention-weighted mean over the selected patches
        # (one light attention pass over the selected patches)
        region_feats = []
        for b in range(B):
            selected_f = selected_features_list[b]  # [n_fps, D]
            # plain mean pooling
            pooled = selected_f.mean(dim=0)  # [D]
            region_feats.append(pooled)
        region_feat = torch.stack(region_feats, dim=0)  # [B, D]

        aux = {
            'attn_weights': attn_weights,                   # [B, N] attention scores
            'selected_indices': selected_indices_list,       # list of [n_fps]
            'topk_used': topk,
            'fps_used': n_fps,
        }
        return region_feat, aux

    def topk_avg_pool(self, tensor, k_ratio=0.1):
        """Top-K mean pooling: mean of the 10% most responsive patches.

        Coloured wires cover only 5%-10% of the image, so a global average is dominated by the background;
        the top-K mean keeps the responsive regions and smooths single-patch noise.
        """
        B, N, D = tensor.shape
        K = max(1, int(N * k_ratio))
        topk_vals, _ = torch.topk(tensor, K, dim=1)  # [B, K, D]
        return topk_vals.mean(dim=1)  # [B, D]

    def forward_fusion(self, x):
        """Memory-efficient training forward pass returning fused CLIP + DINOv2 features [B, 1920].

        CLIP path: det_token [B, N, 768] -> top-K mean (768) -> L2 norm
        DINOv2 path: patches [B, N, 384] -> Gram matrix pooling (1152) -> L2 norm
        Fusion: concatenation -> [B, 1920] -> L2 norm -> prototype memory
        """
        # --- CLIP path ---
        with torch.no_grad():
            x_embed = self.image_encoder.conv1(x)
            x_embed = x_embed.reshape(x_embed.shape[0], x_embed.shape[1], -1)
            x_embed = x_embed.permute(0, 2, 1)
            x_embed = torch.cat(
                [
                    self.image_encoder.class_embedding.to(x_embed.dtype)
                    + torch.zeros(x_embed.shape[0], 1, x_embed.shape[-1], dtype=x_embed.dtype, device=x_embed.device),
                    x_embed,
                ],
                dim=1,
            )
            x_embed = x_embed + self.image_encoder.positional_embedding.to(x_embed.dtype)
            x_embed = self.image_encoder.patch_dropout(x_embed)
            x_embed = self.image_encoder.ln_pre(x_embed)
            x_embed = x_embed.permute(1, 0, 2)

        # CLIP blocks and early adapters, run in the explicit mode
        x_embed = self._run_clip_blocks(x_embed)

        # CLIP det_token [B, N, 1024] (still 1024-d after ln_post)
        det_token = x_embed.permute(1, 0, 2)
        det_token = self.image_encoder.ln_post(det_token)

        # raw_clip_cls: after ln_post and before det_proj, projected to 768 by visual.proj,
        # i.e. in the space of the text encoder, which keeps image-text similarities calibrated
        if self.proto_text_refine == 'logit_boost':
            raw_cls = det_token[:, 0, :]  # [B, 1024] CLS token
            if self.image_encoder.proj is not None:
                raw_cls = raw_cls @ self.image_encoder.proj  # [B, 768]
            self._raw_clip_cls = raw_cls.detach()  # [B, 768]

        det_token = self.image_adapter["det_proj"](det_token)  # [B, N, 768]

        # Switchable CLIP aggregation: mean is the original control, top-K mean the full method.
        if self.clip_pool == 'topk':
            clip_feat = self.topk_avg_pool(det_token, k_ratio=0.1)  # [B, 768]
        else:
            clip_feat = det_token.mean(dim=1)  # [B, 768]
        clip_feat = F.normalize(clip_feat, dim=-1)

        # --- DINOv2 path: Gram matrix pooling ---
        with torch.no_grad():
            if self.dino_pool == 'cls':
                dino_spatial_feat = F.normalize(self.dino_extractor(x), dim=-1)
            else:
                dino_patches = self.dino_extractor.forward_patches(x)  # [B, N, 384]

            if self.dino_pool == 'mean':
                dino_spatial_feat = F.normalize(dino_patches.mean(dim=1), dim=-1)
            elif self.dino_pool == 'gram' and getattr(self, 'dino_mask_ratio', 0.0) > 0:
                # Build the patch mask on the fly:
                # the L2 norm of the projected CLIP features (det_token) serves as the spatial response
                spatial_scores = det_token.norm(dim=-1)  # [B, N_clip]
                K_mask = max(1, int(spatial_scores.shape[1] * self.dino_mask_ratio))
                _, mask_indices = torch.topk(spatial_scores, K_mask, dim=1)  # [B, K_mask]
                
                # masked Gram pooling
                dino_spatial_feat = self.get_dino_gram_pool_masked(dino_patches, mask_indices)
            elif self.dino_pool == 'gram':
                B, N, D = dino_patches.shape

                # Gram pooling directly on the flat patches, without reshaping to 2-D
                # (DINOv2 N may differ from H*W, e.g. N = 575 vs 576 with register tokens)
                dino_avg = dino_patches.mean(dim=1)  # [B, D]

                patches_norm = F.normalize(dino_patches, dim=-1)  # [B, N, D]
                gram = torch.bmm(patches_norm.transpose(1, 2), patches_norm) / N  # [B, D, D]

                mask = ~torch.eye(D, dtype=torch.bool, device=gram.device)
                off_diag = gram[:, mask].reshape(B, D, D - 1)  # [B, D, D-1]

                gram_max = off_diag.max(dim=-1).values   # [B, D]
                gram_mean = off_diag.mean(dim=-1)         # [B, D]

                dino_spatial_feat = torch.cat([dino_avg, gram_max, gram_mean], dim=-1)  # [B, 1152]
                dino_spatial_feat = F.normalize(dino_spatial_feat, dim=-1)

        # --- Fusion: 768 + width of the DINO branch ---
        fusion_feat = torch.cat([clip_feat, dino_spatial_feat], dim=-1)
        fusion_feat = F.normalize(fusion_feat, dim=-1)

        return fusion_feat

    def forward_hybrid_features(self, x, text_feat, mode="all"):
        """Single text query: CLIP_TopKAvg (768) + gamma * text-refinement residual + DINOv2_Gram (1152).

        The residual is added to clip_feat (768-d) rather than concatenated into a 2688-d vector.
        Small initial gamma: residual_gate starts at 0.01, so training starts close to the visual 1920-d baseline.
        Gradient isolation: clip_patches and clip_cls are detached before text_guided_fusion,
           so the auxiliary text gradient cannot alter the visual adapter.

        Returns:
            combined: [B, 1920], L2-normalised, for prototype classification / SupCon
            purified: [B, 768], L2-normalised, for the ITC loss
        """
        # CLIP patch tokens [B, N, 768] (projected)
        clip_patches, _ = self._extract_clip_patch_tokens(x)

        # CLIP global feature + DINOv2 global feature [B, 1920]
        fusion_feat = self.forward_fusion(x)
        clip_feat = fusion_feat[:, :768]   # [B, 768] CLIP Top-K Avg
        dino_feat = fusion_feat[:, 768:]   # [B, 1152] DINOv2 Gram

        # clip_cls is used for the residual
        clip_cls = clip_feat  # [B, 768]

        # Gradient isolation: detach the visual inputs so the auxiliary text gradient cannot alter the visual adapter
        purified, _ = self.text_guided_fusion(
            text_feat, clip_patches.detach(), clip_cls=clip_cls.detach(), mode=mode
        )
        purified = F.normalize(purified, dim=-1)

        # Residual: the increment contributed by text refinement.
        # It must not be normalised: normalisation would stretch the near-zero output of a closed gate
        # into a unit-length random vector, which is what produced spurious K-way responses.
        # The magnitude of the residual itself expresses how well the text matches:
        #   gate closed -> purified ~ clip_cls -> residual ~ 0 -> no spurious signal
        #   gate open   -> purified departs from clip_cls -> large residual -> informative signal
        residual = purified - clip_cls

        # Residual gate gamma (initialised at 0.01; with gamma = 0 the model reduces exactly to the visual baseline);
        # back-propagation opens the gate slowly according to the loss of hard samples
        residual = residual * self.residual_gate

        # The residual is added to clip_feat (768-d) rather than concatenated into a 2688-d vector:
        # concatenation would give the residual a third of the vote (768/2688); addition only adjusts (scaled by gamma).
        # With gamma = 0: clip_enhanced = clip_feat, i.e. the visual 1920-d baseline
        clip_enhanced = clip_feat + residual  # [B, 768]

        # Normalise each branch before concatenation:
        # with a single L2 normalisation of cat([clip_enhanced, dino_feat]), the residual makes |clip_enhanced| > 1,
        # the denominator grows and the DINO branch is scaled down.
        # Normalising each branch separately keeps the text branch from weakening the DINO branch.
        clip_enhanced_norm = F.normalize(clip_enhanced, dim=-1)
        dino_norm = F.normalize(dino_feat, dim=-1)
        combined = torch.cat([clip_enhanced_norm, dino_norm], dim=-1)  # [B, 1920]
        return F.normalize(combined, dim=-1), purified

    def forward_hybrid_features_kway(self, x, text_prototypes, mode="all"):
        """K-way class-conditional features: one text refinement per class text, for K-way training and inference.

        No @torch.no_grad(), so gradients flow back for K-way training.
        The residual is added to clip_feat (768-d) rather than concatenated into a 2688-d vector.
        Small initial gamma: residual_gate starts at 0.01, so training starts close to the visual 1920-d baseline.
        Gradient isolation: clip_patches and clip_cls are detached before text_guided_fusion.

        Returns:
            combined_bk: [B, K, 1920] L2-normalised fused features
            purified_bk: [B, K, 768] L2-normalised refined features (for the K-way ITC loss)
        """
        # CLIP patch tokens [B, N, 768]
        clip_patches, _ = self._extract_clip_patch_tokens(x)

        # fused features [B, 1920]
        fusion_feat = self.forward_fusion(x)
        b = x.shape[0]
        k = text_prototypes.shape[0]
        clip_feat = fusion_feat[:, :768]   # [B, 768] CLIP Top-K Avg
        clip_global = clip_feat             # [B, 768], used for the residual
        dino_feat = fusion_feat[:, 768:]    # [B, 1152] DINOv2 Gram

        # Gradient isolation: detach the visual inputs once
        clip_patches_detached = clip_patches.detach()
        clip_global_detached = clip_global.detach()

        combined_list = []
        purified_list = []
        aux_list = []  # gate outputs, for monitoring
        for c in range(k):
            text_q = text_prototypes[c].unsqueeze(0).expand(b, -1)
            # text-guided fusion on the detached visual features (the text gradient cannot alter the visual adapter)
            purified, aux = self.text_guided_fusion(
                text_q, clip_patches_detached, clip_cls=clip_global_detached, mode=mode
            )
            purified = F.normalize(purified, dim=-1)

            # refinement residual, not normalised (its magnitude carries information);
            # with a closed gate the residual is ~0 and adds no spurious signal
            residual = purified - clip_global

            # residual gate gamma (initialised at 0.01)
            residual = residual * self.residual_gate

            # the residual is added to clip_feat (768-d) rather than concatenated into a 2688-d vector
            clip_enhanced = clip_feat + residual  # [B, 768]

            # normalise each branch before concatenation
            clip_enhanced_norm = F.normalize(clip_enhanced, dim=-1)
            dino_norm = F.normalize(dino_feat, dim=-1)
            comb = torch.cat([clip_enhanced_norm, dino_norm], dim=-1)  # [B, 1920]
            combined_list.append(F.normalize(comb, dim=-1))
            purified_list.append(purified)
            aux_list.append(aux)

        # K visual features, one per text query
        return torch.stack(combined_list, dim=1), torch.stack(purified_list, dim=1), aux_list

    def compute_text_boosted_logits(self, vision_logits, text_prototypes, logit_scale):
        """logit_boost: temperature-softmax boost of the text logits (winner-take-all).

        final_logits = vision_logits + β * logit_scale * softmax(text_sim / τ)

        The visual features are never modified by text; text only adds to the logits.
        Image-text similarity uses raw_clip_cls (after ln_post, before det_proj),
          which lies in the space of the text encoder and is far better calibrated than the adapted feature.
        The temperature softmax removes the background level: only the class the text is most certain about gains.
        beta starts at 0.01 (ControlNet-style) and opens slowly during training.
        tau is learnable, so training finds the strength of the winner-take-all.

        Args:
            vision_logits: [B, K] visual logits (cosine_sim * logit_scale)
            text_prototypes: [K, 768] text prototypes (from compute_text_prototypes)
            logit_scale: scalar, prototype_mem.logit_scale.exp()

        Returns:
            boosted_logits: [B, K] logits after the text boost
        """
        if self.proto_text_refine != 'logit_boost':
            return vision_logits

        # raw_clip_cls (original CLIP CLS after ln_post and before det_proj) is better than the adapted
        # clip_feat here: it lies in the space of the text encoder, so image-text similarities are calibrated
        raw_clip_cls = self._raw_clip_cls  # [B, 768], extracted and detached in forward_fusion

        # detach text_prototypes: no gradient into the CLIP text encoder
        text_proto = text_prototypes.detach()

        # image-text cosine similarity [B, K]
        clip_cls_norm = F.normalize(raw_clip_cls, dim=-1)
        text_norm = F.normalize(text_proto, dim=-1)
        text_sim = clip_cls_norm @ text_norm.T  # [B, K]

        # temperature softmax: smaller tau is sharper (winner-take-all);
        # at tau = 0.07 a 0.35 vs 0.25 similarity gap becomes 0.90 vs 0.05
        text_weight = F.softmax(text_sim / self.text_temp, dim=-1)  # [B, K]

        # residual fusion: beta * logit_scale * text_weight;
        # logit_scale puts text_weight on the scale of vision_logits
        boosted = vision_logits + self.text_logit_gate * logit_scale * text_weight

        return boosted

    def refine_prototypes_with_text(self, visual_prototypes, text_prototypes):
        """proto_refine: text adjusts only the class prototypes, never the features.

        refined_proto = L2_normalize(visual_proto + β * text_proto_proj(text_proto))

        The visual features are never modified by text; text only refines the class prototypes.
        beta starts at 0.01 (ControlNet-style) and opens slowly during training.
        The sum must be L2-normalised; otherwise classes whose prototype norm grows would gain higher absolute scores.

        Args:
            visual_prototypes: [K, 1920] visual prototypes (from PrototypeMemory)
            text_prototypes: [K, 768] text prototypes (from compute_text_prototypes)

        Returns:
            refined_prototypes: [K, 1920] refined prototypes (L2-normalised)
        """
        if self.proto_text_refine != 'proto_refine':
            return visual_prototypes

        # detach text_prototypes: no gradient into the CLIP text encoder
        text_proto = text_prototypes.detach()

        # text_proto_proj: [K, 768] → [K, 1920]
        text_proj = self.text_proto_proj(text_proto)  # [K, 1920]

        # detach visual_prototypes: no text gradient into the purely visual parameters;
        # the text gradient updates only beta and text_proto_proj, the prototypes follow the visual loss
        refined = visual_prototypes.detach() + self.proto_text_gate * text_proj

        # project the refined prototypes back onto the unit sphere (required)
        return F.normalize(refined, dim=-1)

    def encode_text(self, text, adapt_text=True):

        if not adapt_text:
            return self.clipmodel.encode_text(text)
        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(text).to(cast_dtype)
        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        for i in range(12):
            x, _ = self.clipmodel.transformer.resblocks[i](
                x, attn_mask=self.clipmodel.attn_mask
            )
            if i < self.text_adapt_until:
                adapt_out = self.text_adapter[i](x)
                adapt_out = (
                    adapt_out
                    * x.norm(dim=-1, keepdim=True)
                    / adapt_out.norm(dim=-1, keepdim=True)
                )
                x = self.t_w * adapt_out + (1 - self.t_w) * x
        x = x.permute(1, 0, 2)
        x = self.clipmodel.ln_final(x)
        x = self.text_adapter[-1](x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        return x

    @torch.no_grad()
    def get_anomaly_hotspots(self, x, top_k=2):
        """High-response regions (anomalies, edges) located by the norm of DINOv2 patch features.

        Inference only; takes no part in training and changes no global feature.

        Args:
            x: [B, 3, H, W] input images
            top_k: number of most responsive patches returned

        Returns:
            [B, top_k] list of coordinates (center_x, center_y) in [0, 1]
        """
        dino_patches = self.dino_extractor.forward_patches(x)  # [B, N, 384]
        B, N, D = dino_patches.shape

        heatmap = dino_patches.norm(dim=-1)  # [B, N]
        _, topk_indices = torch.topk(heatmap, top_k, dim=1)  # [B, top_k]

        hotspots = []
        for b in range(B):
            batch_hotspots = []
            for k in range(top_k):
                idx = topk_indices[b, k].item()
                row = idx // self.img_hw
                col = idx % self.img_hw
                center_y = (row + 0.5) / self.img_hw
                center_x = (col + 0.5) / self.img_hw
                batch_hotspots.append((center_x, center_y))
            hotspots.append(batch_hotspots)

        return hotspots
