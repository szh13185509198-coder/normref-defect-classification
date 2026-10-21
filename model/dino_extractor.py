"""
DINOv2 feature extractor.

Loading options (in order of priority):
1. torch.hub from the local cache (or online when GitHub is reachable);
2. the transformers library (Hugging Face Hub, or a mirror set by HF_ENDPOINT).

When HF_ENDPOINT points to a mirror, transformers downloads from it without further configuration.
"""

import torch
import torch.nn as nn
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# DINOv2 model configuration
# key: model name used by Dinov2Extractor -> (torch.hub name, transformers name, feature width)
DINOV2_REGISTRY = {
    # small ViT-S/14 (recommended: fast, little memory)
    'dinov2_vits14': {
        'hub_name': 'dinov2_vits14',
        'hf_name': 'facebook/dinov2-small',
        'dim': 384,
    },
    # base ViT-B/14
    'dinov2_vitb14': {
        'hub_name': 'dinov2_vitb14',
        'hf_name': 'facebook/dinov2-base',
        'dim': 768,
    },
    # large ViT-L/14
    'dinov2_vitl14': {
        'hub_name': 'dinov2_vitl14',
        'hf_name': 'facebook/dinov2-large',
        'dim': 1024,
    },
    # giant ViT-G/14 (large memory footprint)
    'dinov2_vitg14': {
        'hub_name': 'dinov2_vitg14',
        'hf_name': 'facebook/dinov2-giant',
        'dim': 1536,
    },
}


def _check_hub_cache():
    """Whether torch.hub has cached the DINOv2 code and weights."""
    hub_dir = os.path.expanduser(
        os.environ.get(
            "TORCH_HOME",
            os.path.join(os.path.expanduser("~"), ".cache", "torch"),
        )
    )
    hub_dir = os.path.join(hub_dir, "hub")
    dino_dir = os.path.join(hub_dir, "facebookresearch_dinov2_main")
    return os.path.exists(dino_dir)


class Dinov2Extractor(nn.Module):
    """DINOv2 feature extractor.

    Returns the CLS token feature, which can be fused with the CLIP det_token.

    Args:
        model_name: model name, 'dinov2_vits14' (default), 'dinov2_vitb14',
                    'dinov2_vitl14', 'dinov2_vitg14'
        freeze: freeze the DINOv2 parameters (default True)
        use_hub_first: try torch.hub first (default True);
                       set False to load through transformers directly (e.g. behind a mirror)
    """
    def __init__(
        self,
        model_name='dinov2_vits14',
        freeze=True,
        use_hub_first=None,
        patch_policy='full',
        strict_backend=False,
        hub_repo_path=None,
        hub_checkpoint_path=None,
    ):
        super().__init__()

        if model_name not in DINOV2_REGISTRY:
            raise ValueError(
                f"Unknown DINOv2 model '{model_name}'. "
                f"Available: {list(DINOV2_REGISTRY.keys())}"
            )

        self.cfg = DINOV2_REGISTRY[model_name]
        self.feature_dim = self.cfg['dim']
        if patch_policy not in {'full', 'legacy_drop_first'}:
            raise ValueError(f"Unsupported DINO patch policy: {patch_policy}")
        self.patch_policy = patch_policy
        self.strict_backend = bool(strict_backend)
        self.hub_repo_path = hub_repo_path or os.environ.get("DINOV2_HUB_REPO")
        self.hub_checkpoint_path = (
            hub_checkpoint_path or os.environ.get("DINOV2_HUB_CHECKPOINT")
        )

        # automatic choice: do not use torch.hub when HF_ENDPOINT sets a mirror
        if use_hub_first is None:
            use_hub_first = (
                'HF_ENDPOINT' not in os.environ
                and _check_hub_cache()
            )

        # load the model
        self.backend = None
        if use_hub_first:
            self.model = self._load_from_hub(model_name)
        else:
            self.model = self._load_from_transformers(model_name)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        print(
            f"[DINOv2] Loaded '{model_name}' (dim={self.feature_dim}), "
            f"method={self.backend}, patch_policy={self.patch_policy}, freeze={freeze}"
        )

    def _apply_patch_policy(self, patch_tokens):
        """Apply an explicit spatial-token protocol for controlled reproduction."""
        if patch_tokens is None:
            return None
        if self.patch_policy == 'legacy_drop_first':
            if patch_tokens.shape[1] <= 1:
                raise RuntimeError("Cannot drop the first token from an empty patch grid")
            return patch_tokens[:, 1:, :]
        return patch_tokens

    def _load_from_hub(self, model_name):
        """Load from torch.hub (local cache first)."""
        try:
            if self.hub_repo_path:
                repo_path = os.path.abspath(os.path.expanduser(self.hub_repo_path))
                if not os.path.isfile(os.path.join(repo_path, "hubconf.py")):
                    raise FileNotFoundError(
                        f"Local DINOv2 repo has no hubconf.py: {repo_path}"
                    )
                if not self.hub_checkpoint_path:
                    raise FileNotFoundError(
                        "DINOV2_HUB_CHECKPOINT/--dino_hub_checkpoint is required "
                        "when loading an offline local hub repository"
                    )
                checkpoint_path = os.path.abspath(
                    os.path.expanduser(self.hub_checkpoint_path)
                )
                if not os.path.isfile(checkpoint_path):
                    raise FileNotFoundError(
                        f"Local DINOv2 checkpoint not found: {checkpoint_path}"
                    )
                model = torch.hub.load(
                    repo_path,
                    self.cfg['hub_name'],
                    source='local',
                    pretrained=True,
                    weights=checkpoint_path,
                )
                self.backend = "hub_local"
                print(
                    "[DINOv2] Loaded from offline local torch.hub repo: "
                    f"{repo_path}"
                )
                print(f"[DINOv2] Local checkpoint: {checkpoint_path}")
                return model

            # try the local cache first (offline)
            hub_dir = os.path.expanduser(
                os.environ.get("TORCH_HOME",
                    os.path.join(os.path.expanduser("~"), ".cache", "torch"))
            )
            model = torch.hub.load(
                'facebookresearch/dinov2',
                self.cfg['hub_name'],
                trust_repo=True,
                force_reload=False,
            )
            self.backend = "hub"
            print(f"[DINOv2] Loaded via torch.hub")
            return model
        except Exception as e:
            if self.strict_backend:
                raise RuntimeError(
                    "DINOv2 torch.hub was explicitly requested but could not be loaded"
                ) from e
            print(f"[DINOv2] torch.hub failed ({type(e).__name__}), falling back to transformers")
            return self._load_from_transformers(model_name)


    def _load_from_transformers(self, model_name):
        """Load from Hugging Face transformers (a mirror can be set with HF_ENDPOINT)."""
        try:
            from transformers import AutoModel
        except ImportError:
            print("[DINOv2] transformers not installed. Installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "transformers", "-q"])
            from transformers import AutoModel

        hf_name = self.cfg['hf_name']

        # report a Hugging Face mirror if one is set
        hf_endpoint = os.environ.get("HF_ENDPOINT", "")
        if hf_endpoint:
            print(f"[DINOv2] Using HF mirror: {hf_endpoint}")

        print(f"[DINOv2] Loading from HuggingFace: {hf_name}")
        try:
            model = AutoModel.from_pretrained(
                hf_name,
                trust_remote_code=True,
            )
            self.backend = "transformers"
            print(f"[DINOv2] Loaded from HuggingFace successfully")
            return model
        except Exception as e:
            raise RuntimeError(
                f"\n{'='*70}\n"
                f"  DINOv2 model '{model_name}' download FAILED.\n"
                f"  Error: {type(e).__name__}: {e}\n"
                f"{'='*70}\n\n"
                f"  If the Hugging Face Hub is not reachable, set a mirror:\n"
                f"      export HF_ENDPOINT=https://hf-mirror.com\n"
                f"      python train_multiclass.py ...\n\n"
                f"  Or download the model in advance:\n"
                f"      python -c \"from transformers import AutoModel; "
                f"AutoModel.from_pretrained('{hf_name}')\"\n"
                f"{'='*70}"
            ) from e

    @staticmethod
    def convert_hub_to_transformer_state_dict(state_dict, dino_prefix="dino_extractor.model"):
        """Rename the keys of a torch.hub DINOv2 state_dict to the transformers convention.
        
        Hub keys (as saved during training):
            model.cls_token → model.embeddings.cls_token
            model.patch_embed.proj.weight → model.embeddings.patch_embeddings.projection.weight
            model.blocks.0.attn.qkv.weight -> split into separate q/k/v weights
            model.blocks.0.ls1.gamma → model.encoder.layer.0.layer_scale1.lambda1
            model.norm.weight → model.layernorm.weight
        
        Args:
            state_dict: full state_dict containing hub-style DINOv2 keys
            dino_prefix: prefix of DINOv2 inside AdaptedCLIP (default "dino_extractor.model")
        
        Returns:
            new state_dict with transformers-style keys
        """
        new_sd = {}
        for key, tensor in state_dict.items():
            # is this a hub-style DINOv2 key?
            if not key.startswith(dino_prefix + ".blocks.") and \
               key != dino_prefix + ".cls_token" and \
               key != dino_prefix + ".pos_embed" and \
               key != dino_prefix + ".mask_token" and \
               not key.startswith(dino_prefix + ".patch_embed.") and \
               key != dino_prefix + ".norm.weight" and \
               key != dino_prefix + ".norm.bias":
                # not a hub-style DINOv2 key: keep it
                new_sd[key] = tensor
                continue
            
            # strip the prefix
            suffix = key[len(dino_prefix) + 1:]  # e.g. "blocks.0.attn.qkv.weight"
            
            if suffix == "cls_token":
                new_key = f"{dino_prefix}.embeddings.cls_token"
            elif suffix == "pos_embed":
                new_key = f"{dino_prefix}.embeddings.position_embeddings"
            elif suffix == "mask_token":
                new_key = f"{dino_prefix}.embeddings.mask_token"
            elif suffix.startswith("patch_embed."):
                # patch_embed.proj.weight → embeddings.patch_embeddings.projection.weight
                sub = suffix[len("patch_embed."):]  # "proj.weight"
                # drop the "proj." prefix; transformers uses projection.weight directly
                sub_parts = sub.split(".")
                if sub_parts[0] == "proj":
                    sub = ".".join(sub_parts[1:])  # "weight"
                new_key = f"{dino_prefix}.embeddings.patch_embeddings.projection.{sub}"

            elif suffix.startswith("blocks."):
                # blocks.{i}.xxx.xxx → encoder.layer.{i}.xxx.xxx
                parts = suffix.split(".")
                block_idx = parts[1]  # "0", "1", ..., "11"
                sub_path = ".".join(parts[2:])  # "norm1.weight", "attn.qkv.weight", etc.
                
                if sub_path == "norm1.weight":
                    new_key = f"{dino_prefix}.encoder.layer.{block_idx}.norm1.weight"
                elif sub_path == "norm1.bias":
                    new_key = f"{dino_prefix}.encoder.layer.{block_idx}.norm1.bias"
                elif sub_path.startswith("attn.qkv."):
                    # split the fused qkv into q/k/v
                    param_type = sub_path.split(".")[-1]  # "weight" or "bias"
                    D = tensor.shape[0] // 3  # feature dim
                    q, k, v = torch.split(tensor, D, dim=0)
                    for name, split_t in [("query", q), ("key", k), ("value", v)]:
                        nk = f"{dino_prefix}.encoder.layer.{block_idx}.attention.attention.{name}.{param_type}"
                        new_sd[nk] = split_t
                    continue  # already added; skip new_sd[key] = tensor below
                elif sub_path == "attn.proj.weight":
                    new_key = f"{dino_prefix}.encoder.layer.{block_idx}.attention.output.dense.weight"
                elif sub_path == "attn.proj.bias":
                    new_key = f"{dino_prefix}.encoder.layer.{block_idx}.attention.output.dense.bias"
                elif sub_path.startswith("ls1."):
                    param_type = sub_path.split(".")[-1]  # "gamma"
                    new_key = f"{dino_prefix}.encoder.layer.{block_idx}.layer_scale1.lambda1"
                elif sub_path == "norm2.weight":
                    new_key = f"{dino_prefix}.encoder.layer.{block_idx}.norm2.weight"
                elif sub_path == "norm2.bias":
                    new_key = f"{dino_prefix}.encoder.layer.{block_idx}.norm2.bias"
                elif sub_path.startswith("mlp.fc1."):
                    param_type = sub_path.split(".")[-1]
                    new_key = f"{dino_prefix}.encoder.layer.{block_idx}.mlp.fc1.{param_type}"
                elif sub_path.startswith("mlp.fc2."):
                    param_type = sub_path.split(".")[-1]
                    new_key = f"{dino_prefix}.encoder.layer.{block_idx}.mlp.fc2.{param_type}"
                elif sub_path.startswith("ls2."):
                    param_type = sub_path.split(".")[-1]  # "gamma"
                    new_key = f"{dino_prefix}.encoder.layer.{block_idx}.layer_scale2.lambda1"
                else:
                    # unknown sub_path: kept unchanged (may fail later)
                    new_key = key
            elif suffix == "norm.weight":
                new_key = f"{dino_prefix}.layernorm.weight"
            elif suffix == "norm.bias":
                new_key = f"{dino_prefix}.layernorm.bias"
            else:
                new_key = key
            
            new_sd[new_key] = tensor
        
        return new_sd

    def forward(self, x):

        """DINOv2 CLS token features.

        Args:
            x: [B, 3, H, W] input images

        Returns:
            [B, feature_dim] CLS token features
        """
        # transformers models return last_hidden_state; take the [CLS] token
        if hasattr(self.model, 'config') and hasattr(self.model, 'forward'):
            # transformers style
            outputs = self.model(x)
            # outputs.last_hidden_state: [B, seq_len, dim]
            if hasattr(outputs, 'last_hidden_state'):
                cls_feat = outputs.last_hidden_state[:, 0, :]  # [B, dim]
            else:
                # some DINOv2 variants return pooler_output
                cls_feat = outputs[0][:, 0, :] if isinstance(outputs, (list, tuple)) else outputs[:, 0, :]
            return cls_feat
        else:
            # torch.hub style: returns the CLS token directly
            return self.model(x)

    @torch.no_grad()
    def forward_patches(self, x):
        """All DINOv2 patch tokens (without the CLS token).

        Args:
            x: [B, 3, H, W] input images

        Returns:
            [B, N, feature_dim] all patch tokens, N = (H/14)*(W/14)
        """
        # transformers style
        if hasattr(self.model, 'config') and hasattr(self.model, 'forward'):
            outputs = self.model(x, output_hidden_states=True)
            if hasattr(outputs, 'last_hidden_state'):
                all_tokens = outputs.last_hidden_state  # [B, seq_len, dim]
            else:
                all_tokens = outputs[0]  # [B, seq_len, dim]
            patch_tokens = all_tokens[:, 1:, :]  # [B, N, dim], CLS removed
            return self._apply_patch_policy(patch_tokens)
        else:
            # torch.hub style: DINOv2 hub model;
            # forward_features returns (x_norm_pooled, x_norm, x_prenorm)
            # x_norm_pooled: [B, D]  CLS + GeM-pooled feature
            # x_norm:        [B, 1+N, D]  all tokens (with CLS) after LayerNorm
            # x_prenorm:     [B, 1+N, D]  all tokens (with CLS) without LayerNorm
            if hasattr(self.model, 'forward_features'):
                out = self.model.forward_features(x)
                # Official DINOv2 hub models return a dict.  The previous
                # fallback called get_intermediate_layers(), which already
                # excludes CLS, and then sliced [:, 1:] again.  That silently
                # dropped the first spatial patch (576 -> 575 at 336 px).
                if isinstance(out, dict):
                    patch_tokens = out.get("x_norm_patchtokens")
                    if torch.is_tensor(patch_tokens):
                        return self._apply_patch_policy(patch_tokens)
                if isinstance(out, tuple) and len(out) >= 2 and torch.is_tensor(out[1]):
                    all_tokens = out[1]  # x_norm: [B, 1+N, D]
                    if all_tokens.dim() == 3 and all_tokens.shape[1] > 1:
                        return self._apply_patch_policy(all_tokens[:, 1:, :])
            # Official get_intermediate_layers() already returns patch tokens
            # unless return_class_token=True; do not remove another token.
            if hasattr(self.model, 'get_intermediate_layers'):
                out_list = self.model.get_intermediate_layers(x, n=1)
                if out_list and torch.is_tensor(out_list[0]) and out_list[0].dim() == 3:
                    return self._apply_patch_policy(out_list[0])
            # last fallback
            return None
