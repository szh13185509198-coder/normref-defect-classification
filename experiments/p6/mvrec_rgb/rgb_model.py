"""Model construction for the MVREC-RGB adaptation.

Builds the official ``modules.model.ClipModel`` with the official backbone
(AlphaClip ViT-L/14 with GRIT-20M pretrained weights), the official text
prior, and the official Zip-Adapter-F classifier (``EchoClassfierF``), then
runs the official support-side pipeline ``ClipModel.set_img_prototype`` which
executes the official classifier training semantics (AdamW lr=1e-4 eps=1e-6,
CE (+triplet) loss, CosineAnnealingLR, ft_epo steps).

No model component is modified, replaced, or retrained differently from the
official code; this module only wires the official components to the
whole-image RGB dataset.
"""

import os

import torch

from experiments.p6.mvrec_rgb.rgb_param import (
    OFFICIAL_BACKBONE,
    OFFICIAL_CLASSIFIER,
    OFFICIAL_CLIP_NAME,
    OFFICIAL_INPUT_SHAPE,
    OFFICIAL_TEXT_TEMPLATE,
    build_rgb_param,
    ensure_mvrec_on_path,
    init_experiment_singleton,
)


def build_mvrec_rgb_model(
    class_names,
    alpha_ckpt_path=None,
    k_shot=1,
    ft_epo=500,
    zip_config_index=6,
    seed=None,
):
    """Construct the official MVREC model under the RGB-only interface.

    Returns (model, param).  The model is the untouched official ClipModel;
    the only difference to upstream run.py is the data path feeding it.
    """
    ensure_mvrec_on_path()

    if seed is not None:
        _fix_seeds(seed)

    param = build_rgb_param(
        class_names=class_names,
        k_shot=k_shot,
        ft_epo=ft_epo,
        zip_config_index=zip_config_index,
    )
    init_experiment_singleton(param)

    if alpha_ckpt_path:
        os.environ["MVREC_ALPHA_L14_CKPT"] = os.path.abspath(alpha_ckpt_path)

    from modules.model import ClipModel  # official upstream component

    model = ClipModel(
        clip_name=OFFICIAL_CLIP_NAME,
        backbone_name=OFFICIAL_BACKBONE,
        classifier=OFFICIAL_CLASSIFIER,
        input_shape=OFFICIAL_INPUT_SHAPE,
        text_list=list(class_names),
    )
    model.to(param.device)
    # Official training-mode parameter policy (modules/model.py
    # set_trainable_attention_params): for the ViT backbone this leaves every
    # backbone parameter frozen; the head controls its own trainable groups.
    model.set_trainable_attention_params()
    return model, param


def official_prompt_texts(class_names):
    """Official text prior (modules/clip_preprocess.py get_label_info)."""
    return [OFFICIAL_TEXT_TEMPLATE.format(label) for label in class_names]


def _fix_seeds(seed):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
