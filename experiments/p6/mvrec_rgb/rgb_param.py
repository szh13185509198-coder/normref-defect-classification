"""lyus Param tree replicating official MVREC defaults for the semantic audit.

Values are copied verbatim from ``MVREC-main/param_space.py`` at the locked
upstream commit ``112c187b6de8a7400fad89ca2a201545842a5d84``.  The official
``modules/model.py`` and ``modules/classifier.py`` read configuration through
the ``lyus.Frame.Experiment`` singleton, so this module only satisfies that
singleton with the official defaults; it does not alter any official value.

The single audit-visible override is ``debug.k_shot`` (few-shot K), which is a
dataset-cardinality parameter exposed on the official command line, not a
model semantic.
"""

import os
import sys

_MVREC_MAIN = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "MVREC-main")
)


def mvrec_main_root() -> str:
    return _MVREC_MAIN


def ensure_mvrec_on_path():
    """Add the vendored upstream MVREC-main directory to sys.path (idempotent)."""
    if _MVREC_MAIN not in sys.path:
        sys.path.insert(0, _MVREC_MAIN)
    return _MVREC_MAIN


# Official classifier head (Zip-Adapter-F) used by MVREC.
OFFICIAL_CLASSIFIER = "EchoClassfierF"
# Official backbone family and pretrained representation.
OFFICIAL_BACKBONE = "ViT-L/14"
OFFICIAL_CLIP_NAME = "AlphaClip"
# Official input resolution: the official README run command and every
# recorded official run pass ``--data.input_shape 224`` (see MVREC-main/OUTPUT
# run names); AlphaCLIP ViT-L/14 has 257 positional embeddings matching 224.
OFFICIAL_INPUT_SHAPE = 224
# Official text prompt template (modules/clip_preprocess.py get_label_info).
OFFICIAL_TEXT_TEMPLATE = "a photo of {} defect on the image"


def build_rgb_param(class_names, k_shot=1, ft_epo=500, zip_config_index=6):
    """Build a lyus ``Param`` tree with the official MVREC defaults.

    Every value below is the exact official default from
    ``MVREC-main/param_space.py`` / ``MVREC-main/modules/classifier.py``.
    ``k_shot`` and ``ft_epo`` are dataset-cardinality / budget knobs that the
    official CLI exposes; the audit pins them to the protocol's values.
    """
    ensure_mvrec_on_path()
    from lyus.Frame.Exp.param import Param  # vendored lyus

    p = Param()
    p.device = "cuda" if _cuda_available() else "cpu"

    p.data = Param()
    p.data.class_names = list(class_names)
    p.data.num_classes = len(class_names)
    p.data.input_shape = OFFICIAL_INPUT_SHAPE
    p.data.data_name = "mvtec_bottle_data"

    p.ClipModel = Param(
        clip_name=OFFICIAL_CLIP_NAME,
        backbone_name=OFFICIAL_BACKBONE,
        classifier=OFFICIAL_CLASSIFIER,
        input_shape=OFFICIAL_INPUT_SHAPE,
        text_list=list(class_names),
    )

    # Official debug defaults (param_space.py).
    p.debug = Param(
        train_process=True,
        load_weight=False,
        filtered_module=["proto"],
        k_patch=2,
        sdpa_scale=300,
        ft_epo=int(ft_epo),
        train_sk=False,
        acti_beta=20,
        fvns=1,
        fvnq=1,
        trip_loss_weight=4,
        trip_loss_margin=0.5,
        mulit_view=1,
        k_shot=int(k_shot),
        num_sampling=1,
        infer_style="assemble_uncertainty",
        zip_config_index=int(zip_config_index),
        text_logits_wight=0,
        uncertainty_alpha=2,
    )
    return p


def _cuda_available():
    import torch

    return torch.cuda.is_available()


def init_experiment_singleton(param, save_root=None):
    """Set the lyus Experiment singleton with the given Param tree.

    The official model/classifier code reads ``Experiment().get_param()``;
    this mirrors upstream ``run.py``'s ``build_new_exper`` + ``set_param``
    without creating any save directory or logger.  ``save_root`` is an
    audit-local directory used only by the official debug feature-dump
    helper ``save_feature_xy`` (its output is not part of any model path).
    """
    ensure_mvrec_on_path()
    from lyus.Frame.Exp.experiment import Experiment

    if save_root is None:
        save_root = os.path.join(os.getcwd(), "ckpt", "p6b_mvrec_rgb_semantic_audit")
    os.makedirs(save_root, exist_ok=True)
    exper = Experiment(
        save_root=save_root, project_name=None, exp_name=None, run_name=None
    )
    exper.set_param(param)
    return exper
