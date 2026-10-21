"""A1 upstream component map for official MVREC.

Static reconstruction of the official MVREC production call graph at the
locked upstream commit ``112c187b6de8a7400fad89ca2a201545842a5d84``, traced
through the actual production path used by ``run.py``:

    run.py main_process
      -> create_train_data / create_valid_data  (MultiviewRoiData, CSV)
      -> create_model                            (ClipModel)
      -> FewShotEvalFunc / EvalTool.run_eval
           -> pre_cal_feature                    (get_mvrec with masks)
           -> PartDatasetTool.get_suport_query_data
           -> model.set_img_prototype           (head.init_weight ->
                                                  EchoClassfierF.train_compact)
           -> eval_once                         (query path; forbidden here)

``planned_status`` vocabulary (Protocol A):
- ``preserved``: kept unchanged in the adaptation;
- ``interface_adapted``: §5-allowed data-interface change;
- ``privileged_and_removed``: mask/bbox/ROI branch removed from the
  adaptation path;
- ``substantively_changed``: would require a rewrite -> STOP gate.
"""

COMPONENT_MAP = [
    # --- backbone family
    dict(
        component="backbone",
        upstream_file="modules/bbox_clip/alpha_clip_backbone.py",
        upstream_symbol="AlphaClip",
        role="AlphaCLIP ViT-L/14 visual encoder with GRIT-20M pretrained weights",
        consumes_mask=True,
        consumes_bbox=False,
        consumes_roi=False,
        annotation_dependency="mask tensor injected via conv1_alpha alpha branch",
        planned_status="interface_adapted",
        reason="official backbone kept unchanged; the only change is the "
               "checkpoint path (compatibility fix, semantic_effect=none) and "
               "the mask input becomes a runtime constant all-ones API "
               "placeholder per Protocol A section 3",
    ),
    dict(
        component="backbone_variant_no_mask",
        upstream_file="modules/bbox_clip/alpha_clip_backbone.py",
        upstream_symbol="AlphaClipWomask",
        role="official no-mask variant (fills mask with constant)",
        consumes_mask=True,
        consumes_bbox=False,
        consumes_roi=False,
        annotation_dependency="mask tensor replaced by constant at runtime",
        planned_status="preserved",
        reason="official ablation variant kept verbatim (not the default)",
    ),
    # --- multi-view / ROI generation
    dict(
        component="multi_view_generation",
        upstream_file="modules/multi_view_data.py",
        upstream_symbol="MultiScaleRoiSampleProcess / MultiScaleOffsetRoiSampleProcess",
        role="bbox-centered multi-scale/multi-offset ROI crops (mso method)",
        consumes_mask=True,
        consumes_bbox=True,
        consumes_roi=True,
        annotation_dependency="roi size/center/offset derived from GT bbox; "
                              "mask polygon derived from GT points",
        planned_status="privileged_and_removed",
        reason="annotation-derived views replaced by the fixed predeclared "
               "global-view list (whole image) per Protocol A section 5",
    ),
    dict(
        component="roi_generation",
        upstream_file="modules/utils.py",
        upstream_symbol="RoiGenerator / MultiScaleRoiGenerator.gen_one",
        role="crop rectangle geometry around defect bbox",
        consumes_mask=False,
        consumes_bbox=True,
        consumes_roi=True,
        annotation_dependency="center/size/offset from bbox",
        planned_status="privileged_and_removed",
    ),
    dict(
        component="bbox_processing",
        upstream_file="modules/utils.py",
        upstream_symbol="Bbox",
        role="bbox object carried through the pipeline",
        consumes_mask=False,
        consumes_bbox=True,
        consumes_roi=False,
        annotation_dependency="x1/x2/y1/y2 from CSV",
        planned_status="privileged_and_removed",
    ),
    dict(
        component="mask_processing",
        upstream_file="modules/multi_view_data.py / modules/clip_preprocess.py",
        upstream_symbol="point2mask / mask_transform",
        role="GT polygon points -> binary mask -> normalized tensor",
        consumes_mask=True,
        consumes_bbox=False,
        consumes_roi=False,
        annotation_dependency="GT points column of the CSV",
        planned_status="privileged_and_removed",
    ),
    # --- data loading
    dict(
        component="dataset",
        upstream_file="modules/fabric_data.py / modules/multi_view_data.py",
        upstream_symbol="MultiviewRoiData (FabricData base)",
        role="reads CONFIG csv (bbox/points/label) + RGB images",
        consumes_mask=True,
        consumes_bbox=True,
        consumes_roi=True,
        annotation_dependency="all localization fields of the CSV",
        planned_status="interface_adapted",
        reason="replaced by whole-image RGB loader reading only the frozen "
               "P6.B manifest (rel_path + rgb_sha256 + class label)",
    ),
    dict(
        component="sample_preprocess",
        upstream_file="modules/clip_preprocess.py",
        upstream_symbol="ClipPreprocess",
        role="resize image+bbox+mask, CLIP normalize, label -> text prompt",
        consumes_mask=True,
        consumes_bbox=True,
        consumes_roi=False,
        annotation_dependency="bbox resize; mask resize",
        planned_status="interface_adapted",
        reason="kept the official text template and normalization constants; "
               "dropped the bbox/mask resizing branches",
    ),
    # --- text prior
    dict(
        component="text_prior",
        upstream_file="modules/clip_preprocess.py",
        upstream_symbol="ClipPreprocess.get_label_info",
        role="class-name prompt: 'a photo of {label} defect on the image'",
        consumes_mask=False,
        consumes_bbox=False,
        consumes_roi=False,
        annotation_dependency="none (image-level class names only)",
        planned_status="preserved",
    ),
    dict(
        component="text_features",
        upstream_file="modules/model.py / modules/bbox_clip/alpha_clip_backbone.py",
        upstream_symbol="ClipModel.__init__ / AlphaClip.encode_text",
        role="frozen text encoder over class-name prompts",
        consumes_mask=False,
        consumes_bbox=False,
        consumes_roi=False,
        annotation_dependency="none",
        planned_status="preserved",
    ),
    # --- classifier / adapter
    dict(
        component="classifier_head",
        upstream_file="modules/classifier.py",
        upstream_symbol="EchoClassfierF (Zip-Adapter-F)",
        role="support-conditioned few-shot classifier: zifa adapter + "
             "support_key + SdpaModule one-hot proxy",
        consumes_mask=False,
        consumes_bbox=False,
        consumes_roi=False,
        annotation_dependency="none (consumes features + image-level labels)",
        planned_status="preserved",
    ),
    dict(
        component="adapter_module",
        upstream_file="modules/torch_demo.py",
        upstream_symbol="AdapterModule",
        role="residual Linear+SiLU feature adapter (zifa)",
        consumes_mask=False,
        consumes_bbox=False,
        consumes_roi=False,
        annotation_dependency="none",
        planned_status="preserved",
    ),
    dict(
        component="sdpa_module",
        upstream_file="modules/classifier.py",
        upstream_symbol="SdpaModule",
        role="support attention with one-hot class proxy",
        consumes_mask=False,
        consumes_bbox=False,
        consumes_roi=False,
        annotation_dependency="none",
        planned_status="preserved",
    ),
    # --- loss / optimizer / scheduler / budget
    dict(
        component="loss",
        upstream_file="modules/torch_demo.py / modules/classifier.py",
        upstream_symbol="CombinedLoss (CE + trip_loss_weight*triplet) / "
                        "CrossEntropyLoss",
        role="zip_config_index=6 -> use_triple_loss=True",
        consumes_mask=False,
        consumes_bbox=False,
        consumes_roi=False,
        annotation_dependency="none",
        planned_status="preserved",
    ),
    dict(
        component="optimizer",
        upstream_file="modules/classifier.py",
        upstream_symbol="optim.AdamW(lr=1e-4, eps=1e-6) inside train_compact",
        role="official head fine-tuning optimizer",
        consumes_mask=False,
        consumes_bbox=False,
        consumes_roi=False,
        annotation_dependency="none",
        planned_status="preserved",
    ),
    dict(
        component="scheduler",
        upstream_file="modules/classifier.py",
        upstream_symbol="CosineAnnealingLR(optimizer, ft_epo, eta_min=lr*0.1)",
        role="official head fine-tuning schedule",
        consumes_mask=False,
        consumes_bbox=False,
        consumes_roi=False,
        annotation_dependency="none",
        planned_status="preserved",
    ),
    dict(
        component="step_budget",
        upstream_file="param_space.py",
        upstream_symbol="debug.ft_epo=500",
        role="official classifier fine-tuning step budget",
        consumes_mask=False,
        consumes_bbox=False,
        consumes_roi=False,
        annotation_dependency="none",
        planned_status="preserved",
    ),
    # --- feature aggregation
    dict(
        component="feature_aggregation",
        upstream_file="modules/bbox_clip/alpha_clip_backbone.py",
        upstream_symbol="AlphaClip.encode_image k_patch top-k selection",
        role="top-k local patch selection by cosine similarity to the global "
             "token (feature-driven, annotation-free)",
        consumes_mask=False,
        consumes_bbox=False,
        consumes_roi=False,
        annotation_dependency="none",
        planned_status="preserved",
    ),
    dict(
        component="view_aggregation",
        upstream_file="modules/model.py",
        upstream_symbol="ClipModel.forward / torch_demo.weighted_logits_with_uncertainty",
        role="multi-view logit assembly (assemble_uncertainty); fvns/fvnq=1 "
             "keeps the CLS token only",
        consumes_mask=False,
        consumes_bbox=False,
        consumes_roi=False,
        annotation_dependency="none (view count is a fixed global list now)",
        planned_status="preserved",
    ),
    # --- support construction / decision rule
    dict(
        component="support_prototype",
        upstream_file="modules/model.py",
        upstream_symbol="ClipModel.set_img_prototype",
        role="cache support features per class -> init head weights",
        consumes_mask=True,
        consumes_bbox=False,
        consumes_roi=False,
        annotation_dependency="upstream feeds mask-derived features; the "
                              "adaptation feeds whole-image features",
        planned_status="interface_adapted",
        reason="identical official logic; input comes from the whole-image "
               "RGB dataset instead of mask/ROI crops",
    ),
    dict(
        component="decision_rule",
        upstream_file="modules/classifier.py",
        upstream_symbol="EchoClassfierF.forward (logits/tau softmax, tau=0.11)",
        role="support-conditioned inference decision rule",
        consumes_mask=False,
        consumes_bbox=False,
        consumes_roi=False,
        annotation_dependency="none",
        planned_status="preserved",
    ),
    # --- training loop / checkpoint
    dict(
        component="training_loop",
        upstream_file="modules/classifier.py",
        upstream_symbol="EchoClassfierF.train_compact",
        role="official compact fine-tuning loop",
        consumes_mask=False,
        consumes_bbox=False,
        consumes_roi=False,
        annotation_dependency="none",
        planned_status="preserved",
    ),
    dict(
        component="checkpoint_selection",
        upstream_file="run.py / fewshot_process.py",
        upstream_symbol="main_process (no checkpoint load in eval path)",
        role="final trained head state is used directly; no query-selected "
             "checkpoint",
        consumes_mask=False,
        consumes_bbox=False,
        consumes_roi=False,
        annotation_dependency="none",
        planned_status="preserved",
    ),
]

PRIVILEGED_READ_PATHS = [
    {
        "field": "mask",
        "paths": [
            "modules/multi_view_data.py: MultiScaleRoiSampleProcess.point2mask",
            "modules/multi_view_data.py: MultiScaleRoiSampleProcess.__call__ (roi_mask_list)",
            "modules/clip_preprocess.py: ClipPreprocess.__call__ (sample['masks'])",
            "modules/model.py: ClipModel.get_mvrec (input_dict['masks'] -> encode_image)",
            "modules/bbox_clip/alpha_clip_backbone.py: AlphaClip.encode_image (alpha_list -> conv1_alpha)",
        ],
    },
    {
        "field": "bbox",
        "paths": [
            "modules/fabric_data.py: FabricData.load (x1/x2/y1/y2 CSV)",
            "modules/multi_view_data.py: MultiScaleRoiGenerator.gen_one (bbox2rect)",
            "modules/clip_preprocess.py: ClipPreprocess.__call__ (bbox resize)",
            "modules/utils.py: Bbox.from_dict",
        ],
    },
    {
        "field": "roi",
        "paths": [
            "modules/multi_view_data.py: MultiScaleRoiGenerator.gen_one (roi rectangle)",
            "modules/multi_view_data.py: MultiScaleRoiSampleProcess.__call__ (roi_samples)",
            "modules/utils.py: RoiGenerator.gen_one",
        ],
    },
    {
        "field": "points",
        "paths": [
            "modules/multi_view_data.py: MultiScaleRoiSampleProcess.__call__ (src_bbox.kwargs['points'])",
            "modules/multi_view_data.py: MultiScaleRoiSampleProcess.point2mask",
        ],
    },
    {
        "field": "crop_center",
        "paths": [
            "modules/multi_view_data.py: MultiScaleRoiGenerator.gen_one (center_x/center_y from bbox)",
        ],
    },
    {
        "field": "crop_scale",
        "paths": [
            "modules/multi_view_data.py: MultiScaleRoiGenerator.gen_roi_size_list (bbox_min_size)",
            "modules/multi_view_data.py: MultiScaleRoiGenerator.gen_one (size from roi_size_list)",
        ],
    },
    {
        "field": "crop_offset",
        "paths": [
            "modules/multi_view_data.py: MvParam (x_step_list/y_step_list offsets)",
        ],
    },
]
