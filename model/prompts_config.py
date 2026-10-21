"""
Multi-prompt fusion configuration
===================================================

Text prompt templates used by the image-text contrastive (ITC) loss. Here you can:
1. change the default templates;
2. add templates for a specific dataset;
3. override the global defaults.

Usage:
    from model.prompts_config import get_prompt_templates

    # 1. Global defaults (4 groups, 14 templates)
    templates = get_prompt_templates()

    # 2. Dataset-specific templates, loaded automatically if defined in DATASET_PROMPTS
    templates = get_prompt_templates(dataset_name="MT_Dataset")

    # 3. A dict that overrides selected groups
    templates = get_prompt_templates({
        "direct": ["a {}"],
        "classification": ["category: {}"],
    })

    # 4. A complete list of templates that replaces all others
    my_list = ["a {}", "the {}", "showing {}", "classified as {}"]
    templates = get_prompt_templates(my_list)
"""

# ============================================================
# Global default prompt templates
# ============================================================
# Templates can be added, removed or edited below without other code changes.
# Every template uses {} as the placeholder for the class description.
#
# +============================================================+
# |   Recommended pairing:                                     |
# |   --cross_modal_mode strip  -> strip_aware templates       |
# |   --cross_modal_mode scale  -> scale_aware templates       |
# |   More specific REAL_NAMES give a stronger ITC signal      |
# +============================================================+
# ============================================================

PROMPTS = {

    # ===== Direct description =====

    # Plain full description without any wrapper,
    # e.g. "magnetic tile with blowhole defect" (different for each of the 6 classes)
    "direct": [
        "{}",               # "magnetic tile with blowhole defect"
        "a {}",             # "a magnetic tile with blowhole defect"
        "the {}",           # "the magnetic tile with blowhole defect"
        "this is {}",       # "this is magnetic tile with blowhole defect"
    ],

    # ===== Contextual (inspection scene) =====
    # Describes an industrial inspection scene, so that the model reads the text as a defect type;
    # these prompts widen the semantic differences between defect names
    # (blowhole/break/crack/fray/uneven) and support the CE classification
    "contextual": [
        "showing {}",               # "showing magnetic tile with blowhole defect"
        "inspecting {}",            # "inspecting magnetic tile with blowhole defect"
        "checking for {}",          # "checking for magnetic tile with blowhole defect"
        "detecting {}",             # "detecting magnetic tile with blowhole defect"
    ],

    # ===== Classification =====
    # Phrased as a classifier output, which ties the class name to its description
    "classification": [
        "classified as {}",         # "classified as magnetic tile with blowhole defect"
        "category: {}",             # "category: magnetic tile with blowhole defect"
        "result: {}",               # "result: magnetic tile with blowhole defect"
    ],

    # ===== CLIP standard style =====
    # Matches the image-text distribution of CLIP pre-training (photo captions)
    "clip_style": [
        "a photo of {}.",           # "a photo of magnetic tile with blowhole defect."
        "an image of {}.",          # "an image of magnetic tile with blowhole defect."
        "a picture of {}.",         # "a picture of magnetic tile with blowhole defect."
    ],
}

# ---- Dataset-specific prompt overrides ----
# Add an entry here when a dataset needs entirely different prompts.
# Format: "dataset name": [templates] or {"key": [templates]}
# Example:
DATASET_PROMPTS = {

    "MT_Dataset": {
        # Direct description: short and explicit
        "direct": [
            "{}",                    # "magnetic tile with blowhole defect"
            "a {}",                  # "a magnetic tile with blowhole defect"
        ],
        
        # Industrial inspection scene: stresses the inspection context and the defect type
        "contextual": [
            "inspecting {}",         # "inspecting magnetic tile with blowhole defect"
            "detecting {}",          # "detecting magnetic tile with blowhole defect"
            "quality check: {}",     # "quality check: magnetic tile with blowhole defect"
            "defect type: {}",       # "defect type: magnetic tile with blowhole defect"
        ],
        
        # Classification: phrased as a classifier output
        "classification": [
            "classified as {}",      # "classified as magnetic tile with blowhole defect"
            "category: {}",          # "category: magnetic tile with blowhole defect"
            "label: {}",             # "label: magnetic tile with blowhole defect"
        ],
        
        # Strip-aware templates (use with --cross_modal_mode strip)
        # Stress linear, strip-like and elongated shapes; complement the H/V strip pooling of the strip path
        "strip_aware": [
            "linear defect: {}",
            "horizontal scratch: {}",
            "vertical crack: {}",
            "elongated strip: {}",
            "line pattern: {}",
        ],
        
        # Scale-aware templates (use with --cross_modal_mode scale)
        # Stress size, extent and density of a defect; complement the multi-scale pooling of the scale path
        "scale_aware": [
            "tiny spot defect: {}",
            "large area defect: {}",
            "small dot: {}",
            "big patch: {}",
            "dense region: {}",
            "sparse scatter: {}",
        ],
        
        # Normal samples only: no defect, conforming part
        "normal_sample": [

            "a perfect {}",          # "a perfect normal magnetic tile in perfect condition"
            "an ideal {}",           # "an ideal normal magnetic tile in perfect condition"
            "flawless {}",           # "flawless normal magnetic tile in perfect condition"
            "qualified {}",          # "qualified normal magnetic tile in perfect condition"
        ],
        
        # CLIP standard style: keeps the pre-training distribution
        "clip_style": [
            "a photo of {}.",        # "a photo of magnetic tile with blowhole defect."
            "an image of {}.",       # "an image of magnetic tile with blowhole defect."
        ],
    },

    # ===============================================================
    # PVEL-AD photovoltaic panel defects
    # 8 defect classes (4 frequent and 4 image-level):
    #
    #   # Linear / strip-like defects (strip_aware)
    #   +-- crack         - irregular thin dark line
    #   +-- finger        - interrupted grid finger (strictly along the grid line)
    #   +-- thick_line    - thick grid line (very wide, blurred edges)
    #   +-- horizontal_dislocation - horizontal dislocation
    #   +-- vertical_dislocation   - vertical dislocation
    #
    #   # Area / spot defects (scale_aware)
    #   +-- black_core    - large dark central blob (blurred edges)
    #   +-- short_circuit - whole cell darkened (no clear outline)
    #   +-- star_crack    - radial pattern from a centre (lines and points)
    #
    #   CrossModalAttention uses these texts as queries to locate
    #   the matching local regions among the 1369 patches
    # ===============================================================
    "PVEL-AD": {
        # Direct description
        "direct": [
            "{}",
            "a {}",
        ],
        # Photovoltaic inspection scene: electroluminescence (EL) imaging context
        "contextual": [
            "electroluminescence image of {}",
            "solar panel showing {}",
            "photovoltaic cell with {}",
            "inspecting {} on solar cell",
            "defect type: {}",
        ],
        # Classification
        "classification": [
            "classified as {}",
            "category: {}",
            "label: {}",
            "diagnosis: {}",
        ],

        # ===========================================================
        # Strip-path templates
        # Use with --cross_modal_mode strip
        # These prompts stress linear, strip-like and directional shapes,
        # complementing the horizontal/vertical strip pooling of the strip path
        # ===========================================================
        "strip_aware": [
            # Basic strip descriptions
            "linear pattern: {}",
            "thin dark line: {}",
            "elongated strip: {}",
            "directional scratch: {}",
            "line-shaped anomaly: {}",
            # Direction
            "horizontal dark stripe: {}",
            "vertical dark line: {}",
            "grid-aligned defect: {}",
            "broken line pattern: {}",
            # Thick versus thin
            "narrow crack line: {}",
            "thick wide stripe: {}",
            "intermittent dashed line: {}",
            "shifted misaligned line: {}",
        ],

        # ===========================================================
        # Scale-path templates
        # Use with --cross_modal_mode scale
        # These prompts stress size, extent, density and shape,
        # complementing the multi-scale pooling of the scale path
        # ===========================================================
        "scale_aware": [
            # Scale
            "tiny dark spot: {}",
            "small dot defect: {}",
            "medium patch: {}",
            "large dark area: {}",
            "big region anomaly: {}",
            # Shape
            "blob-shaped dark region: {}",
            "circular dark patch: {}",
            "irregular dark shape: {}",
            "scattered dark specks: {}",
            "dense cluster: {}",
            "fully darkened cell: {}",
            "radial burst pattern: {}",
            "star-shaped crack: {}",
            "large blackened area: {}",
            "central dark region: {}",
        ],

        # CLIP standard style
        "clip_style": [
            "a photo of {}.",
            "an image of {}.",
            "a EL image of {}.",
        ],
    },

    # ===============================================================
    # MVTec AD Cable - targeted prompt templates (one semantic group per class)
    #
    #   good                  -> normal_sample (intact insulation, no repeated colour, straight)
    #   cable_swap            -> color_duplicate_aware (repeated or swapped colour)
    #   bent_wire             -> geometry_aware (physical bending)
    #   missing_wire          -> missing_wire_aware (shadow / gap)
    #   cut_inner_insulation  -> surface_damage_aware (transverse cut)
    #   cut_outer_insulation  -> surface_damage_aware (transverse cut)
    #   poke_insulation       -> surface_damage_aware (local puncture)
    #   combined              -> combined_aware (several defects at once)
    # ===============================================================
    "MVTecAD-Cable": {
        # 1. Normal samples
        # Key cues: intact insulation, no repeated colour, straight wires
        "normal_sample": [
            "A flawless {} with completely intact insulation and distinct, non-repeating colored wires",
            "An ideal {}, showing the correct multi-color coding with exactly one wire of each color",
            "A perfectly straight {}, defect-free and structurally sound",
        ],

        # 2. Repeated colour: cable swap (e.g. two blue wires)
        # Key cues: colour repetition, no spatial-crossing semantics
        "color_duplicate_aware": [
            "{} showing a color sequence error with duplicate colored wires",
            "Wire assembly indicating incorrect color substitution, such as two wires of the same color",
            "{} with repeated wire colors instead of distinct ones",
        ],

        # 3. Geometric deformation: bent wire
        # Key cues: physical bending, no surface damage
        "geometry_aware": [
            "{} exhibiting a sharp physical kink or abnormal bent structure",
            "Deformed {}, showing structural curvature but intact surface",
        ],

        # 4. Spatial gap: missing wire (shadow and spacing left by the missing wire)
        # Key cues: dark gap, irregular spacing, structural hole
        "missing_wire_aware": [
            "{} showing an abnormal dark gap between wires due to a missing strand",
            "wire bundle with irregular spacing, revealing a structural void where a wire should be",
            "incomplete {} missing a color component, leaving an exposed dark shadow",
        ],

        # 5. Surface damage: cut and poke
        # Key cues: transverse cut, local puncture
        "surface_damage_aware": [
            "{} with a transverse cut severing the insulation",
            "Severely damaged cable showing a deep horizontal slice",
            "{} with a localized puncture or irregular poked hole on the sheath",
        ],

        # 6. Combined defects
        # Key cues: several anomalies at once, kept apart from the other templates
        "combined_aware": [
            "Severely defective {} showing multiple simultaneous anomalies, such as bent structure combined with surface damage",
            "{} exhibiting a complex mix of physical deformation and insulation destruction",
        ],
    },
}


# ===============================================================
# Defect class -> prompt groups
# ===============================================================
# Different defects call for different template groups:
#   - linear or strip-like defects (cracks, scratches) -> strip_aware
#   - round, spot-like or large defects (blowholes, breaks) -> scale_aware
#   - normal samples -> only normal_sample + clip_style (no defect semantics)
# Format: {dataset name: {class name: [prompt group keys]}}
# ===============================================================

CLASS_PROMPT_GROUPS = {
    "MT_Dataset": {
        # Linear crack -> mainly strip_aware
        "MT_Crack": ["direct", "contextual", "classification", "clip_style", "strip_aware"],
        # Round blowhole -> mainly scale_aware
        "MT_Blowhole": ["direct", "contextual", "classification", "clip_style", "scale_aware"],
        # Large break -> mainly scale_aware (large area)
        "MT_Break": ["direct", "contextual", "classification", "clip_style", "scale_aware"],
        # Frayed edge -> strip (edge strips) + scale (degree of wear)
        "MT_Fray": ["direct", "contextual", "classification", "clip_style", "strip_aware", "scale_aware"],
        # Uneven surface -> scale-aware
        "MT_Uneven": ["direct", "contextual", "classification", "clip_style", "scale_aware"],
        # Normal samples -> normal templates only, free of defect semantics
        "MT_Free": ["normal_sample", "clip_style"],
    },
    # ============================================================
    # PVEL-AD photovoltaic defect classes
    # (from a defect-by-defect analysis of the dataset)
    #
    #   Criteria:
    #   +--------------+----------+----------+----------------------+
    #   | defect        | shape      | position   | distinguishing cue     |
    #   +--------------+----------+----------+----------------------+
    #   | black_core    | blob       | centre     | blurred edge, gradient |
    #   | crack         | thin line  | anywhere   | irregular course       |
    #   | finger        | broken line| along grid | strictly along grid    |
    #   | h_dislocation | horizontal | middle     | horizontal break       |
    #   | short_circuit | dark area  | large area | fully dark, no outline |
    #   | star_crack    | radial     | centre     | radial structure       |
    #   | thick_line    | thick line | along grid | very wide, blurred     |
    #   | v_dislocation | vertical   | middle     | vertical break         |
    #   +--------------+----------+----------+----------------------+
    # ============================================================
    "PVEL-AD": {
        "black_core":               ["direct", "contextual", "classification", "clip_style", "scale_aware"],
        "crack":                    ["direct", "contextual", "classification", "clip_style", "strip_aware"],
        "finger":                   ["direct", "contextual", "classification", "clip_style", "strip_aware"],
        "horizontal_dislocation":   ["direct", "contextual", "classification", "clip_style", "strip_aware"],
        "short_circuit":            ["direct", "contextual", "classification", "clip_style", "scale_aware"],
        "star_crack":               ["direct", "contextual", "classification", "clip_style", "strip_aware", "scale_aware"],
        "thick_line":               ["direct", "contextual", "classification", "clip_style", "strip_aware", "scale_aware"],
        "vertical_dislocation":     ["direct", "contextual", "classification", "clip_style", "strip_aware"],
    },
    "MVTecAD-Cable": {
        # good -> normal templates only
        "good": ["normal_sample"],
        # cable_swap -> repeated colour
        "cable_swap": ["color_duplicate_aware"],
        # bent_wire -> deformation
        "bent_wire": ["geometry_aware"],
        # missing_wire -> shadow / gap
        "missing_wire": ["missing_wire_aware"],
        # cut -> transverse cut
        "cut_inner_insulation": ["surface_damage_aware"],
        "cut_outer_insulation": ["surface_damage_aware"],
        # poke_insulation -> puncture
        "poke_insulation": ["surface_damage_aware"],
        # combined -> several defects at once
        "combined": ["combined_aware"],
        # missing_cable -> global absence (reuses the gap semantics of missing_wire_aware)
        "missing_cable": ["missing_wire_aware"],
    },
}



def get_prompt_templates(prompts=None, dataset_name=None):
    """Return all prompt templates, with optional overrides.

    Priority (highest first):
    1. dataset-specific templates, if dataset_name is in DATASET_PROMPTS;
    2. the templates passed in prompts (override the global defaults);
    3. the global defaults PROMPTS.

    Args:
        prompts: optional custom prompt configuration, either a dict
                 (overrides the matching keys of PROMPTS)
                 or a list (replaces all templates).
        dataset_name: optional dataset name used to load dataset-specific templates.

    Returns:
        prompt_templates: list[str], all templates without duplicates
    """

    # 1. Base prompt source
    if dataset_name and dataset_name in DATASET_PROMPTS:
        # Dataset-specific override
        base = DATASET_PROMPTS[dataset_name]
    else:
        base = PROMPTS

    # 2. Apply the custom prompts
    if prompts is not None:
        if isinstance(prompts, dict):
            # Shallow override: only keys present in the dict are replaced
            merged = dict(base)  # copy base
            merged.update(prompts)  # override with custom
            base = merged
        elif isinstance(prompts, list):
            # A list replaces all templates
            return _deduplicate(prompts)
        else:
            raise TypeError(f"prompts must be dict or list, got {type(prompts)}")

    # 3. Merge all templates into one list
    all_templates = []
    if isinstance(base, dict):
        for key, templates in base.items():
            if isinstance(templates, list):
                all_templates.extend(templates)
    elif isinstance(base, list):
        all_templates = base
    else:
        raise TypeError(f"base must be dict or list, got {type(base)}")

    return _deduplicate(all_templates)


def _deduplicate(templates):
    """Remove duplicates, keeping the order."""
    seen = set()
    result = []
    for t in templates:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


# ============================================================
# Usage examples
# ============================================================
#
# Case 1: change the default templates
#   -> edit the PROMPTS dict above; the training script loads it automatically.
#
# Case 2: templates for a specific dataset
#   -> add them to DATASET_PROMPTS:
#       DATASET_PROMPTS["MyDataset"] = ["a {} in warehouse", "the {} on conveyor"]
#   -> and train with --data MyDataset.
#
# Case 3: temporary override at training time
#   -> pass custom prompts to train_multiclass.py:
#       python train_multiclass.py --custom_prompts "a {} on belt" "the {} in factory" ...
#   (not implemented yet; edit PROMPTS instead)
#
# Case 4: fully custom logic
#   -> subclass or rewrite compute_text_prototypes()
#   -> see compute_text_prototypes in train_multiclass.py
