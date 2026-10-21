import os

# project root, derived from the location of this file
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# can be overridden with the environment variable AACLIP_BASE_PATH (Linux/macOS)
if os.name != 'nt':
    BASE_PATH = os.environ.get("AACLIP_BASE_PATH", BASE_PATH)

DATA_PATH = {
    "Brain": f"{BASE_PATH}/data/MedAD/Brain_AD",
    "Liver": f"{BASE_PATH}/data/MedAD/Liver_AD",
    "Retina": f"{BASE_PATH}/data/MedAD/Retina_RESC_AD",
    "Colon_clinicDB": f"{BASE_PATH}/data/Colon/CVC-ClinicDB",
    "Colon_colonDB": f"{BASE_PATH}/data/Colon/CVC-ColonDB",
    "Colon_cvc300": f"{BASE_PATH}/data/Colon/CVC-300",
    "Colon_Kvasir": f"{BASE_PATH}/data/Colon/Kvasir",
    "BTAD": f"{BASE_PATH}/data/BTech_Dataset_transformed",
    "MPDD": f"{BASE_PATH}/data/MPDD",
    "MVTec": f"{BASE_PATH}/data/mvtec_ad",
    "VisA": f"{BASE_PATH}/data/VisA_20220922",
    "MT_Dataset": f"{BASE_PATH}/data/mt_dataset",
    "PVEL-AD": f"{BASE_PATH}/data/PVEL-AD",
    "MVTecAD-Cable": f"{BASE_PATH}/data/mvtecad-cable",
}

CLASS_NAMES = {
    "Brain": ["Brain"],
    "Liver": ["Liver"],
    "Retina": ["Retina"],
    "Colon_clinicDB": ["Colon_clinicDB"],
    "Colon_colonDB": ["Colon_colonDB"],
    "Colon_Kvasir": ["Kvasir"],
    "Colon_cvc300": ["CVC-300"],
    "MVTec": [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "transistor",
        "toothbrush",
        "wood",
        "zipper",
    ],
    "VisA": [
        "candle",
        "pcb3",
        "capsules",
        "pipe_fryum",
        "pcb4",
        "macaroni2",
        "pcb2",
        "chewinggum",
        "macaroni1",
        "cashew",
        "fryum",
        "pcb1",
    ],
    "MPDD": [
        "connector",
        "tubes",
        "metal_plate",
        "bracket_white",
        "bracket_brown",
        "bracket_black",
    ],
    "BTAD": ["01", "02", "03"],
    "MT_Dataset": [
        "MT_Blowhole",
        "MT_Break",
        "MT_Crack",
        "MT_Fray",
        "MT_Free",
        "MT_Uneven",
    ],
    "PVEL-AD": [
        "black_core",
        "crack",
        "finger",
        "horizontal_dislocation",
        "short_circuit",
        "star_crack",
        "thick_line",
        "vertical_dislocation",
    ],
    "MVTecAD-Cable": [
        "good",
        "bent_wire",
        "cable_swap",
        "combined",
        "cut_inner_insulation",
        "cut_outer_insulation",
        "missing_cable",
        "missing_wire",
        "poke_insulation",
    ],
}
DOMAINS = {
    "VisA": "Industrial",
    "BTAD": "Industrial",
    "MPDD": "Industrial",
    "MVTec": "Industrial",
    "MT_Dataset": "Industrial",
    "Brain": "Medical",
    "Liver": "Medical",
    "Retina": "Medical",
    "Colon_clinicDB": "Medical",
    "Colon_colonDB": "Medical",
    "Colon_Kvasir": "Medical",
    "Colon_cvc300": "Medical",
    "PVEL-AD": "Industrial",
    "MVTecAD-Cable": "Industrial",
}
REAL_NAMES = {
    "Brain": {"Brain": "scan"},
    "Liver": {"Liver": "scan"},
    "Retina": {"Retina": "scan"},
    "MVTec": {
        "bottle": "dark bottle",
        "cable": "top view of three cables",
        "capsule": "black and orange capsule",
        "carpet": "gray carpet",
        "grid": "metal or plastic mesh",
        "hazelnut": "single brown hazelnut",
        "leather": "brown leather",
        "metal_nut": "metal nut which has four notched edges",
        "pill": "oval white pill with small red speckles and the letters 'FF' engraved",
        "screw": "screw",
        "tile": "speckled tile surface",
        "transistor": "a three-legged transistor placed vertically",
        "toothbrush": "toothbrush head",
        "wood": "wood surface",
        "zipper": "a black zipper",
    },
    "VisA": {
        "candle": "candle",
        "pcb3": "infrared sensor pcb module",
        "capsules": "capsules",
        "pipe_fryum": "pipe-shaped fryum",
        "pcb4": "battery charging pcb module",
        "macaroni2": "scattered yellow macaroni",
        "pcb2": "integrated circuits board",
        "chewinggum": "chewing gum",
        "macaroni1": "orange macaroni",
        "cashew": "cashew nut",
        "fryum": "wheel-shaped fryum snack",
        "pcb1": "dual ultrasonic distance sensor pcb module",
    },
    "Colon_clinicDB": {
        "Colon_clinicDB": "colon endoscopy image",
    },
    "Colon_colonDB": {
        "Colon_colonDB": "colon endoscopy image",
    },
    "Colon_cvc300": {"CVC-300": "colon endoscopy image"},
    "Colon_Kvasir": {"Kvasir": "colon endoscopy image"},
    "MPDD": {
        "connector": "metal clamps with black adjustment knobs",
        "tubes": "scattered metal objects",
        "metal_plate": "blue rectangular metal plate with a notch on one side",
        "bracket_white": "white, elongated triangular metal bracket with a smooth, matte finish",
        "bracket_brown": "brown L-shaped metal bracket with smooth, glossy finish and multiple mounting holes along its arms",
        "bracket_black": "black ornamental metal bracket with spiral design attached to a rectangular frame",
    },
    "BTAD": {
        "01": "Bright concentric rings in neon yellow and blue tones against a dark blue background, resembling a stylized wave or energy field radiating outward.",
        "02": "vertical fabric lines in warm, dusty pink and beige tones",
        "03": "oval concentric circular rings in gradient shades of blue and white",
    },
    "MT_Dataset": {
        "MT_Blowhole": "magnetic tile with a small circular blowhole defect, tiny hole with irregular porous edges",
        "MT_Break":   "magnetic tile with a large broken corner defect, large missing chunk with chipped fracture edge",
        "MT_Crack":   "magnetic tile with a thin elongated crack defect, long linear scratch-like fracture running across the surface",
        "MT_Fray":    "magnetic tile with frayed edge defect, worn and rough frayed border with fibrous strip-like texture",
        "MT_Free":    "perfect normal magnetic tile in flawless condition, smooth intact surface without any defects or damage",
        "MT_Uneven":  "magnetic tile with uneven surface defect, irregular bumpy texture with visible raised spots and dips",
    },
    # PVEL-AD descriptions based on detailed defect feature analysis
    # All descriptions are in pure English to ensure CLIP tokenizer compatibility
    "PVEL-AD": {
        "black_core": "large irregular dark blob in the solar cell center with blurred gradient edges and uneven internal brightness, dark core patch occupying big region",
        "crack": "thin irregular dark line running across the solar cell surface with uniform narrow width, fine fracture line with no fixed direction appearing at random positions",
        "finger": "intermittent dark broken segments strictly along the grid finger lines, disconnected grid line with dash pattern on the solar cell surface",
        "horizontal_dislocation": "horizontal shift offset of grid lines creating a misalignment across the cell middle, brightness discontinuity between upper and lower parts",
        "short_circuit": "completely dark or largely blackened solar cell with extremely low luminescence over large area, fully darkened cell without clear defect boundary",
        "star_crack": "radial star-shaped crack pattern with multiple thin dark lines spreading outward from a central impact point, snowflake-like branching crack structure",
        "thick_line": "abnormally wide dark thick grid line with blurred edges and lateral spread broader than normal fingers, appearing as a thick dark stripe along the grid direction",
        "vertical_dislocation": "vertical shift offset of grid lines creating a misalignment across the cell middle, brightness difference between left and right sections",
    },
    # MVTec AD Cable - physical descriptions tuned to the gated paths of the cross-attention
    # each description uses keywords that address a specific gated path
    "MVTecAD-Cable": {
        # [all gates closed]: strong negations, so that every defect path outputs a zero residual
        "good": "flawless, perfect industrial cable with intact insulation and correct wire colors, showing absolutely no defects, no cuts, and no anomalies",

        # [direct path (global / colour)]: wrong and missing colours
        "cable_swap": "defective cable with swapped wire colors, focusing on the incorrect color sequence and the unexpected absence of the correct hue",

        # [scale path (multi-scale / blobs)] + [strip path]: mixed defects of different sizes, holes and cuts
        "combined": "heavily damaged cable with mixed and combined defects of various sizes, containing both directional cuts and localized puncture holes",

        # [strip path (direction / strips)]: the straight continuous structure is broken
        "bent_wire": "defective cable with a severely bent wire, disrupting the straight continuous line of the inner structure",

        # [strip path]: transverse cut
        "cut_inner_insulation": "defective cable showing a sharp, transverse cut on the inner color insulation, breaking the continuous edge",
        "cut_outer_insulation": "defective cable showing a sharp, transverse cut on the outer black insulation, breaking the continuous edge",

        # [direct path + strip path]: a longitudinal track is missing entirely
        "missing_wire": "defective cable missing a single internal color wire, leaving an empty longitudinal track along the straight direction",
        "missing_cable": "missing cable with the entire structure completely absent, showing a totally empty background",

        # [scale path]: irregular local round spots / holes
        "poke_insulation": "defective cable with a localized poke or puncture hole on the surface, exhibiting an irregular spot of varying scale",
    },
}
# PROMPTS has moved to model/prompts_config.py;
# kept here for backward compatibility; edit model/prompts_config.py instead
from model.prompts_config import PROMPTS  # noqa: F401
