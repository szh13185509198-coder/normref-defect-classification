# Normal-referenced defect evidence for few-shot industrial defect classification

Code, protocols, result files and figures for the paper

> *Normal-referenced defect evidence for training-free few-shot industrial defect classification with calibrated
> conformity screening*
> (submitted to *Measurement Science and Technology*; citation to be added on publication).

The method decides whether a part conforms and, if not, which defect type it carries, from a few image-level
examples per defect type and no parameter training. Twenty qualified (conforming) images of the product define the
nominal patch variability:
- **evidence selection**: query patches are scored against the nominal patch bank; a leave-one-out 0.99 quantile of
  the bank's own scores sets the cutoff, and patches above it are pooled with weights proportional to their
  exceedance (the *calibrated soft region*);
- **classification**: an appearance descriptor, a deviation descriptor (patch minus its nearest nominal patch) and the
  global SimpleShot score are z-normalized and summed (the *dual-stream method*);
- **conformity screening**: a split-conformal acceptance limit from N = 100 further qualified parts sets the expected
  false-reject rate in advance; a three-zone variant sends uncertain parts to manual review;
- **uncertainty evaluation**: a GUM-aligned budget of the benchmark accuracy, single-calibration false-reject
  dispersion, decision-risk intervals, influence quantities and device latency / energy.

The deployed form is a single TensorRT engine whose nominal bank, threshold and prototypes are runtime inputs, so
adding a product or a defect type never rebuilds the engine.

This repository does **not** contain any dataset, model weight or third-party code that is not under a permissive
license (see [Licenses](#licenses)).

## Names in the code and in the paper

The code and the protocols use the internal names of the evaluation rounds:

| In the code / protocols | In the paper |
|---|---|
| `ARR`, "dual-stream (ARR)", main C1 method | dual-stream method (proposed) |
| `A`, normal-calibrated soft region | calibrated soft region (appearance stream only) |
| `H1`, fixed-ratio normal reference | fixed-ratio normal reference |
| `H3` / `CONS` | cross-class contrast / consensus background |
| `SS`, SimpleShot (CL2N) | SimpleShot |
| `P6.B`, `P10` … `P49` | evaluation rounds, each with its protocol `docs/P*.md` (Supplementary Material S2) |
| `U1`–`U6` (round P47) | uncertainty evaluation (Section 5.5, Supplementary Material S16) |

## Where each part of the paper comes from

Main sources only; Supplementary Material S2 lists every round and what it decided.

| Paper | Rounds | Code | Results |
|---|---|---|---|
| Section 6.1, Table 1, Figure 4 (where accuracy is lost) | Revision C, P10, P21, P27 | `experiments/revision`, `p10`, `p21`, `p27` | `ckpt/revision_c`, `ckpt/p10_*`, `ckpt/p21`, `ckpt/p27` |
| Section 6.2, Table 2, Figure 5 (classification) | P14, P19 (Real-IAD blind test), P22 | `experiments/p14`, `p19`, `p22` | `ckpt/p14_formal`, `ckpt/p19`, `ckpt/p22` |
| Section 6.3 (same-backbone comparison) | P21, P24, P25, P29 (CLIP-LoRA), P44 (GDA) | `experiments/p21`, `p24`, `p25`, `p29`, `p44` | `ckpt/p21`, `ckpt/p24`, `ckpt/p25`, `ckpt/p29`, `ckpt/p44` |
| Section 6.4, Table 3, Figure 6 (contribution of each step) | P10–P14, P12 (threshold sources), P17, P18 | `experiments/p10`–`p14`, `p17`, `p18` | `ckpt/p10_*`–`ckpt/p14_formal`, `ckpt/p12`, `ckpt/p17`, `ckpt/p18` |
| Section 6.5, Tables 4–5, Figures 7–8 (conformity screening and its risks) | P15, P46, P47 (U1-D, U2-B) | `experiments/p15`, `p46`, `p47` | `ckpt/p15`, `ckpt/p46`, `ckpt/p47` |
| Section 6.6, Table 6 (uncertainty of the reported performance) | P47 (U3, U5-A, U6-D) | `experiments/p47` | `ckpt/p47` |
| Section 6.7, Table 7, Figure 9 (influence quantities and recalibration) | P47 (U4-A, decision agreement) | `experiments/p47` | `ckpt/p47` |
| Section 6.8, Figure 10 (deployment cost) | P26, P30–P46, P49 | `experiments/p26`, `p30`, `p32`–`p46`, `revision/c3_resource_profile.py` | `ckpt/p26*`, `ckpt/p30`–`ckpt/p46`, `ckpt/p49` |
| Statistics (cluster bootstrap, Holm) | P16, P20 | `experiments/p16`, `p20`, `revision` | — |
| Supplementary S2 (rounds, deviations, result files) | all | `docs/P*.md` | [`RESULTS.md`](RESULTS.md) |
| Supplementary S8 (deployment), S10 (industrial metrics), S11–S12 (calibration, purity, baselines) | P26–P49, P15, P40, P48, P17, P18, P44 | as above | as above |
| Supplementary S13 (Real-IAD), S14 (per product), S15 (error decomposition), S16 (uncertainty) | P19, P20, P28, P47 | `experiments/p19`, `p20`, `p28`, `p47` | `ckpt/p19`, `ckpt/p28`, `ckpt/p47` |

Each protocol in `docs/` was committed before the outputs it governs; [`docs/PROTOCOL_INDEX.md`](docs/PROTOCOL_INDEX.md) lists the
commit of every protocol, and the supplementary material (S2) lists the order of information use, the decision
criteria and every deviation. Protocol files are kept byte-identical to their committed versions; those written
in Chinese have an English translation in [`docs/translations/`](docs/translations/).

## Layout

| Path | Contents |
|---|---|
| `model/` | CLIP / DINOv2 feature extraction (CLIP code derived from AA-CLIP and open_clip) |
| `experiments/p6`, `train_multiclass.py` | Frozen support manifests and query contract; trainable support-side adaptation and the MVREC-RGB adaptation (baselines) |
| `experiments/p10`–`p14` | Normal-reference evidence selection, dual-stream method, formal and external evaluation (MVTec-FS, VisA, MTD, Cable) |
| `experiments/p15` | Two-stage inspection (split-conformal screening + type classification) |
| `experiments/p17`, `p18` | Robustness grid; adapted baselines, other backbones, evidence purity |
| `experiments/p19` | Real-IAD one-time blind test |
| `experiments/p21`, `p25`, `p29` | CLIP-LoRA (official code) runners and local scoring |
| `experiments/p22`–`p24`, `p27`, `p28` | Lightweight backbones, MVREC-RGB at 336 px, matched controls, VisA / MTD controls, error decomposition |
| `experiments/p16`, `p20` | Figures and statistics of the earlier rounds (Holm correction) |
| `experiments/p26`, `p30`, `p32`–`p46` | Deployment: timing, ONNX / TensorRT export, fused engine, Jetson measurements, GDA, line simulation, on-device screening |
| `experiments/p47` | Uncertainty evaluation U1–U6 (budget, calibration law, decision-risk intervals, influence quantities, repeatability, jackknife+, device latency / energy) and device scaling / tiling |
| `experiments/p48` | Patch-level false selection on the deployed engine |
| `experiments/revision` | Shared data interface, metrics, paired bootstrap, frozen-feature baselines, resource profile (also used by P49) |
| `docs/P*.md`, `docs/PROTOCOL_INDEX.md` | Protocols, each committed before the outputs it governs, and their commit record |
| `docs/translations/` | English translations of the protocols written in Chinese |
| `docs/reproduction/` | Full record of the supplementary material before condensation (all tables and run details) |
| `docs/p19`, `docs/p21`, `docs/paper/*_EVIDENCE_MANIFEST.json` | Real-IAD manifest, MVTec-FS episode list, hashes of all intermediate files of P10–P16 |
| `data/*_manifest_bundle` | Frozen support / query manifests (file names and hashes only) |
| `dataset/metadata` | Image lists for MVTec AD, VisA, MTD and the Cable split |
| `ckpt/` | Result files of the runs (JSON / JSONL / logs; no features, no images); see [`RESULTS.md`](RESULTS.md) |
| `paper_figures/` | Every figure of the paper: [`paper_figures/final`](paper_figures/final/README.md) holds them exactly as in the manuscript, with their sources, scripts and plotted data |
| `MVREC-main/` | Vendored MVREC source (Apache-2.0), used only for the MVREC-RGB baseline |
| `tests/` | Unit tests of the protocol code (no model loading) |

## Setup

Python 3.10+, PyTorch 2.x with CUDA. Install the dependencies:

```bash
pip install -r requirements.txt
```

The results in `ckpt/` were produced with Python 3.12, PyTorch 2.9.0 (CUDA 12.8) and kornia 0.6.9 on an
NVIDIA GeForce RTX 4050 Laptop GPU; rerunning the external evaluation in this environment reproduces every
published cell exactly. Other PyTorch builds run the same code but can shift individual balanced accuracies.
On Windows with conda, set `KMP_DUPLICATE_LIB_OK=TRUE` if two OpenMP runtimes are loaded.

The TensorRT parts (`p35`, `p38`, `p41`, `p43`–`p48`) also need TensorRT 10. They were run on a Jetson Orin Nano
Super 8 GB with JetPack 6.

### Weights (not included)

| File | Source |
|---|---|
| `model/ViT-L-14-336px.pt`, `model/ViT-B-16.pt` | OpenAI CLIP, downloaded by `clip.load` from the URLs in `model/openai.py` |
| `pretrained/dinov2_vits14_pretrain.pth`, `pretrained/dinov2_vitb14_pretrain.pth` | DINOv2 official release |
| `model/clip_l14_grit20m_fultune_2xe.pth`, `model/clip_l14_336_grit_20m_4xe.pth` | Alpha-CLIP release (MVREC-RGB baseline only) |

### Third-party code (not included)

See [`third_party/README.md`](third_party/README.md) for the upstream repositories and the pinned commits:
DINOv2, CLIP-LoRA and Alpha-CLIP.

### Datasets (not included)

Download each dataset from its official source and accept its license:
- MVTec AD (MVTec-FS annotations from the MVREC release);
- VisA;
- Magnetic Tile Defect (MTD);
- Real-IAD.

The manifests in this repository record the relative path and SHA-256 of every image used. The runners that verify
these hashes stop on any mismatch.

Expected locations (relative to the repository root):
- `data/mvtec-fs` and `data/mvtecad-cable` (the Cable split); the MVTec AD root itself is passed with `--mvtec-root`;
- `data/VisA_raw`;
- `data/MTD_raw`;
- `data/Real-IAD`.

`AACLIP_BASE_PATH` can point the dataset loaders of `dataset/constants.py` to another data root; by default they
use the repository root.

## Running

Every script is a module run from the repository root. The exact arguments are in each script's docstring and
protocol (the docstrings show the Windows paths of the original runs). The main results:

```bash
python -m experiments.p14.final_formal --stage mvtec --device cuda
python -m experiments.p14.final_formal --stage external --device cuda
python -m experiments.p15.two_stage --stage external --device cuda
python -m experiments.p20.stats_holm
```

Formal runners refuse to run with uncommitted changes to tracked code. Each formal output directory is written
once. Before computing anything new, each round reproduces earlier results bitwise, so later rounds need the
outputs of the rounds they build on. Run the rounds in order.

The figures are redrawn from the result files with the scripts listed in
[`paper_figures/final/README.md`](paper_figures/final/README.md).

Unit tests:

```bash
python -m unittest discover -s tests -t . -p "test_*.py"
```

Notes on the tests:
- `tests/core/test_initialization_consistency.py` needs `third_party/dinov2`, and `tests/protocols/test_p21.py`
  needs `third_party/CLIP-LoRA`.
- Run `tests.protocols.test_p6b_stage2_support_runner` on its own, with `HF_ENDPOINT` unset. Importing
  `train_multiclass` in an earlier test sets a default `HF_ENDPOINT`, and the offline DINOv2 gate then refuses
  to run.

## Result files

[`RESULTS.md`](RESULTS.md) lists every result file cited in Supplementary Material S2.4 with the SHA-256 of the
original run output. The files under `ckpt/` are stored byte-for-byte (`.gitattributes` disables line-ending
conversion there). A few files had the device user name or the rented cloud machine's host name and directory
replaced by placeholders; no number was changed, and the original hashes are listed.

## Licenses

- **This repository** is released under the Apache License 2.0 (`LICENSE`). It builds on AA-CLIP, which is
  Apache-2.0; see `NOTICE` for attributions and changes.
- **Vendored MVREC source** (`MVREC-main/`) keeps its own Apache-2.0 license; a proprietary font file of the
  upstream repository is not redistributed (see `NOTICE`).
- **Code not included because of its license:**
  - CLIP-LoRA (no license file upstream): our runners import an unmodified checkout that you download yourself.
  - The official GDA code (AGPL-3.0): `experiments/p44/gda.py` implements GDA from the paper's equations and
    includes none of that code.
- **Datasets and weights** are not redistributed. Their own licenses apply (several datasets are non-commercial).
- **Example images inside the figures** and in `paper_figures/mst/**/assets*` come from MVTec AD
  (CC BY-NC-SA 4.0) and VisA (CC BY 4.0) and remain under those licenses.
