"""P47 helpers shared by U3 and U4 (docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md).

Pure functions except ``single_shot`` (git guard + write-once staging) and ``PerturbedLoader`` (process pool).
Nothing here imports the model code, so the self-tests run on a CPU-only machine without the checkpoints.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import math
import random
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUARDED = ("experiments/p6", "experiments/p10", "experiments/p11", "experiments/p12", "experiments/p13",
           "experiments/p14", "experiments/p15", "experiments/p47", "experiments/revision",
           "docs/P47_UNCERTAINTY_EVALUATION_PROTOCOL.md")


class P47Error(RuntimeError):
    pass


# --------------------------------------------------------------------------- run guard

@contextlib.contextmanager
def single_shot(out: Path, guarded=GUARDED):
    """Refuse to run with uncommitted code or an existing output; write into <out>.staging and rename at the end.

    The <out>.lock directory is left in place as the single-shot marker (as P14/P15): an interrupted run is reported,
    never silently repeated."""
    status = subprocess.run(["git", "status", "--porcelain", "--", *guarded],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise P47Error(f"uncommitted changes in guarded paths:\n{status}")
    if out.exists():
        raise P47Error(f"{out} exists; this P47 stage is write-once")
    lock = out.with_name(out.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock.mkdir()
    except FileExistsError:
        raise P47Error(f"{lock} exists: another run is active or was interrupted (report it; do not delete)")
    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    yield staging
    staging.rename(out)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def np_rng(key: str) -> np.random.Generator:
    """numpy generator seeded from a string key through random.Random (reproducible across platforms)."""
    return np.random.default_rng(random.Random(key).getrandbits(63))


# --------------------------------------------------------------------------- U4 perturbations

# Influence quantities (protocol table U4). The unit x is the one used for sensitivity coefficients.
LEVELS = {"P1": (0.8, 0.9, 1.1, 1.2),      # illumination gain (x = (g - 1) * 100, %)
          "P2": (0.8, 1.25),               # gamma (x = ln gamma)
          "P3": (2.0, 5.0, 10.0),          # Gaussian noise sigma, DN
          "P4": (0.5, 1.0, 2.0),           # Gaussian blur sigma, px (native resolution)
          "P5": (1.0, 2.5, 5.0),           # translation, % of width, random direction
          "P6": (2.0, 5.0),                # rotation, degrees, random sign
          "P7": (90, 70, 50)}              # JPEG quality (x = 100 - q)
NAMES = {"P1": "illumination gain", "P2": "gamma", "P3": "sensor noise (DN)", "P4": "defocus blur (px)",
         "P5": "misalignment (% of width)", "P6": "rotation (deg)", "P7": "JPEG quality"}
UNITS = {"P1": "per 1 % gain change", "P2": "per unit ln(gamma)", "P3": "per DN", "P4": "per px",
         "P5": "per 1 % of width", "P6": "per degree", "P7": "per quality step (100 - q)"}
R_JITTER = 5
P_CONDITIONS = tuple(f"{c}={v:g}" for c, levels in LEVELS.items() for v in levels)
J_CONDITIONS = tuple(f"J{r}" for r in range(1, R_JITTER + 1))
JPEG_SUBSAMPLING = 2   # 4:2:0, the usual camera setting; fixed so the result does not depend on the Pillow default


def level_x(cond: str) -> float:
    code, v = cond.split("=")
    v = float(v)
    return {"P1": (v - 1) * 100, "P2": math.log(v), "P7": 100 - v}.get(code, v)


def _rotate(arr, deg):
    return ndimage.rotate(arr, deg, axes=(1, 0), reshape=False, order=1, mode="reflect")


def _shift(arr, dy, dx):
    return ndimage.shift(arr, (dy, dx, 0), order=1, mode="reflect")


def _jpeg(arr_u8, quality):
    buf = io.BytesIO()
    Image.fromarray(arr_u8).save(buf, format="JPEG", quality=int(quality), subsampling=JPEG_SUBSAMPLING,
                                 optimize=False)
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB"))


def perturb(img: Image.Image, cond: str, sha: str) -> Image.Image:
    """Apply one influence quantity to the decoded RGB image at native resolution.

    Deterministic per image and condition: random.Random(f"p47pert|{sha}|{cond}") (protocol U4)."""
    img = img.convert("RGB")
    if cond == "none":
        return img
    rng = random.Random(f"p47pert|{sha}|{cond}")
    arr = np.asarray(img).astype(np.float64)
    h, w = arr.shape[:2]
    if cond.startswith("J"):
        gain = rng.uniform(0.97, 1.03)
        sigma = rng.uniform(0.0, 2.0)
        dx, dy = (rng.uniform(-0.5, 0.5) / 100 * w for _ in range(2))
        deg = rng.uniform(-0.5, 0.5)
        noise_seed = rng.getrandbits(63)
        arr = _shift(_rotate(arr, deg), dy, dx) * gain
        arr = arr + np.random.default_rng(noise_seed).normal(0.0, sigma, arr.shape)
        return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))
    code, v = cond.split("=")
    v = float(v)
    if code == "P1":
        arr = arr * v
    elif code == "P2":
        arr = 255.0 * (arr / 255.0) ** v
    elif code == "P3":
        arr = arr + np.random.default_rng(rng.getrandbits(63)).normal(0.0, v, arr.shape)
    elif code == "P4":
        arr = ndimage.gaussian_filter(arr, sigma=(v, v, 0), mode="reflect", truncate=4.0)
    elif code == "P5":
        theta = rng.uniform(0.0, 2 * math.pi)
        d = v / 100 * w
        arr = _shift(arr, d * math.sin(theta), d * math.cos(theta))
    elif code == "P6":
        arr = _rotate(arr, v * rng.choice((-1, 1)))
    elif code == "P7":
        return Image.fromarray(_jpeg(np.clip(np.rint(arr), 0, 255).astype(np.uint8), v))
    else:
        raise P47Error(f"unknown condition {cond}")
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))


def _init_worker():
    import torch
    torch.set_num_threads(1)


def _load_view(job):
    """(path, sha, cond) -> the locked 336 px evaluation tensor of the perturbed image, as a numpy array."""
    from experiments.p6 import p6b_stage2_models as models
    path, sha, cond = job
    with Image.open(path) as handle:
        img = handle.convert("RGB").copy()
    return models.deterministic_eval_tensor(perturb(img, cond, sha)).numpy()


class PerturbedLoader:
    """Decoding + perturbation + the evaluation view in a process pool; batches come back in input order."""

    def __init__(self, workers: int):
        self.pool = ProcessPoolExecutor(workers, initializer=_init_worker) if workers > 0 else None

    def views(self, jobs):
        if self.pool is None:
            return map(_load_view, jobs)
        return self.pool.map(_load_view, jobs, chunksize=4)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


# --------------------------------------------------------------------------- metrics

def balanced_accuracy(true_idx: np.ndarray, pred_idx: np.ndarray, n_classes: int) -> float:
    """Mean per-class recall over class_order; a class without queries counts 0 (as classification_metrics)."""
    rec = []
    for c in range(n_classes):
        m = true_idx == c
        rec.append(float((pred_idx[m] == c).mean()) if m.any() else 0.0)
    return float(np.mean(rec))


def stratified_bootstrap_ba(true_idx: np.ndarray, preds: np.ndarray, n_classes: int, reps: int, key: str) -> np.ndarray:
    """Query bootstrap stratified by true class. preds [cells, n] -> BA [reps, cells] (same resamples for all cells)."""
    rng = np_rng(key)
    correct = preds == true_idx[None, :]
    out = np.zeros((reps, preds.shape[0]))
    for c in range(n_classes):
        idx = np.flatnonzero(true_idx == c)
        if idx.size == 0:
            continue
        draw = idx[rng.integers(0, idx.size, (reps, idx.size))]          # [reps, n_c]
        out += correct[:, draw].mean(axis=2).T                             # [reps, cells]
    return out / n_classes


def fleiss_kappa(ratings: np.ndarray, n_categories: int) -> float:
    """ratings [subjects, raters] of category indices. Fleiss (1971)."""
    n_sub, n_rat = ratings.shape
    counts = np.stack([(ratings == j).sum(axis=1) for j in range(n_categories)], axis=1).astype(float)
    p_j = counts.sum(axis=0) / (n_sub * n_rat)
    p_i = ((counts ** 2).sum(axis=1) - n_rat) / (n_rat * (n_rat - 1))
    p_bar, p_e = p_i.mean(), (p_j ** 2).sum()
    if p_e >= 1 - 1e-15:
        return 1.0 if p_bar >= 1 - 1e-15 else float("nan")
    return float((p_bar - p_e) / (1 - p_e))


def attribute_agreement(ratings: np.ndarray, truth: np.ndarray, reference: np.ndarray, n_categories: int) -> dict:
    """AIAG MSA-4 attribute agreement for one system with R repeated trials.

    within: share of parts whose R trials all agree; vs_standard: all R agree and equal the reference label;
    vs_unperturbed: all R agree with the decision on the unperturbed image."""
    agree = (ratings == ratings[:, :1]).all(axis=1)
    return {"parts": int(ratings.shape[0]), "within_repeatability": float(agree.mean()),
            "fleiss_kappa": fleiss_kappa(ratings, n_categories),
            "agreement_vs_standard": float((agree & (ratings[:, 0] == truth)).mean()),
            "agreement_vs_unperturbed": float((ratings == reference[:, None]).all(axis=1).mean())}


def auroc(normal, defect) -> float:
    """P(defect > normal) + 0.5 P(tie), via ranks (same value as experiments.p15.screening.auroc)."""
    from scipy.stats import rankdata
    normal, defect = np.asarray(normal, float), np.asarray(defect, float)
    r = rankdata(np.concatenate([normal, defect]))
    return float((r[len(normal):].sum() - len(defect) * (len(defect) + 1) / 2) / (len(normal) * len(defect)))


def conformal_threshold(cal, alpha: float) -> float:
    """As experiments.p15.screening.conformal_threshold."""
    s = sorted(cal)
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float("inf") if k > len(s) else s[k - 1]


# --------------------------------------------------------------------------- U3 statistics

def two_way_random(y: np.ndarray) -> dict:
    """Two-way random-effects ANOVA without replication (method of moments, negatives set to 0).

    y [a, b]: rows = support draws S, columns = bank draws B. With one deterministic value per cell the residual is
    the S x B interaction, so sigma_SB^2 = MS_SB. Satterthwaite degrees of freedom for the component estimates."""
    a, b = y.shape
    g = y.mean()
    ri, cj = y.mean(axis=1), y.mean(axis=0)
    ms_s = b * ((ri - g) ** 2).sum() / (a - 1)
    ms_b = a * ((cj - g) ** 2).sum() / (b - 1)
    resid = y - ri[:, None] - cj[None, :] + g
    df_sb = (a - 1) * (b - 1)
    ms_sb = (resid ** 2).sum() / df_sb

    def comp(ms, df, n):
        v = (ms - ms_sb) / n
        if v <= 0:
            return 0.0, math.inf
        return v, (ms - ms_sb) ** 2 / (ms ** 2 / df + ms_sb ** 2 / df_sb)
    s2_s, nu_s = comp(ms_s, a - 1, b)
    s2_b, nu_b = comp(ms_b, b - 1, a)
    return {"mean": float(g), "ms_S": float(ms_s), "ms_B": float(ms_b), "ms_SB": float(ms_sb),
            "var_S": s2_s, "var_B": s2_b, "var_SB": float(ms_sb), "nu_S": nu_s, "nu_B": nu_b, "nu_SB": float(df_sb)}


def welch_satterthwaite(parts) -> float:
    """parts: iterable of (variance contribution u_i^2, nu_i). Returns nu_eff (inf if every nu_i is inf)."""
    parts = [(v, n) for v, n in parts if v > 0]
    den = sum(v ** 2 / n for v, n in parts if math.isfinite(n))
    num = sum(v for v, _ in parts) ** 2
    return math.inf if den == 0 else num / den


def t975(nu: float) -> float:
    from scipy import stats
    return 1.959963984540054 if not math.isfinite(nu) else float(stats.t.ppf(0.975, nu))
