"""P47 U4 diagnostic: is the U4 geometric sensitivity caused by the reflect padding? (run once in the cloud, 2026-10-04)

Diagnostic only: Hugging Face openai/clip-vit-large-patch14-336, layer-18 patch tokens (the paper's definitions, not
the frozen feature caches); MVTec AD train/good images only; not used for any reported number. It motivated U4-A.

For good parts of an MVTec AD product: bank = 20 train/good, calibration = 30 further train/good, test = 20 further
train/good. Screening score = top1pct of a(t) (P15 statistic), threshold = conformal at alpha = 5 % on the unperturbed
calibration scores. Two ways to move the part:
  A  as U4: translate / rotate the whole image with reflect padding (experiments/p47/common.perturb), full-image view;
  B  padding-free: the view is a centred window (88 % of each side) of the original; a translation moves the window
     by the same distance in the same direction (real image content enters, no padding); a rotation rotates the
     original and takes the centred window (no padded pixels reach it for 5 degrees). Bank and calibration use the
     centred window.
Reports per pipeline: FRR on the test goods, mean score shift, and the share of exceeding patches (a > tau) in the
outer two patch rings.
"""
import hashlib, json, math, random, sys, time
from pathlib import Path

import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.p47 import common as cm
from scipy import ndimage

torch.set_num_threads(4)
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
FR = 0.88

from transformers import CLIPVisionModel
model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").eval()
model.vision_model.encoder.layers = model.vision_model.encoder.layers[:18]


def tens(img):
    im = img.convert("RGB").resize((336, 336), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im).astype(np.float32) / 255).permute(2, 0, 1)
    return (x - MEAN) / STD


@torch.no_grad()
def tokens(imgs):
    out = []
    for i in range(0, len(imgs), 8):
        b = torch.stack([tens(im) for im in imgs[i:i + 8]])
        h = model(pixel_values=b, output_hidden_states=True).hidden_states[18][:, 1:]
        out.append(F.normalize(h, dim=-1))
    return torch.cat(out)


def window(img, dx=0.0, dy=0.0):
    w, h = img.size
    ww, hh = FR * w, FR * h
    x0, y0 = (w - ww) / 2 + dx, (h - hh) / 2 + dy
    return img.crop((round(x0), round(y0), round(x0 + ww), round(y0 + hh)))


def b_shift(img, sha, cond):
    v = float(cond.split('=')[1])
    rng = random.Random(f"p47pert|{sha}|{cond}")
    th = rng.uniform(0.0, 2 * math.pi)            # the same direction as U4
    d = v / 100 * img.size[0]
    return window(img, d * math.cos(th), d * math.sin(th))


def b_rot(img, sha, cond):
    v = float(cond.split('=')[1])
    rng = random.Random(f"p47pert|{sha}|{cond}")
    deg = v * rng.choice((-1, 1))
    arr = ndimage.rotate(np.asarray(img.convert('RGB')).astype(float), deg, axes=(1, 0), reshape=False, order=1,
                         mode='reflect')
    return window(Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8)))


def anomalies(q, bank):
    flat = bank.reshape(-1, bank.shape[-1])
    return torch.cat([1 - (c @ flat.T).max(dim=-1).values for c in q.split(4)])


def loo_tau(bank):
    vals = []
    for i in range(bank.shape[0]):
        others = torch.cat([bank[:i], bank[i + 1:]]).reshape(-1, bank.shape[-1])
        vals.append(1 - (bank[i] @ others.T).max(dim=1).values)
    return float(torch.quantile(torch.cat(vals), 0.99))


def top1pct(a):
    return a.topk(math.ceil(0.01 * a.shape[1]), dim=1).values.mean(dim=1)


ring = torch.zeros(24, 24, dtype=torch.bool)
ring[:2, :] = ring[-2:, :] = ring[:, :2] = ring[:, -2:] = True
ring = ring.flatten()


def main(product, root=Path(sys.argv[1]) if len(sys.argv) > 1 else Path('mvtec_ad')):
    t0 = time.time()
    good = sorted((root / product / 'train' / 'good').glob('*.png'))
    bank_p, cal_p, test_p = good[:20], good[20:50], good[50:70]
    load = lambda p: Image.open(p).convert('RGB')
    sha = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in test_p}
    res = {}
    for pipe in ('A', 'B'):
        view = (lambda im: im) if pipe == 'A' else window
        bank = tokens([view(load(p)) for p in bank_p])
        tau = loo_tau(bank)
        cal = top1pct(anomalies(tokens([view(load(p)) for p in cal_p]), bank)).tolist()
        thr = cm.conformal_threshold(cal, 0.05)
        for cond in ('none', 'P5=5', 'P6=5'):
            if cond == 'none':
                imgs = [view(load(p)) for p in test_p]
            elif pipe == 'A':
                imgs = [cm.perturb(load(p), cond, sha[p]) for p in test_p]
            else:
                f = b_shift if cond.startswith('P5') else b_rot
                imgs = [f(load(p), sha[p], cond) for p in test_p]
            a = anomalies(tokens(imgs), bank)
            s = top1pct(a)
            ex = a > tau
            res[f'{pipe}|{cond}'] = {'frr': float((s > thr).float().mean()), 'mean_score': float(s.mean()),
                                    'exceed_per_image': float(ex.sum(1).float().mean()),
                                    'exceed_share_in_outer_ring': float(ex[:, ring].sum() / max(1, ex.sum()))}
        print(product, pipe, f'{time.time() - t0:.0f}s', flush=True)
    return res


if __name__ == '__main__':
    # usage: python -m experiments.p47.diag_geometry_padding <mvtec_root> <product> [<product> ...]
    out = {p: main(p) for p in sys.argv[2:]}
    dst = Path(__file__).resolve().parents[2] / 'ckpt' / 'p47' / 'diag_geometry_padding'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / ('geo_results_' + '_'.join(sys.argv[2:]) + '.json')).write_text(json.dumps(out, indent=1))
    for p, r in out.items():
        print(p)
        for k, v in r.items():
            print(f"  {k:10s} FRR {100 * v['frr']:5.1f}%  mean score {v['mean_score']:.4f}  exceed/img {v['exceed_per_image']:6.1f}  "
                  f"in outer ring {100 * v['exceed_share_in_outer_ring']:5.1f}%")
