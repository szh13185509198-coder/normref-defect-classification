# Figure generators

The figures of the paper are listed, with their editable sources and plotted data, in
[`../final/README.md`](../final/README.md). This folder holds the scripts that produce them.

## Data charts

```bash
python paper_figures/mst/make_mst_figures.py              # plotted numbers -> out/*.csv (and greyscale PDF/PNG)
python paper_figures/mst/make_mst_figures.py fig3 fig7    # selected figures only
python paper_figures/mst/schematics/make_color_charts.py  # editable draw.io charts -> drawio/charts/
```

- The scripts read only the result-file copies in `paper_figures/data/ckpt` (hashes of the originals in
  `paper_figures/data/README.md`), the device results in `ckpt/p47/p47_jetson/`, and the plotted data of earlier
  exports in `paper_figures/legacy_main` and `paper_figures/legacy_supplementary`.
- `out/NOTES.txt` records the source of every plotted number and how each interval was computed.
- Every bar, point, error bar and axis of a draw.io chart is a separate editable object.
- The greyscale matplotlib versions in `out/` are kept as a numerical cross-check.

## Schematics

```bash
python paper_figures/mst/schematics/make_schematics.py     # Figures 2 and 3 (base versions)
python paper_figures/mst/schematics/make_figs_v2.py fig1   # Figure 1 (edit of schem_measurement_chain_base.drawio)
python paper_figures/mst/schematics/make_figs_v2.py fig6   # Figure 6
python paper_figures/mst/schematics/make_figs_v3.py        # Figures 8 and 10
```

- The `.drawio` files open in draw.io / diagrams.net. PNG and PDF are exported with the draw.io desktop app
  (`File > Export as > PDF`, "crop"); `render.js` renders them headless with the diagrams.net viewer
  (`DRAWIO_VIEWER` points to `viewer-static.min.js`).
- Colour roles: blue = quantities from labelled defect images; aqua = the normal reference (conforming parts);
  orange = per-part inspection; violet = decisions and reported results; grey = uncertainty tags. Every box also
  carries text, so colour is never the only cue, and the palette was checked for colour-vision deficiency.
- Small illustrative glyphs (`schematics/mini.py`: score histograms, prototype scatter, chip icon) carry no axis
  values; all other insets are real data or images.

## Part images of the schematics

`schematics/make_assets.py` computes the images and maps used in Figures 1-3 from the official MVTec AD images
(CC BY-NC-SA 4.0, credited in the captions):

- query `bottle/test/broken_small/014.png`, a correctly classified case of the published qualitative figure
  (K = 5, seed 111); 87% of the weight w falls inside its ground-truth mask;
- nominal bank: the first 20 images of `bottle/train/good`;
- maps recomputed with the definitions of the paper (CLIP ViT-L/14@336, layer-18 patch tokens, leave-one-out
  0.99 quantile tau, w = max(a - tau, 0)) using the Hugging Face weights and a plain 336 x 336 resize. They
  illustrate the mechanism and enter no reported number. `ASSETS.json` records paths, SHA-256, tau and the
  number of patches above tau.

## Size and text checks

Figures are drawn at their printed size (15 cm or 8.5 cm wide; 567 px = 15 cm) with text of 8.25-9 pt.

```bash
TEXT_METRICS_DIR=/tmp/m DRAWIO_VIEWER=<viewer-static.min.js> python paper_figures/mst/schematics/make_color_charts.py
TEXT_METRICS_DIR=/tmp/m DRAWIO_VIEWER=<viewer-static.min.js> python paper_figures/mst/schematics/make_schematics.py
python paper_figures/mst/schematics/check_figs.py /tmp/m   # draw.io figures
python paper_figures/mst/check_pdf.py                      # matplotlib figures
```

The checks fail (non-zero exit) if a page is wider than 15 cm, any text is smaller than 8 pt at print size, a
labelled draw.io cell has fontSize below 10.67 px, texts overlap or leave the page, or a PDF font is not embedded.
Text lying over data markers is not detected and has to be checked on the rendered figure.
