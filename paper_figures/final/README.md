# Figures of the paper

Every image linked from the manuscript and supplementary material, copied byte-for-byte (SHA-256 checked
against the manuscript files) under English names, together with the PDF of the same export where one exists.

Paths in the "Source" and "Script" columns are relative to `paper_figures/mst/` unless they start with
`paper_figures/` or `experiments/`. The figure scripts read only the result-file copies in `paper_figures/data/ckpt`
(hashes of the originals in `paper_figures/data/README.md`) and the plotted data of earlier exports; rerunning them
redraws the figures. Draw.io export of PNG / PDF needs the draw.io desktop app.

## Main text

| Figure | File | Source image in the repository | Editable source | Script | Plotted data |
|---|---|---|---|---|---|
| 1 | `main/Figure01_measurement_chain.png` + `.pdf` | `paper_figures/mst/drawio/schematics/schem_measurement_chain.png` | `paper_figures/mst/drawio/schematics/schem_measurement_chain.drawio` | schematics/make_figs_v2.py fig1 (edit of the hand-drawn schem_measurement_chain_base.drawio) | — |
| 2 | `main/Figure02_method_detail.png` + `.pdf` | `paper_figures/mst/drawio/schematics/schem_method_detail.png` | `paper_figures/mst/drawio/schematics/schem_method_detail.drawio` | schematics/make_schematics.py | — |
| 3 | `main/Figure03_uncertainty_sources.png` + `.pdf` | `paper_figures/mst/drawio/schematics/schem_uncertainty_sources.png` | `paper_figures/mst/drawio/schematics/schem_uncertainty_sources.drawio` | schematics/make_schematics.py, then edited in draw.io | — |
| 4 | `main/Figure04_accuracy_loss.png` + `.pdf` | `paper_figures/mst/drawio/charts/fig2_headroom.png` | `paper_figures/mst/drawio/charts/fig2_headroom.drawio` | make_mst_figures.py → schematics/make_color_charts.py | out/fig2_headroom.csv |
| 5 | `main/Figure05_gains_vs_shots.png` + `.pdf` | `paper_figures/mst/drawio/charts/fig3_gains.png` | `paper_figures/mst/drawio/charts/fig3_gains.drawio` | make_mst_figures.py → schematics/make_color_charts.py | out/fig3_gains.csv |
| 6 | `main/Figure06_patch_evidence.png` + `.pdf` | `paper_figures/mst/drawio/schematics/fig5_patch_selection.png` | `paper_figures/mst/drawio/schematics/fig5_patch_selection.drawio` | schematics/make_figs_v2.py fig6 | drawio/schematics/fig5_patch_selection.csv; ../data/ckpt/fig6_evidence |
| 7 | `main/Figure07_frr_calibration.png` + `.pdf` | `paper_figures/mst/drawio/charts/fig6_frr_theory.png` | `paper_figures/mst/drawio/charts/fig6_frr_theory.drawio` | make_mst_figures.py → schematics/make_color_charts.py | out/fig6_frr_theory.csv |
| 8 | `main/Figure08_operating_points.png` + `.pdf` | `paper_figures/mst/drawio/charts/fig7_operating_points.png` | `paper_figures/mst/drawio/charts/fig7_operating_points.drawio` | schematics/make_figs_v3.py fig8 | out/fig7_operating_points.csv |
| 9 | `main/Figure09_influence_quantities.png` + `.pdf` | `paper_figures/mst/drawio/charts/fig8_influence.png` | `paper_figures/mst/drawio/charts/fig8_influence.drawio` | schematics/make_color_charts.py | out/fig8_influence.csv |
| 10 | `main/Figure10_accuracy_and_cost.png` + `.pdf` | `paper_figures/mst/drawio/charts/fig9_cost.png` | `paper_figures/mst/drawio/charts/fig9_cost.drawio` | schematics/make_figs_v3.py fig10 | out/fig9_cost.csv; ckpt/p47/p47_jetson/p47_u6d_summary.json |

## Supplementary material

| Figure | File | Source image in the repository | Editable source | Script | Plotted data |
|---|---|---|---|---|---|
| S1.1 | `supplementary/FigureS1-1_mvtecfs_coverage.png` + `.pdf` | `paper_figures/legacy_supplementary/figS1-1_mvtecfs_coverage/figS1-1_mvtecfs_coverage.png` | `paper_figures/legacy_supplementary/figS1-1_mvtecfs_coverage/figS1-1_mvtecfs_coverage.svg` | hand-drawn (draw.io) | — |
| S6.1 | `supplementary/FigureS6-1_cable_combined_failures.png` + `.pdf` | `paper_figures/legacy_supplementary/figS6-1_cable_failures/figS6-1_cable_failures.png` | — | experiments/p16/make_figures_v4.py | CSV in the source folder |
| S10.1 | `supplementary/FigureS10-1_false_rejects_without_screening.png` + `.pdf` | `paper_figures/legacy_supplementary/figS10-1_false_rejects/figS10-1_false_rejects.png` | — | experiments/p16/make_figures_v4.py | CSV in the source folder |
| S11.1 | `supplementary/FigureS11-1_robustness.png` + `.pdf` | `paper_figures/legacy_supplementary/figS11-1_robustness/figS11-1_robustness.png` | — | experiments/p20/make_figures_v5.py | CSV in the source folder |
| S11.2 | `supplementary/FigureS11-2_threshold_source.png` + `.pdf` | `paper_figures/mst/drawio/charts/figS_threshold_source.png` | `paper_figures/mst/drawio/charts/figS_threshold_source.drawio` | make_mst_figures.py → schematics/make_color_charts.py | out/figS_threshold_source.csv |
| S12.1 | `supplementary/FigureS12-1_evidence_purity.png` + `.pdf` | `paper_figures/mst/drawio/charts/figS_evidence_purity.png` | `paper_figures/mst/drawio/charts/figS_evidence_purity.drawio` | make_mst_figures.py → schematics/make_color_charts.py | out/figS_evidence_purity.csv |
| S12.2 | `supplementary/FigureS12-2_backbone_swap.png` + `.pdf` | `paper_figures/legacy_supplementary/figS12-2_backbone_swap/figS12-2_backbone_swap.png` | — | experiments/p20/make_figures_v5.py | CSV in the source folder |
| S14.1 | `supplementary/FigureS14-1_per_product_gains.png` + `.pdf` | `paper_figures/legacy_supplementary/figS14-1_per_product/figS14-1_per_product.png` | — | experiments/p16/make_figures_v4.py | CSV in the source folder |
| S16-1 | `supplementary/FigureS16-1_u3_uncertainty_budget.png` + `.pdf` | `paper_figures/mst/drawio/charts/figS_u3_budget.png` | `paper_figures/mst/drawio/charts/figS_u3_budget.drawio` | schematics/make_color_charts.py | out/figS_u3_budget.csv |

Some figures show example images from MVTec AD and VisA, used as in the paper under the datasets' licenses.
