# P49 — The trainable support-side adaptation on the Jetson: training time and inference latency (report only)

Status: **LOCKED BEFORE ANY P49 OUTPUT** (2026-10-03, user decision after an external review). No method, constant or
earlier result changes. Disclosed in S2.

## 1. Why
Table 12 lists the restricted (trainable) support-side adaptation as "not finished: killed for memory after the 80
training steps". The retrieved run records (`ckpt/p26/jetson_orin_nano_r3/restricted_unit_{1,2}`) show that both
passes finished training: 80 optimizer steps, checkpoint and FAISS index saved, the support stage marked complete.
The process was killed (rc = −9) afterwards, when the P26 harness called `cmd_latency` **in the same process**, which
builds a second CLIP-L/14 + DINOv2-S pipeline while the trained one is still in memory. The kill is an artefact of the
harness, not of the method.

## 2. What is reported
- **Training time** (no new run): the P26 harness rule applied to the saved logs, first log line → "Checkpoint saved":
  pass 1 571.8 s, pass 2 571.0 s; 80 steps; CUDA peak allocated 2.34 GiB during training.
- **Inference latency** (new, on the device): exactly the call the harness would have made,
  `python -m experiments.revision.c3_resource_profile latency --pipeline ours --device cuda` (FP32, batch 1,
  warm-up 50, 200 timed calls; model forward and end-to-end from the decoded image; random prototypes, timing is
  value-independent), each in its **own fresh process**, twice (pass 1, pass 2). Conditions as Table 12: MAXN_SUPER,
  clocks locked, headless, nothing else running.

## 3. Use
Table 12's row is filled with the two-pass means, with a note that the original run's process was killed by the
harness after training finished. Descriptive; no selection.
