<!-- translation-header -->
> English translation of [`README_U6D.md`](README_U6D.md), which is kept unchanged (SHA-256 `02248b6fda84294b852633232e5e75da3cf469e8e9aaef32a3f053038c4cd399`). The translation keeps the structure and numbers of the original.

# P47 U6-D device outputs

- Run: 2026-10-03 19:44–20:10, Jetson Orin Nano Super 8 GB, MAXN_SUPER, GPU clock locked at 1.02 GHz (read back from
  sysfs before and after the run), headless, drm_holder.py running, no other program on the device (the earlier tiling
  screen had finished and its outputs had been retrieved).
- Code: experiments/p47/{jetson_u6.py,run_jetson_u6.sh,analyze_u6d.py}, with the hashes stated in the hand-over
  document, committed to the local research repository (951c999) before any output; all local self-tests passed.
- Retrieval: tar 210672b5… identical on both ends; each of the 24 device files below matches its sha256 on the device.
  p47_u6d_summary.{json,md} were generated locally by analyze_u6d.py.
- Changes: so that the device user name does not appear in the repository, the path `/home/…/` containing it was
  replaced by `/home/<USER>/` in the files marked below (it occurs only in engine paths and in the source paths of
  warnings; no number was changed), so the hashes of these files differ from the originals listed below.
- Anomalies: every process shows the TensorRT warning "engine plan file across different models of devices"; the same
  engine already showed it in the P43 logs of 2026-10-02, so it is not new. No errors, no throttling.
- The device rebooted unexpectedly at 15:58 on 2026-10-03 (unrelated to this run, during another experiment); the
  premise "no reboot since 10-02" of section 3.2 of the hand-over document therefore does not hold; after the reboot
  the locked clocks and the headless mode were restored.

| File | SHA-256 of the original | User name replaced |
|---|---|---|
| `ckpt/p43/p47_jetson_e1/p43_energy_fused_e2e.json` | `827c74b6f5f3a9e11318fd9b796dcf1178c33c0442c02a00cc44bf9cf2853d57` | — |
| `ckpt/p43/p47_jetson_e1/p43_idle.json` | `64807d6354eaa476d403f747656050349582286e6d29d2624bf80a0eea5dc850` | — |
| `ckpt/p43/p47_jetson_e2/p43_energy_fused_e2e.json` | `cca2f6c5645fff86443075106a3a265e6b393ddbe83a6eee3c52b9247bc8afb8` | — |
| `ckpt/p43/p47_jetson_e2/p43_idle.json` | `4bcc6f536375d0bf791e10233fa428bf5cafcd7cae259c87aa93263ffff719c1` | — |
| `ckpt/p43/p47_jetson_e3/p43_energy_fused_e2e.json` | `7f68943316cd5f8e1003e25759236f9404b76597b8f7d5c3c628a3e4d8379fcb` | — |
| `ckpt/p43/p47_jetson_e3/p43_idle.json` | `046fdd97761d9a5ba9a7770ae0d683e1b74e66ee6d33f5b86a54b730ce3ad0b6` | — |
| `ckpt/p43/p47_jetson_e4/p43_energy_fused_e2e.json` | `09c3417c209b41f5dc06f01e5816df2b4b229245f21a550d6e4ec112dd19e7bd` | — |
| `ckpt/p43/p47_jetson_e4/p43_idle.json` | `8c7ed45e8e7d5371afdec88f63c56d4e3636540dfad77655201abc8ed6e9d5d6` | — |
| `ckpt/p43/p47_jetson_e5/p43_energy_fused_e2e.json` | `9017e455ca83d812467d61c83398af72306a2abe74b00d1cf1fe33c809fd40d7` | — |
| `ckpt/p43/p47_jetson_e5/p43_idle.json` | `e1339f31d97111d36d0b5976034cea6a20e28ba82100f038db12f6767e6f81dd` | — |
| `ckpt/p43/p47_jetson/p43_repeat_s01.json` | `37085e185aa0c7886f6d0c52c2b1a1cf9acaf75eb99b1aa2fbf9b2e925984c4d` | replaced |
| `ckpt/p43/p47_jetson/p43_repeat_s02.json` | `3a6be029ad9cff152b8517a9abee174445faf102fd24d32462b0dde558aebb51` | replaced |
| `ckpt/p43/p47_jetson/p43_repeat_s03.json` | `13fa08cf6a001694f6e207be3eb002262d1c3b14cdec9a3cef75d281ca57616d` | replaced |
| `ckpt/p43/p47_jetson/p43_repeat_s04.json` | `611a9502af8a97d88b30d20edf125d51bc6a8d73dd25aa04848eba8cf5c24393` | replaced |
| `ckpt/p43/p47_jetson/p43_repeat_s05.json` | `d3618996598b88520879fcbdb164d2d333f138569e70be2e833002993872c0a5` | replaced |
| `ckpt/p43/p47_jetson/p43_repeat_s06.json` | `521a9139cb9e9bd831ae365b2dd4bd5712bac1e4f273ca34ef08df66b9a0ed45` | replaced |
| `ckpt/p43/p47_jetson/p43_repeat_s07.json` | `ef9623741d0a59e11994cb93db8708d5656bee9f696ebca4996ce5bc1ebbc0c5` | replaced |
| `ckpt/p43/p47_jetson/p43_repeat_s08.json` | `65a67d984634aaa448e7f65d650da00e228e508f63cafb8aa877935fa82bec39` | replaced |
| `ckpt/p43/p47_jetson/p43_repeat_s09.json` | `830b2d438cb394386cace9cabdf5e7674b01e3b9dcceedabf8e5d40567aaadfb` | replaced |
| `ckpt/p43/p47_jetson/p43_repeat_s10.json` | `250eb2a7988f654e22c2a11e80cee916f5a6f4e364ef179b79259fcdd444676a` | replaced |
| `ckpt/p47/p47_jetson/DONE` | `92391ab9bc05953a10655e23fe680163df3077558b5b60e0dcd2b5f424f09d7b` | — |
| `ckpt/p47/p47_jetson/p47_u6_scaling.json` | `18f48594e598e6993b9cc2decfb052eb676f4442074bd312b83242af9e292fbd` | replaced |
| `ckpt/p47/p47_jetson/run_log.txt` | `b9a9fa117ddaf22b5d0d2d73bd10af71bfa006598d76f530f02e8ab83bf105a2` | replaced |
| `ckpt/p47/p47_jetson/tegrastats.log` | `b2ad93c59f36159d780d9400553998efd2a74c97681ccbdf4ba51681c767572d` | — |
| `ckpt/p47/p47_jetson/p47_u6d_summary.json` (generated locally) | `7acfc7450d589d74ceeb261857b5f86502e95b7a078a2916979d332a88f89a74` | — |
| `ckpt/p47/p47_jetson/p47_u6d_summary.md` (generated locally) | `ada5b69dbe65d3474cc47074dd8201fce421f6bb855d467de2449ee4eec6624a` | — |
