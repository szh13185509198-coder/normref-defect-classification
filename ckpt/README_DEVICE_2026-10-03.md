# 2026-10-03 设备实验结果（供审核）

Jetson Orin Nano 8 GB 上当天的全部实验结果文件（只含 JSON / JSONL / 日志，不含特征 `.pt`、`.npy` 与任何图像），路径与研究仓库 `ckpt/` 一致；设备运行日志放在 `ckpt/device_logs/`。各轮的协议、修订、运行经过与数值见研究仓库 `docs/agent/RESEARCH_LEDGER.md` 的同日条目与补充材料 S2、S8.11–S8.13、S10.6。U6-D 的输出与说明见 `ckpt/p47/p47_jetson/README_U6D.md`。

- 编号提醒：本分支的 P47 是不确定度评估（U6-D）；表中 "P47 规模扩展 / 切块 / 修订 A1–A4" 是同日另一轮实验（研究仓库 `docs/P47_SCALING_AND_TILING_PROTOCOL.md`），正式命名时需改号。
- 替换：设备用户名、云容器主机名与云端目录换成了占位符（`<USER>`、`<cloud-container>`、`<cloud-dir>`），数值未改；这些文件的哈希与下表原件不同。
- 设备 15:58 曾意外重启（P47 r3 期间），原因无法确认；重启后由执行 agent 恢复无图形界面与锁频，DRM 保持进程由我们以普通用户重启。

| 内容 | 文件 | 大小（字节） | 原件 SHA256 | 占位符替换 |
|---|---|---|---|---|
| P43 A–D (10-02 夜) 与修订 A2 B4/B5（10-03） | `p43/jetson_orin_nano/p43_cold_run1.json` | 1,339 | `1049c5b3b537b28eca1222e3eb8a58a57c6a7378c2fa1223f92b782857063ee5` | — |
| P43 A–D (10-02 夜) 与修订 A2 B4/B5（10-03） | `p43/jetson_orin_nano/p43_cold_run2.json` | 1,344 | `1ff671a39701e8e50544a62ad81df8a08c9bc5bb66fd8c8891bce4864fb1a005` | — |
| P43 A–D (10-02 夜) 与修订 A2 B4/B5（10-03） | `p43/jetson_orin_nano/p43_cold_run3.json` | 1,338 | `fdd09726df92a58d507847571c88e9ac86cd74242d9de02c656aed1dc2f82137` | — |
| P43 A–D (10-02 夜) 与修订 A2 B4/B5（10-03） | `p43/jetson_orin_nano/p43_decode.json` | 2,105 | `549004b8ee30cd805dff1c4e0af8c7bdd1cd3e0723df6f82cd1431c056886a72` | — |
| P43 A–D (10-02 夜) 与修订 A2 B4/B5（10-03） | `p43/jetson_orin_nano/p43_energy_fused_e2e.json` | 320 | `4178fb52b5867957f04c8add8bb86d1afa60630c2e208cd7d449d5136d7ef262` | — |
| P43 A–D (10-02 夜) 与修订 A2 B4/B5（10-03） | `p43/jetson_orin_nano/p43_energy_fused_engine_only.json` | 328 | `bc3cd82e3a4458ae20efe7ae8102926b671d74dcbe5aca207c27cf3d6acebbd8` | — |
| P43 A–D (10-02 夜) 与修订 A2 B4/B5（10-03） | `p43/jetson_orin_nano/p43_energy_gate_dino_pass.json` | 369 | `b97801d545f0da6d064815ab31d924104cf61ab8a99457309d165915c120f714` | — |
| P43 A–D (10-02 夜) 与修订 A2 B4/B5（10-03） | `p43/jetson_orin_nano/p43_energy_gate_l18_pass.json` | 368 | `0110ac5b0cb04914c1a7c60cf05113a3fd7fa979cc7ec3018b217d020ad57019` | — |
| P43 A–D (10-02 夜) 与修订 A2 B4/B5（10-03） | `p43/jetson_orin_nano/p43_energy_trt_b336_engine.json` | 328 | `f6e9b4eec24c65fd456cb324902779b91a2e7223a9b44d94d66a9e564d1a05ea` | — |
| P43 A–D (10-02 夜) 与修订 A2 B4/B5（10-03） | `p43/jetson_orin_nano/p43_energy_trt_s336_engine.json` | 327 | `d39fa38a03eecd7fee8515908937c952b8a578a6a6f0df6a8767853fc5000eb7` | — |
| P43 A–D (10-02 夜) 与修订 A2 B4/B5（10-03） | `p43/jetson_orin_nano/p43_idle.json` | 635 | `51a1c4dde0179c2738076d6697f24cbccd0df1dda84f76a9009611592e4e29fb` | — |
| P43 A–D (10-02 夜) 与修订 A2 B4/B5（10-03） | `p43/jetson_orin_nano/p43_latency.json` | 2,570 | `3a6e4e809997ddc2c6c1723fa833d9027b4d3bf1685b4b5b91f28fe6c4f932fe` | 1 处 |
| P43 A–D (10-02 夜) 与修订 A2 B4/B5（10-03） | `p43/jetson_orin_nano/p43_repeat_r1.json` | 2,601 | `cdc0a15bbde7a8ee40a2ec551722ee9c85a7bb0b113450091805a83f0da4abb8` | 1 处 |
| P43 A–D (10-02 夜) 与修订 A2 B4/B5（10-03） | `p43/jetson_orin_nano/p43_repeat_r2.json` | 2,595 | `d0d66d6cf4d39114a84098066fe112f70b4085e5a87691ec799d73e5b81b8fea` | 1 处 |
| P43 A–D (10-02 夜) 与修订 A2 B4/B5（10-03） | `p43/jetson_orin_nano/p43_single_scores.json` | 195,121 | `ab426e8692fafc37793f4adfb58697ebf79a4b569f9908f1bd7cfb04f072b08d` | — |
| P43 修订 A2 A3：单一引擎 25 W | `p43/jetson_orin_nano_25w/p43_energy_fused_e2e.json` | 320 | `0ee099a5c87cf6502fde88e42409b179ba009560cce6b29cdc84ec8e494c8627` | — |
| P43 修订 A2 A3：单一引擎 25 W | `p43/jetson_orin_nano_25w/p43_energy_fused_engine_only.json` | 328 | `ca3074c509e62a361a794f73c58f578e92976fa75e04b8f585aea89961d10867` | — |
| P43 修订 A2 A3：单一引擎 25 W | `p43/jetson_orin_nano_25w/p43_energy_gate_dino_pass.json` | 368 | `da3eb3e6db579ce8ce17e3e9578a6979697ad1e087837b87965304cd3dd8feff` | — |
| P43 修订 A2 A3：单一引擎 25 W | `p43/jetson_orin_nano_25w/p43_idle.json` | 636 | `5d160228550bbb3a12894702f80d591ea85ff8da7269870f09cb622f8e6efabf` | — |
| P43 修订 A2 A3：单一引擎 25 W | `p43/jetson_orin_nano_25w/p43_repeat_25w.json` | 1,340 | `5ebe68f2eca278b1afbcb36fc6198b3ed2e1d2ef51bc0a7302127f854692aac8` | 1 处 |
| P43 修订 A2 A3：单一引擎 15 W | `p43/jetson_orin_nano_15w/p43_energy_fused_e2e.json` | 319 | `f02d1448ad11fb192513133c1459cfebb9cbd7b4d544497602c28d1f412bcb84` | — |
| P43 修订 A2 A3：单一引擎 15 W | `p43/jetson_orin_nano_15w/p43_energy_fused_engine_only.json` | 328 | `25e36118744f3cb14d6aece78ddb0535b9b765fd12478e07415439204addf555` | — |
| P43 修订 A2 A3：单一引擎 15 W | `p43/jetson_orin_nano_15w/p43_energy_gate_dino_pass.json` | 370 | `425f0221dc14faab5f831d445e622dc5f061464fd78925b6497683433c7f8f8a` | — |
| P43 修订 A2 A3：单一引擎 15 W | `p43/jetson_orin_nano_15w/p43_idle.json` | 635 | `93740ccb56ba825ba24492cffd4e4dbd0c8e7fa4d1d49f75e1daa606c78f2cad` | — |
| P43 修订 A2 A3：单一引擎 15 W | `p43/jetson_orin_nano_15w/p43_repeat_15w.json` | 1,333 | `3a0f3dbf2cd2af22ddff82fa31bbd1af6766628182fa757072d2f44aece1994d` | 1 处 |
| P44 GDA 设备计时 | `p44/jetson_orin_nano/pass1_b16_224.json` | 1,013 | `02c7080ccf0b9da78419328d7eb78bb2eb249ff61dbba5146cb25814c9fcc4aa` | — |
| P44 GDA 设备计时 | `p44/jetson_orin_nano/pass1_l14_336.json` | 1,011 | `983a2c37f165d996687b9953a3ed6e39b661cf438e313795718c8bab9f73548e` | — |
| P44 GDA 设备计时 | `p44/jetson_orin_nano/pass2_b16_224.json` | 1,015 | `ace5385e87a5ecc2121d8d7339d226f8c465a45280e641b4f044836e7b2b80e2` | — |
| P44 GDA 设备计时 | `p44/jetson_orin_nano/pass2_l14_336.json` | 1,011 | `8ecab0cf380bb24be867b8b78c4f02839498040a4d1fe9eb617fdecb09f1cc0e` | — |
| P45 产线模拟、A1 其余配置、A2 连续 1 小时 | `p45/jetson_orin_nano/p45_file_pipe.json` | 12,963 | `7fdffd5eac4b1eaceaccd9135f910aa046b69c6db4233fc4cf6b1a56f327aa84` | 1 处 |
| P45 产线模拟、A1 其余配置、A2 连续 1 小时 | `p45/jetson_orin_nano/p45_file_seq.json` | 12,956 | `ceffff90490e213edb4331a09896c53e9c11588412a515be6ae8fe42e22571cf` | 1 处 |
| P45 产线模拟、A1 其余配置、A2 连续 1 小时 | `p45/jetson_orin_nano/p45_frame_seq.json` | 12,941 | `0f1289bd53b1a83e9f57eeb06ffcc2c87396f94300675fff2e00b4d15131ec11` | 1 处 |
| P45 产线模拟、A1 其余配置、A2 连续 1 小时 | `p45/jetson_orin_nano/p45_idle.json` | 160 | `1a54c7be2a340cc82944f55d40baad7b21352b0cd8effa197f3e4e510510d058` | — |
| P45 产线模拟、A1 其余配置、A2 连续 1 小时 | `p45/jetson_orin_nano_cfg/p45_clip_l18_file_pipe.json` | 13,457 | `9e43fed3345ac9bdab7682ae15e6edfabfa7cc2f247a7ca30287d3599054ba63` | — |
| P45 产线模拟、A1 其余配置、A2 连续 1 小时 | `p45/jetson_orin_nano_cfg/p45_clip_l18_frame_seq.json` | 13,432 | `278c59121024d34ab47640a37ee8cc53777dc151edf429ce8eaf77248f041c2f` | — |
| P45 产线模拟、A1 其余配置、A2 连续 1 小时 | `p45/jetson_orin_nano_cfg/p45_clipb16_224_file_pipe.json` | 13,430 | `f4766620756f7a8d2470b5d306c5dbcfbb8821811e8aa2b56d5fa8b76d99fced` | — |
| P45 产线模拟、A1 其余配置、A2 连续 1 小时 | `p45/jetson_orin_nano_cfg/p45_clipb16_224_frame_seq.json` | 13,452 | `5a6abaf00927dc6bad54b1de49de4bbe1318219d7ed1be27408cae7a773aa872` | — |
| P45 产线模拟、A1 其余配置、A2 连续 1 小时 | `p45/jetson_orin_nano_cfg/p45_clipb16_336_file_pipe.json` | 13,448 | `3eba60f6b293342168965eafb5ccda8475f8de89339afe1d64a7fd6feeb4f2d7` | — |
| P45 产线模拟、A1 其余配置、A2 连续 1 小时 | `p45/jetson_orin_nano_cfg/p45_clipb16_336_frame_seq.json` | 13,458 | `2dc0189ba7540b6f71cff34474058c20a9fb8edfedef19a25028e31d05ac721e` | — |
| P45 产线模拟、A1 其余配置、A2 连续 1 小时 | `p45/jetson_orin_nano_cfg/p45_dino_b_file_pipe.json` | 13,448 | `71c46fe879de8979dee28f4b8d13183199500f2372e9b5ad3932b1350556b3dc` | — |
| P45 产线模拟、A1 其余配置、A2 连续 1 小时 | `p45/jetson_orin_nano_cfg/p45_dino_b_frame_seq.json` | 13,443 | `8a93eac9a6eac23779012a0123b9a57bede232b28e6cc377343c539fdbbce61f` | — |
| P45 产线模拟、A1 其余配置、A2 连续 1 小时 | `p45/jetson_orin_nano_cfg/p45_dino_s_file_pipe.json` | 13,449 | `dd4c03f253b6e6d8c220a326478731eaf693df7609dccaf9e8924c5bf318a927` | — |
| P45 产线模拟、A1 其余配置、A2 连续 1 小时 | `p45/jetson_orin_nano_cfg/p45_dino_s_frame_seq.json` | 13,469 | `7e9c9c9cd8aaf36e503544d90403bf9496f6446e1fa674a0111a8fb701178340` | — |
| P45 产线模拟、A1 其余配置、A2 连续 1 小时 | `p45/jetson_orin_nano_cfg/p45_idle.json` | 161 | `ebaab993456c24db2af05c9965f38f7bba73b80b6f8276a7b0093e122ea690b4` | — |
| P45 产线模拟、A1 其余配置、A2 连续 1 小时 | `p45/jetson_orin_nano_sus/p45_sustain.json` | 23,759 | `532e2bb9bd1242ed015984af2c158db02b0574aff5c7a313803c55b286d992dc` | — |
| P46 部署引擎上的筛查 | `p46/jetson_orin_nano/p46_mtd__mtd6way.json` | 106,434 | `cb5bf2b11ac54ffd4dc6eaea5f25a91cca99af84b88004a26c738da3413e8af3` | — |
| P46 部署引擎上的筛查 | `p46/jetson_orin_nano/p46_results.json` | 32,724 | `2e8a77bcb4d765515824950d03274130ff37850ec946baa6d43dd70c6c3bf497` | — |
| P46 部署引擎上的筛查 | `p46/jetson_orin_nano/p46_visa__candle.json` | 89,184 | `f1181d628fd5a284a766a48f98452da1c67a482a461fe2f7123664720bde6b3b` | — |
| P46 部署引擎上的筛查 | `p46/jetson_orin_nano/p46_visa__cashew.json` | 51,603 | `1b16504173040d8439216085b938e236ad1e97dbf8413f751853d1f48287677f` | — |
| P46 部署引擎上的筛查 | `p46/jetson_orin_nano/p46_visa__chewinggum.json` | 49,380 | `f9a967865a3dba4d749d7e4bad055f8bfbd5aceccf525f72d497bf24916004fe` | — |
| P46 部署引擎上的筛查 | `p46/jetson_orin_nano/p46_visa__fryum.json` | 51,016 | `4cf77092cca03a58a059324d9939623d935ca99308c0f7ceab10f3d4b7624e6c` | — |
| P46 部署引擎上的筛查 | `p46/jetson_orin_nano/p46_visa__macaroni1.json` | 90,437 | `483131445371b062f1b324a641e6f871dea7028b9976308bf6edb030da6c8b58` | — |
| P46 部署引擎上的筛查 | `p46/jetson_orin_nano/p46_visa__macaroni2.json` | 87,367 | `2b11483aa0a79402709fdf315614ab836268e8d06a61f6215dccd26569fc8693` | — |
| P46 部署引擎上的筛查 | `p46/jetson_orin_nano/p46_visa__pcb1.json` | 89,514 | `dbfb644fc47bcbf80e712ac6217d43fe6631c779603de9eb5691865a2f13780e` | — |
| P46 部署引擎上的筛查 | `p46/jetson_orin_nano/p46_visa__pcb2.json` | 89,267 | `7732a57a68b5a9a5a84ab7d1d26cd1af7faf6559c5d2b2f946b21c7bcae906a6` | — |
| P46 部署引擎上的筛查 | `p46/jetson_orin_nano/p46_visa__pcb3.json` | 89,668 | `27193be63b53ecd0f2b3b5392829785ad9a5771355d8c90305d96598ec0b41fb` | — |
| P46 部署引擎上的筛查 | `p46/jetson_orin_nano/p46_visa__pcb4.json` | 84,979 | `be3848a0ae7c77a3ab9ed6259d8cb3a03fec8f51d1e59a8c111bfa2a8ae6f383` | — |
| P46 部署引擎上的筛查 | `p46/jetson_orin_nano/p46_visa__pipe_fryum.json` | 52,347 | `69ad77873b3a6f7748636c850175c794a6e05e7614c9a438169f1970d806361b` | — |
| P47 修订 A4：2 × 2 切块引擎上的筛查（P46 打分） | `p46/jetson_orin_nano_g2/p46_mtd__mtd6way.json` | 106,942 | `5c5a00ea9a6ff5a60d341e40cd4490a8aec48b5289e1b4048f22bd7c394b7802` | — |
| P47 修订 A4：2 × 2 切块引擎上的筛查（P46 打分） | `p46/jetson_orin_nano_g2/p46_results.json` | 32,818 | `311655f5954c7cdd1d6e4e571a88b5c07cb1601503addc70fa4b75d0a5953c23` | — |
| P47 修订 A4：2 × 2 切块引擎上的筛查（P46 打分） | `p46/jetson_orin_nano_g2/p46_visa__candle.json` | 87,748 | `41074332657710f6d7b952bddf5a102286d5cb66d84829efda6887210878d852` | — |
| P47 修订 A4：2 × 2 切块引擎上的筛查（P46 打分） | `p46/jetson_orin_nano_g2/p46_visa__cashew.json` | 50,030 | `0c33ee9df495f5d46c0b0185e0fde00840028a4d69c2a01f550b20fdef89b561` | — |
| P47 修订 A4：2 × 2 切块引擎上的筛查（P46 打分） | `p46/jetson_orin_nano_g2/p46_visa__chewinggum.json` | 48,389 | `f3293a4c3c0c41d6e8fb92ae880d5405768fabe23c199bc55e3cc375233b07a7` | — |
| P47 修订 A4：2 × 2 切块引擎上的筛查（P46 打分） | `p46/jetson_orin_nano_g2/p46_visa__fryum.json` | 49,605 | `52e5338671e0d38397d5f5fd1b4919660b1de76ed8568c8a0ca477d07e1b4054` | — |
| P47 修订 A4：2 × 2 切块引擎上的筛查（P46 打分） | `p46/jetson_orin_nano_g2/p46_visa__macaroni1.json` | 89,613 | `83a890e9da16de51856be436fdad170855510e89fff1eab3da8caa7f7a15d03c` | — |
| P47 修订 A4：2 × 2 切块引擎上的筛查（P46 打分） | `p46/jetson_orin_nano_g2/p46_visa__macaroni2.json` | 87,161 | `62fee25f07bc8fb1451881d575b9f9d3e12f9a8891782e3d44f9ffcef9db67e9` | — |
| P47 修订 A4：2 × 2 切块引擎上的筛查（P46 打分） | `p46/jetson_orin_nano_g2/p46_visa__pcb1.json` | 88,847 | `3f2d4501f4469918e107f8b53c23eb6f655eabc158614f30f716ee18d3993e9f` | — |
| P47 修订 A4：2 × 2 切块引擎上的筛查（P46 打分） | `p46/jetson_orin_nano_g2/p46_visa__pcb2.json` | 88,679 | `d9c5cdeddf7694769fbbaa36477bdfe98a1cfa9db8d1ddeeeea94d3b17261eca` | — |
| P47 修订 A4：2 × 2 切块引擎上的筛查（P46 打分） | `p46/jetson_orin_nano_g2/p46_visa__pcb3.json` | 89,376 | `8a9f7972b6380362cc6e2e1f78e6d6233c520d52c25b8c0205a9482ead5231b4` | — |
| P47 修订 A4：2 × 2 切块引擎上的筛查（P46 打分） | `p46/jetson_orin_nano_g2/p46_visa__pcb4.json` | 84,496 | `124a4f338f40914ccaed543d22a55cc2984c9455fd3585531d450c72716f3cb4` | — |
| P47 修订 A4：2 × 2 切块引擎上的筛查（P46 打分） | `p46/jetson_orin_nano_g2/p46_visa__pipe_fryum.json` | 51,263 | `c8e4d734d6faff67625acf20cd138196d427b97f24979125d252a8c83ffe7e92` | — |
| P47 规模扩展（M、类别数、SKU）与 2 × 2 切块计时 | `p47/jetson_orin_nano/pass1.json` | 3,581 | `8e80f420135381d2489fc0da57ebef731f34bc8558f5a62cd93241392f0789cd` | — |
| P47 规模扩展（M、类别数、SKU）与 2 × 2 切块计时 | `p47/jetson_orin_nano/pass2.json` | 3,574 | `21fcd70c25acb1acd41103d742af192c28a585dbce485cc7da15e2bbc0d88c09` | — |
| P47 修订 A2/A3：3 × 3 切块计时（M = 10） | `p47/jetson_orin_nano_g3/pass1.json` | 614 | `583707910d67e58983b3602cec4adfd90628a5777ea00e6785d1b57bc0a462be` | — |
| P47 修订 A2/A3：3 × 3 切块计时（M = 10） | `p47/jetson_orin_nano_g3/pass2.json` | 612 | `da7c1234498dd5e4176fa7c6f266c221294eca56b0a61e4d0b9b4fe79bebd288` | — |
| P47 修订 A1：P41 判决翻转拆分（本地，基于设备特征） | `p47/flips/p41_copy/p41_scores.json` | 104,859 | `b948b37d9a7fd225a344563c4e5d7076e991cecc503794cab1564ea29a5f3d08` | — |
| P47 修订 A1：P41 判决翻转拆分（本地，基于设备特征） | `p47/flips/p47_flips.json` | 730 | `d814e4089fab749af4e55998a10bb0b241bafbf5bc0b179945890260c1de386a` | — |
| P47 切块精度（本地 FP32，g = 1 / 2 / 3 打分结果） | `p47/mvtec_fs/g1/p41_scores.json` | 102,937 | `73a891c8df216fa4d89214401b13dad66ffff22c856c3a6c39837bedad59156e` | — |
| P47 切块精度（本地 FP32，g = 1 / 2 / 3 打分结果） | `p47/mvtec_fs/g2/p41_scores.json` | 103,930 | `65dc2ee81258a5b2bf31e298502033fe5a7e2f0b3d253185aa4b19c2856c3c81` | — |
| P47 切块精度（本地 FP32，g = 1 / 2 / 3 打分结果） | `p47/mvtec_fs/g3/p41_scores.json` | 103,759 | `b74fec3bf92895179dd1fd23e38258fad0c832977a3899ab91848d13e92628e7` | — |
| P48 部署路径上的 patch 级误选率（参考值本地、设备值引擎） | `p48/jetson_orin_nano/p48.log` | 2,658 | `edd73de3e972c53e2a92717c8c24999d1b5f5d282c876261403b447624d5a351` | 1 处 |
| P48 部署路径上的 patch 级误选率（参考值本地、设备值引擎） | `p48/jetson_orin_nano/p48_device.json` | 15,947 | `d025627c6045dfc39561aa099649597a6414d028e44b709a894bbdddc5dad9db` | 1 处 |
| P48 部署路径上的 patch 级误选率（参考值本地、设备值引擎） | `p48/p48_reference.json` | 51,307 | `73aa149d614c48234661da3b4018fb6df2f53abb7a3a38affaccda145c73b40a` | — |
| 设备运行日志 | `device_logs/p47_r5/p47_chain_r5.log` | 580 | `a61ae59d2e2db8041d4b32ff575eae5aa8a667a790ac018b0311284b8e25564a` | — |
| 设备运行日志 | `device_logs/p47_r5/p47_check_dyn_r4.log` | 1,109 | `1bde8f62aeeb8509b02e88aa69fc712b0e4ee9e8e59243a906c48fd2719cf0ae` | — |
| 设备运行日志 | `device_logs/p47_r5/p47_check_g2_r4.log` | 421 | `599071e2c62f43861cc887cd80b38a44eb5955e182a978e6ca30cc8491313e86` | — |
| 设备运行日志 | `device_logs/p47_r5/p47_dyn_build.log` | 11,209 | `7f5781d9f0e40834dfac0fc9dec5edb0ef99603d0cd73611fc466e27157ebbd0` | — |
| 设备运行日志 | `device_logs/p47_r5/p47_export_dyn.log` | 3,222 | `ddf1efc363504a08779e84a93434eee9325ebbc2eb176d06ffb27c0d89be93d8` | 12 处 |
| 设备运行日志 | `device_logs/p47_r5/p47_export_dyn_r2.log` | 2,178 | `9f04809d61843eea943b22a74e62e056b3fa371f282c0666ee5b0f7f137b051d` | 7 处 |
| 设备运行日志 | `device_logs/p47_r5/p47_export_dyn_r3.log` | 479 | `ac98cf548a63f435329beffadd7f670d5efec1b094402cf5ae49d5801b6412a5` | 2 处 |
| 设备运行日志 | `device_logs/p47_r5/p47_export_dyn_r4.log` | 2,414 | `10b74b5042425cc824f1943f9a268225e73dc45c9636f4c676f93cc310ce7359` | 7 处 |
| 设备运行日志 | `device_logs/p47_r5/p47_export_g2_r4.log` | 2,640 | `f7a2a08254236210e94fa053de17af926bc3073576c857efe45bd70c579043c7` | 7 处 |
| 设备运行日志 | `device_logs/p47_r5/p47_export_g3_r4.log` | 2,120 | `e8e313450868496fdaaf74c462d7a2bf239124dbfbc2f3aae71f9bc1b80ad67c` | 10 处 |
| 设备运行日志 | `device_logs/p47_r5/p47_g2_build.log` | 9,878 | `e36dc0f49994d986a90db0ccda841e1746fd44a1fee7c2ad5404463ddac890c0` | — |
| 设备运行日志 | `device_logs/p47_r5/p47_status_r5.txt` | 209 | `0e096c2a6c19d3aa6e31b9e69f8d8a4dc167c40eb3a3bf57c1091154e40ab0ff` | — |
| 设备运行日志 | `device_logs/p47_r5/p47_tegrastats.log` | 200,234 | `3f28a45d5406119ba34c0ca8b84d2f30f6856c4838c96ec8c12295e9c4fd5a93` | — |
| 设备运行日志 | `device_logs/p47_r5/p47_time_1.log` | 1,886 | `2359853968bb0a18a33d4d73bf7af4d51d832158d12e2c8bbe51a7f59a83a4cd` | 1 处 |
| 设备运行日志 | `device_logs/p47_r5/p47_time_2.log` | 1,886 | `29a3a5b0a01f7242cab115fdd994d52a7d464d7acb3d90e769785322bd66c4b9` | 1 处 |
| 设备运行日志 | `device_logs/p47_g3/p47_check_g3_a3.log` | 421 | `3dcd226311096eb330b59717134ff9577dcf25d208470599c083661c771f7a08` | — |
| 设备运行日志 | `device_logs/p47_g3/p47_export_g3_a3.log` | 2,639 | `9d1c6a63791c5ab2d958fd228af8bea51266c3ebf2f34c6ff69b485c60b726d5` | 7 处 |
| 设备运行日志 | `device_logs/p47_g3/p47_g3_a3.log` | 157 | `2ef3233dced9efd5d0045aa6452a43b291c53d58516fa74a5cead397e0b21d6e` | — |
| 设备运行日志 | `device_logs/p47_g3/p47_g3_build.log` | 9,879 | `5535dcc712ace1d142014ab050167018c731b6c8c991ed48de777494ead88523` | — |
| 设备运行日志 | `device_logs/p47_g3/p47_g3_status_a3.txt` | 159 | `45cbf369f3ec24fe2a401dc02073e3bdc5f71df865a698138846f2f6903c1798` | — |
| 设备运行日志 | `device_logs/p47_g3/p47_g3_time_1.log` | 892 | `0771dbb820672b14d9507cd0ef5bf9f0108ea888025d057b003573ebd0df1b6a` | 1 处 |
| 设备运行日志 | `device_logs/p47_g3/p47_g3_time_2.log` | 892 | `bae4e2f5e2259e9153f2541ba4d00d69d4fbfaa5e0827f83bda0e0fd577a65d6` | 1 处 |
| 设备运行日志 | `device_logs/p47_a4/p47_a4.log` | 1,331 | `d372851bdfac58aaafa5b7365afb7bb74a6cbba7cacc83eef6f9e2ecc2295299` | 1 处 |
| 设备运行日志 | `device_logs/p43/int8/clip_l14_336_p43i8_build.log` | 9,084 | `2d03e8f9ae4c1584824ad8bec7bdf870010e39cbcddc06cbbd6b6ee5faae8f0b` | 7 处 |
| 设备运行日志 | `device_logs/p43/int8/clip_l14_336_p43i8_time.log` | 8,276 | `c6e9d3c1871e6c375463ce1f1da01d826efbbd7c3f02ac20193530ba7881f34b` | 2 处 |
| 设备运行日志 | `device_logs/p43/int8/clip_l14_336_r2_fp16_p43i8_anchor_time.log` | 8,569 | `e706f3aaed0bd8278dd13bb94163825e25968f973ca6a9f5e27a1d1751ce2609` | 2 处 |
| 设备运行日志 | `device_logs/p43/int8/dinov2_b_336_ex_p43i8_build.log` | 8,650 | `ad781a7e7a930022a8790f80c9b26e557db7ee7736837a19675afd095c5c73fb` | 7 处 |
| 设备运行日志 | `device_logs/p43/int8/dinov2_b_336_ex_p43i8_time.log` | 8,719 | `37fbd74fe8008aea81b0b8742c666dbc5e5af94534e1f102b2447e98267b5602` | 2 处 |
| 设备运行日志 | `device_logs/p43/int8/dinov2_b_336_ex_r5_p43i8_anchor_time.log` | 8,831 | `e71ea435ee43b4ce196f968abea905486f297570d02009e87f27847fd068a6fa` | 2 处 |
| 设备运行日志 | `device_logs/p43/int8/dinov2_s_336_ex_p43i8_build.log` | 8,647 | `3edbb7dd92b9ed3dc9769615875e398a70e7e2a9122f10a43fff8e6569eb0fc8` | 7 处 |
| 设备运行日志 | `device_logs/p43/int8/dinov2_s_336_ex_p43i8_time.log` | 8,308 | `a25bcf308d35c941cd9289523dc8b037c74e8d307651ccd7c0cda3039246d700` | 2 处 |
| 设备运行日志 | `device_logs/p43/int8/dinov2_s_336_ex_r5_p43i8_anchor_time.log` | 8,424 | `65269c452311e463e9d5e124d75846402641dece5e111010f47c16e2748babd7` | 2 处 |
| 设备运行日志 | `device_logs/p43/int8/p43_int8.log` | 2,029 | `d79c662365eacbf2cb827334593a633383d61758a975ec574c3fa9a6cd167bdc` | — |
| 设备运行日志 | `device_logs/p43/int8/p43_int8_summary.txt` | 2,083 | `6a986706533c8ed0d500655caf3ad976bf9e57a48526da5608cc32486281138e` | — |
| 设备运行日志 | `device_logs/p43/main/p43_cold.log` | 3,894 | `9f5dd7d4b9fffc6b2e58510c94a35dee2291dec4c312d0dc1a843d92f26b8a98` | 3 处 |
| 设备运行日志 | `device_logs/p43/main/p43_energy.log` | 3,951 | `91cfd8a9608bc7cb6ea089bcfb7530909160791ae7c09a4fc0f0970ed82e2ed0` | 3 处 |
| 设备运行日志 | `device_logs/p43/main/p43_latency.log` | 1,133 | `d5dcdd5cd1ab47d81f2486f95f386574762630694320655f268f3a70e12b9e17` | 1 处 |
| 设备运行日志 | `device_logs/p43/main/p43_night_chain.log` | 436 | `360add7c2bafe55968850f46e3c3cddc2d5c2c51ab1a3ddbd1c31d58b54ed56d` | 1 处 |
| 设备运行日志 | `device_logs/p43/main/p43_single.log` | 1,241 | `00c602ee32b04501c7ba6ce6d31e341012dd9a65846de995655e2a7c3998c524` | — |
| 设备运行日志 | `device_logs/p43/main/p43_status.txt` | 252 | `b35da28fb02299603e5b5817c2506490607e6d1e5fe1907b9475c26ace785e68` | — |
| 设备运行日志 | `device_logs/p43/main/p43_tegrastats.log` | 1,736,468 | `d7df19929a1bbff4992b895e489d605ed3d18418fc2900ca5319a7dc340ab719` | — |
| 设备运行日志 | `device_logs/p43/modes/logs/p43_energy_15w.log` | 3,608 | `aa8829305dbd6b78de7acbed72e939d7e1d9e8ba4b0a8fb5ca1f99fd936f5627` | 3 处 |
| 设备运行日志 | `device_logs/p43/modes/logs/p43_energy_25w.log` | 3,609 | `4636744c69f73c4cb5379bb7ab5a67f8327c6f672f06eb7f544093a0eddc93c3` | 3 处 |
| 设备运行日志 | `device_logs/p43/modes/logs/p43_modes.log` | 374 | `617ba8e4ee0e5d3f5d7a2bbb89adba2170701f8e8b63ef3984338911b6eaf914` | — |
| 设备运行日志 | `device_logs/p43/modes/logs/p43_modes_status.txt` | 100 | `4ce18697af8cfc12fa6190e3c1b8f82944298f28a1bd2cab2f1e1c1c9d4d2673` | — |
| 设备运行日志 | `device_logs/p43/modes/logs/p43_repeat_15w.log` | 1,049 | `6def20151d41b9a0123095b5d1a57dbe1ea3199840f857de9bc7175c6577fdc5` | 1 处 |
| 设备运行日志 | `device_logs/p43/modes/logs/p43_repeat_25w.log` | 1,049 | `691ed545ca536c04a7afe402a03fd4017e1005b399d9dff471be425c26e9756b` | 1 处 |
| 设备运行日志 | `device_logs/p43/modes/logs/p43_tegrastats_15w.log` | 955,200 | `510c7a87866233ddb33909a3f33d44d273ef107031eb64f5d69dacf5908fa9d9` | — |
| 设备运行日志 | `device_logs/p43/modes/logs/p43_tegrastats_25w.log` | 952,166 | `0b5bbf74151558d80a66226a99c6b6308177d94b601f32ad6cad0186b6a2374f` | — |
| 设备运行日志 | `device_logs/p43/modes/p43_energy_15w.log` | 3,608 | `aa8829305dbd6b78de7acbed72e939d7e1d9e8ba4b0a8fb5ca1f99fd936f5627` | 3 处 |
| 设备运行日志 | `device_logs/p43/modes/p43_energy_25w.log` | 3,609 | `4636744c69f73c4cb5379bb7ab5a67f8327c6f672f06eb7f544093a0eddc93c3` | 3 处 |
| 设备运行日志 | `device_logs/p43/modes/p43_modes.log` | 374 | `617ba8e4ee0e5d3f5d7a2bbb89adba2170701f8e8b63ef3984338911b6eaf914` | — |
| 设备运行日志 | `device_logs/p43/modes/p43_modes_status.txt` | 100 | `4ce18697af8cfc12fa6190e3c1b8f82944298f28a1bd2cab2f1e1c1c9d4d2673` | — |
| 设备运行日志 | `device_logs/p43/modes/p43_repeat_15w.log` | 1,049 | `6def20151d41b9a0123095b5d1a57dbe1ea3199840f857de9bc7175c6577fdc5` | 1 处 |
| 设备运行日志 | `device_logs/p43/modes/p43_repeat_25w.log` | 1,049 | `691ed545ca536c04a7afe402a03fd4017e1005b399d9dff471be425c26e9756b` | 1 处 |
| 设备运行日志 | `device_logs/p43/modes/p43_tegrastats_15w.log` | 955,200 | `510c7a87866233ddb33909a3f33d44d273ef107031eb64f5d69dacf5908fa9d9` | — |
| 设备运行日志 | `device_logs/p43/modes/p43_tegrastats_25w.log` | 952,166 | `0b5bbf74151558d80a66226a99c6b6308177d94b601f32ad6cad0186b6a2374f` | — |
| 设备运行日志 | `device_logs/p45/cfglogs/p45cfg.log` | 17,857 | `ceed782ed0347947e6cd3d21c047bdfcc23f1aa05339d2fb1c1c4d67e870dbc6` | — |
| 设备运行日志 | `device_logs/p45/cfglogs/p45cfg_status.txt` | 12 | `80e0c64f9e23c30e1e372cc31d70bcbaabb6ecc661c7cc40e885bad1d1896488` | — |
| 设备运行日志 | `device_logs/p45/cfglogs/p45cfg_tegrastats.log` | 5,962,211 | `e6ebbee5858d1efd6e7caf9d804224c4758c59a61a0d428ea8a61ed975144818` | — |
| 设备运行日志 | `device_logs/p45/logs/p45.log` | 6,312 | `570b8c0a38722abd055bf000b18073d3909c62dea26daa9fbc05d3ccd5a0f3c7` | 3 处 |
| 设备运行日志 | `device_logs/p45/logs/p45_status.txt` | 9 | `3a0050a0585d2d43ab4c6f856eb43dd486614b81b53394c0ff48863224b70620` | — |
| 设备运行日志 | `device_logs/p45/logs/p45_tegrastats.log` | 2,895,679 | `9a5fe42f79dde929721dbe7c870c2747731ba207ea511e7016f5407bd8dfc09f` | — |
| 设备运行日志 | `device_logs/p45/suslogs/p45sus_tegrastats.log` | 1,110,506 | `d693a7dd7533b1e22099e969f9358998d02db06230c9cdecf0723adcc71716fc` | — |
