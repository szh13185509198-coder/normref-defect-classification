# Result files

Every JSON / JSONL result file listed in Supplementary Material S2.4, under the same path as in the research
repository. The archives S2.4 also lists (`ckpt/p29/p29_out.tar.gz`, `ckpt/p26_cloud/p26_rtx4090_r3_results.tar.gz`,
`p31_r3_results.tar.gz` and the P33 / P43 device archives) hold the raw per-episode and per-pass outputs of the
cloud and device runs that the JSON files above summarize; they are not included here.
The second column is the SHA-256 of the file as written by the run (the value S2.4 cites, there truncated);
the third says whether the copy here is byte-identical or had machine names replaced by placeholders
(`<USER>` for the device user name, `<cloud-container>` / `<cloud-dir>` for the rented cloud machine; no number
is changed). Placeholder copies hash differently; replace the placeholders back to check them.

Markdown tables (`*_table.md`, `*_report.md`) are summaries written by the same runs; S2.4 cites their JSON.

| File | SHA-256 of the original | Copy here |
|---|---|---|
| `ckpt/p10_cable/results/cable_results.json` | `28b433e29fa735d6ec47f7baaecd61d90b5feee7c6d196d29dd39da6dd532f1a` | identical |
| `ckpt/p10_external/results/external_results.json` | `05060cbec6adad6c456102af10061651e9006901b01d0f1af86dc3da79b084cf` | identical |
| `ckpt/p10_formal/results/formal_results.json` | `5d94619baa6933ee175cad9308ce16a4410532b33f6c4f47ef916a2a4b9da96c` | identical |
| `ckpt/p10_pilot/pilot_results.json` | `869260db336fbf6f1e0fae9a778e5818ea60a8fc6ad8dfbb4780f9105c2e097b` | identical |
| `ckpt/p11_formal/external/p11_formal_external_predictions.jsonl` | `f755175b4e7d98aaf13e30771e2b696d64d36f839bc52309b502fbc21311a9d2` | identical |
| `ckpt/p11_formal/external/p11_formal_external_results.json` | `88f8b5b0faef994d1314210e00a5fcbad99cb09ffd5dd78d4434f09fa4181858` | identical |
| `ckpt/p11_formal/external/p11_formal_external_table.md` | `20ba141cf0988086ac5266c825c91ca54bf010ce35c4feff3d6cb928fa6036cc` | identical |
| `ckpt/p11_formal/mvtec/p11_formal_mvtec_results.json` | `d44cbd131b8c63f737a080ec7597b90d83233143c86e45e38ad6acdc9cbe19cf` | identical |
| `ckpt/p11_formal/mvtec/p11_formal_mvtec_table.md` | `428926d3576699d5cd0d5620bdd04d8f533baa20cb0f9ebd13398bd06efd9021` | identical |
| `ckpt/p11_pilot/pilot_ab_results.json` | `82561efa6d38fb623debe5d2f1a6d71b4ed374f8b9b03838a520187810661bef` | identical |
| `ckpt/p12/threshold_external/threshold_external_results.json` | `4f389dcfb99fbee7e6f7f5c08e142be06c714b5ad726415760d5c80564db5272` | identical |
| `ckpt/p12/threshold_mvtec/threshold_mvtec_results.json` | `048eb181e260bda19cb2a5c76f08b8a30f948346ce4c3bb739143a61cac0ae18` | identical |
| `ckpt/p13_pilot/dev/p13_dev_results.json` | `fcb5c59be8538e679f6569feef9278e63e8ebe8110edab89a6600d6ee3e76a05` | identical |
| `ckpt/p14_formal/cable/p14_cable_results.json` | `161ed833dd2df78d7d220a50d4849d426a9ee6b32334a5093e5c74f87bb7c016` | identical |
| `ckpt/p14_formal/external/p14_external_results.json` | `dd2915db6092f31a568ed5a7e6e3c1f02e11bb0ff7098851d9be66d11b5179a6` | identical |
| `ckpt/p14_formal/mvtec/p14_mvtec_results.json` | `c5a0c22638dee7fcb2a4665849f9bcfec2bc8e09c1170c3978065b7638428ee2` | identical |
| `ckpt/p15/cable_test/p15_cable_test_results.json` | `03580085ac883d6d6acae7576eb034c2b0656688e58e295b4ef79a1ef5f525d8` | identical |
| `ckpt/p15/external/p15_external_results.json` | `66f270d3476dc24d30ebf9d04f3adf30be93829b70d997e01e9e736aa22e8bf5` | identical |
| `ckpt/p17/external/p17_external_results.json` | `65a1202398836fc56357f39edaf6f6ce85aeffc769c3f57f7895c6844187fdd5` | identical |
| `ckpt/p17/mvtec/p17_mvtec_results.json` | `a8d930a3a7087a466df77de1d4f582883489a46f93c207b73545ce6f825ac4bc` | identical |
| `ckpt/p18/external/p18_external_results.json` | `e67ffcca0cc10997ee7a3ba2a6860996d4828e60213bf2fde1d5c8349148cd1c` | identical |
| `ckpt/p18/mvtec/p18_mvtec_results.json` | `8dbd7fe94f74c789bd4983aac3b436965e865481816f285e1af8168cc44d7925` | identical |
| `ckpt/p19/blind/p19_results.json` | `eaa8d11e913f9bf16afad5d7af02bce3719e03940bf1295fe358b1e787ad792c` | identical |
| `ckpt/p21/mvtec/p21_results.json` | `29ee02ecbd300e95447a2fd49618deb1024fafdf7a9ad1f0cd2ff37b402c5a1a` | placeholders (60) |
| `ckpt/p22/formal/external/p22_formal_external_results.json` | `8cb183af137de3447da81495abe5a627315bb0e586a7b560186bebcba9787dad` | identical |
| `ckpt/p22/formal/mvtec/p22_formal_mvtec_results.json` | `719fe0e4fbb2ecd8d2bbf177e83dffac4b3d2001aee0ed9c2806919968e3c0a6` | identical |
| `ckpt/p23/formal/p23_results.json` | `14314ce7f94c818b2d80d75e701afcc8bbc877c9242c9758057b7c15a7aac4e3` | identical |
| `ckpt/p24/mvtec/p24_results.json` | `0f7398dc60baca8eb48ca7b120259b8e2a318cb39a322f23974e5c63e8c82ec8` | identical |
| `ckpt/p25/mvtec/p25_results.json` | `49c26389c052f70608777663e229189cb6c1d0f7cb14a4d8c7ffa26952033f8a` | placeholders (38) |
| `ckpt/p26/jetson_orin_nano_r2/cost_results.json` | `e55fa506fdf373b052751c841a9ed57115342f921bc8d12b76be85fb732f14a2` | placeholders (22) |
| `ckpt/p26/jetson_orin_nano_r3/cost_results.json` | `8d6557ec68b15f8fd0f981486d8f8f866c738ae76ff9193d0e011310f8b5cd22` | identical |
| `ckpt/p26_cloud/rtx4090_r2/cost_results.json` | `53695fcb6e59dc1b70cf6c49bbe2cf26afaf7f693ff6b92a66cfbac3303d62d2` | identical |
| `ckpt/p26_cloud/rtx4090_r3/cost_results.json` | `df2f6cef7836d1cfdcb8277d32bbd104154feafc910a96405072305c069afdf0` | identical |
| `ckpt/p27/external/p27_results.json` | `ee9c4bbf8fbed17b0331a5acbe989ce36a7bc67d3ad4e50c5c4caed16332d784` | identical |
| `ckpt/p28/p28_external_breakdown.json` | `95addb1cc0885836e8056c7b107a4482747e27350cd0a365c431c721016501d0` | identical |
| `ckpt/p28/p28_external_error_decomposition.json` | `82c23afaff19d060a8e2752fd77baa9ab8dd108f9d20a5c07156d0b65a0650af` | identical |
| `ckpt/p28/p28_mvtec_error_decomposition.json` | `61f8052442545de7ca7d0c31bd1ee0bd2355d9caf6a215e6a40834e2f77ae7af` | identical |
| `ckpt/p29/mvtec/p29_results.json` | `ea2e87f2471f0dbf94935cb77cdb8baabaf02a8d851be265177b89f52617662f` | placeholders (8) |
| `ckpt/p30/fp16/p30_fp16_results.json` | `3400078ba2aa2f51c48035376bd756c2987b4d92b5bc39c5d16ec75b5314808c` | identical |
| `ckpt/p32/jetson_orin_nano+jetson_orin_nano_b/p32_scores.json` | `1d1fd6a05cb4a41e08b48d59883dd4e00361ba8db43c61f487fe1414e4728ee0` | identical |
| `ckpt/p32/jetson_orin_nano_trt/p32_scores.json` | `43ff1f830ffa30759c9c6c44e43b30adc8919cdb83e0be1b498e7bced1d5a2b3` | identical |
| `ckpt/p33/sustained_maxn/p33_power_ours_main_fp16.json` | `e612a960006ff1aab9757d9d7f32d35cedf83a81370c1ec18518eac6c9f05454` | identical |
| `ckpt/p34/p34_results.json` | `a677479d9d9939b6f284166a739d46ac3d3ea87c288d52b1053a1b580a864c4e` | identical |
| `ckpt/p36/jetson_orin_nano/p36_results.json` | `e3e2120082e065fd4d252cdd3d4a8c796b04890ec15408c275a9f25b2b668cdd` | identical |
| `ckpt/p37/cable_test/p37_cable_test_results.json` | `d26bef1c2f1d7c2263f7dd13dd636e2504ff7ca18c9dda40f745320f76b55dad` | identical |
| `ckpt/p37/cable_test/p37_cable_test_table.md` | `da3c6da658f5c8c1858822d22834c41990251cb7cb279db23242d4ba3d197c86` | identical |
| `ckpt/p37/dev/p37_dev_results.json` | `c7ffded07c3e297d8f3cd426cff7f9654308c395ca8588bae6f23c71a90e6db1` | identical |
| `ckpt/p37/dev/p37_dev_table.md` | `1e6a437859fce77937907cee418648ba226f7c2ff46d8137d292e417e0dc40a0` | identical |
| `ckpt/p37/external/p37_external_results.json` | `bbd9a5c2441dec09902bd833b88cfdeb4e96cf2e56ced08fcea9aa30df779604` | identical |
| `ckpt/p37/external/p37_external_table.md` | `0f695dfb73c54fd0d40e0e6b119d37b38074bed285aada8706a99ff314389e58` | identical |
| `ckpt/p37/report/p37_report.json` | `10094a9082ad6cf87b2564d425cf1634e52e9476972048ff749f865f2bf037c3` | identical |
| `ckpt/p37/report/p37_report.md` | `e505ba3bfdee5ab514e5201b40864158c17dd54a2ba3f62fdf012d75e9e6ab80` | identical |
| `ckpt/p38/jetson_orin_nano/p38_results.json` | `9b1768944087b0cfeb562ebf08e3bca568cf2d83dd3c16199a8dbd8daf8b27b6` | identical |
| `ckpt/p38/jetson_orin_nano_l18/p38_results.json` | `ff4db46e5249c332ca2f9eeddc706501de6549ab2c916d3295b1d05da647af9a` | identical |
| `ckpt/p40/cable_test/p40_cable_test_results.json` | `5fdf7d4752962704d9fa2fa8bef3aee106f1f8632364d8fc8449ea27fb273ccd` | identical |
| `ckpt/p40/cable_test/p40_cable_test_table.md` | `35d904d920c8d801042854a7a261b3e34a374af27be6f0e8c9dd48dcccf83c0e` | identical |
| `ckpt/p40/dev/p40_dev_results.json` | `091ff13ed69006f887df469c6312efe67dcea0124207b0982d71d5014003eed0` | identical |
| `ckpt/p40/dev/p40_dev_table.md` | `bf2ef853df86d794568f4d992b4923d7e9885ecf37d8ca5fbee6e9fc9c5c159f` | identical |
| `ckpt/p40/external/p40_external_results.json` | `37f6c8ad04306e5ac5fab58631684937822825097a436b18555ca45688e28e31` | identical |
| `ckpt/p40/external/p40_external_table.md` | `b288b3ef7881bdf9ed0ed63d200112aa9d5c5d44b011d995fdf9a0e14d5c223d` | identical |
| `ckpt/p40/report/p40_report.json` | `cab9f3259758f18b39eda4cc3111b87017d05f6ea0ae19c910570ee8df9cc741` | identical |
| `ckpt/p40/report/p40_report.md` | `3427f9ad1c63eac95831b7879726a814a6acbda7485b51c39d39455d5aa7121c` | identical |
| `ckpt/p41/jetson_orin_nano/p41_scores.json` | `06ee0dfa2b9d49956b66b15a385c2c4943e3269ce4bdf360b5d849bfaa40791b` | identical |
| `ckpt/p41/jetson_orin_nano/pass1_fused.json` | `6f18397fce2b06796402f445fd56da326482588a2b4c12e4c8da8e8dc0810ea2` | placeholders (1) |
| `ckpt/p41/jetson_orin_nano/pass2_fused.json` | `a5ccab56da683b6ed8e5fd1059e3f497a70724fa880faf39725b23dc07521cac` | placeholders (1) |
| `ckpt/p43/jetson_orin_nano/p43_latency.json` | `3a6e4e809997ddc2c6c1723fa833d9027b4d3bf1685b4b5b91f28fe6c4f932fe` | placeholders (1) |
| `ckpt/p43/jetson_orin_nano/p43_single_scores.json` | `ab426e8692fafc37793f4adfb58697ebf79a4b569f9908f1bd7cfb04f072b08d` | identical |
| `ckpt/p44/p44_results.json` | `c4748df68fb908ae6d3c51989e14e5d7fbf2dcbfc7d2396e259706d685993f2a` | identical |
| `ckpt/p45/jetson_orin_nano_sus/p45_sustain.json` | `532e2bb9bd1242ed015984af2c158db02b0574aff5c7a313803c55b286d992dc` | identical |
| `ckpt/p46/jetson_orin_nano/p46_results.json` | `2e8a77bcb4d765515824950d03274130ff37850ec946baa6d43dd70c6c3bf497` | identical |

Other run outputs (device logs, per-pass timings, uncertainty-evaluation outputs) are under `ckpt/` as well;
their original hashes are listed in `ckpt/README_DEVICE_2026-10-03.md`, `ckpt/p47/p47_jetson/README_U6D.md` and
`paper_figures/data/README.md`.
