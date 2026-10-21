#!/usr/bin/env python3
"""P35 diagnostic: the DINOv2-S `_pb` ONNX with intermediate tensors exposed as extra outputs, and onnxruntime (CPU, FP32)
values of every output as the reference, to locate where the TensorRT engine starts to disagree.

Usage:
    D:\Anaconda\envs\cole\python.exe -m experiments.p35.make_debug_onnx --dir D:\intergration\p35_trt
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import onnx

TAPS = ["/patch_embed/Transpose_output_0", "/Concat_output_0", "/Add_output_0",
        "/blocks.0/norm1/LayerNormalization_output_0", "/blocks.0/attn/qkv/Add_output_0",
        "/blocks.0/attn/Sqrt_1_output_0", "/blocks.0/attn/Mul_output_0", "/blocks.0/attn/MatMul_1_output_0",
        "/blocks.0/attn/proj/Add_output_0", "/blocks.0/Add_output_0", "/blocks.0/mlp/act/Mul_1_output_0",
        "/blocks.0/Add_1_output_0", "/blocks.5/Add_1_output_0", "/blocks.11/Add_1_output_0"]


def main(argv=None) -> int:
    import onnxruntime as ort
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, type=Path)
    a = ap.parse_args(argv)
    src, dst = a.dir / "dinov2_s_336_pb.onnx", a.dir / "dinov2_s_336_dbg.onnx"
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    m = onnx.load(str(src))
    produced = {o for n in m.graph.node for o in n.output}
    missing = [t for t in TAPS if t not in produced]
    if missing:
        raise SystemExit(f"tensors not in the graph: {missing}")
    for t in TAPS:
        m.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    onnx.save(m, str(dst))
    x = np.fromfile(a.dir / "dinov2_s_336_pb_input.bin", dtype=np.float32).reshape(1, 3, 336, 336)
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(None, {"input": x})
    np.savez(a.dir / "dinov2_s_336_dbg_ref.npz", **{n: o for n, o in zip(names, outs)})
    shutil.copyfile(a.dir / "dinov2_s_336_pb_input.bin", a.dir / "dinov2_s_336_dbg_input.bin")
    for n, o in zip(names, outs):
        print(n, o.shape, o.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
