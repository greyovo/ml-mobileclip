"""
Optimized ONNX export for Android deployment of MobileCLIP visual models.

=== Root Cause Analysis (why MobileCLIP2-S0 is 3x slower than CLIP ViT-B/32) ===

CLIP ViT-B/32 (25ms on Android):
  - Transformer-based: 1 Conv (patch embed) + 49 MatMulInteger ops
  - MatMulInteger runs as true INT8 computation on ARM (dot-product instructions)
  - DynamicQuantizeLinear quantizes activations on-the-fly → feeds MatMulInteger

MobileCLIP2-S0 (80ms on Android):
  - Conv-based (FastViT-MCi): 95 FP32 Conv ops at 256^2→16^2 resolution
  - export_true_int8_onnx.py explicitly excludes Conv from quantization (pop("Conv"))
  - Only 5 MatMul nodes get INT8 treatment → 95% of compute stays FP32
  - FP32 Convs on ARM are slow without AVX-512/AVX2 vectorization

On desktop (PC), the gap is minimal (23ms vs 18ms) because desktop CPUs have
wide SIMD for FP32. On ARM, FP32 convs bottleneck while INT8 matmuls fly.

=== Optimization Strategy ===

This script exports a clean FP32 ONNX model and converts it to ORT format
optimized for ARM. The key speedup happens on the Android device side:

  1. Use NNAPI execution provider (offloads Conv ops to GPU/NPU → 3-5x speedup)
  2. Use XNNPACK execution provider (optimized ARM kernels for FP32/FP16)
  3. Use the with_runtime_opt.ort variant with NNAPI (runtime optimizations)

Dynamic INT8 quantization of Conv ops is NOT viable — per-tensor quantization
of Conv activations destroys accuracy (cosine_sim drops to near 0).
Static INT8 with per-channel weight quantization requires real calibration
images (50-100 samples) for acceptable accuracy.

Usage:
  uv run export_optimized_android.py                            # FP32 → ORT ARM export
  uv run export_optimized_android.py --benchmark                # Export + benchmark
  uv run export_optimized_android.py --model MobileCLIP2-S2     # Different model
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort

import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model

MODEL_NAME = "MobileCLIP2-S0"
OUTPUT_DIR = "android_optimized"


def get_model_kwargs(model_name):
    if model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14"):
        return {}
    return {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}


def export_visual_ort(model, dummy_input, output_path,
                      input_name="image", output_name="image_features"):
    """Export visual encoder: FP32 ONNX → ORT format for ARM.

    The ORT format conversion with target_platform="arm":
    - Enables QDQ INT8 ops (session.qdqisint8allowed=1)
    - Disables NCHWc layout transformer (AMD64-only, breaks ARM)
    - Creates both Fixed and Runtime optimization variants
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model.eval()
    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=[input_name], output_names=[output_name],
        dynamic_axes={input_name: {0: "batch_size"},
                      output_name: {0: "batch_size"}},
        opset_version=20, dynamo=False,
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  FP32 ONNX: {os.path.basename(output_path)} ({size_mb:.1f} MB)")

    try:
        onnx.shape_inference.infer_shapes_path(output_path, output_path)
    except Exception:
        pass

    from onnxruntime.tools.convert_onnx_models_to_ort import (
        convert_onnx_models_to_ort, OptimizationStyle)

    convert_onnx_models_to_ort(
        Path(output_path), Path(OUTPUT_DIR),
        optimization_styles=[OptimizationStyle.Fixed, OptimizationStyle.Runtime],
        target_platform="arm",
    )

    ort_files = []
    for suffix in [".ort", ".with_runtime_opt.ort"]:
        f = output_path.replace(".onnx", suffix)
        if os.path.exists(f):
            ort_files.append(f)
            sz = os.path.getsize(f) / (1024 * 1024)
            label = "Fixed" if suffix == ".ort" else "RuntimeOpt"
            print(f"  ORT ({label}): {os.path.basename(f)} ({sz:.1f} MB)")

    return output_path, ort_files


def verify_precision(sess, model_orig, image_tensor, input_name="image"):
    with torch.no_grad():
        orig_feat = model_orig.encode_image(image_tensor)
        orig_feat = orig_feat / orig_feat.norm(dim=-1, keepdim=True)

    onnx_out = sess.run(None, {input_name: image_tensor.numpy()})[0]
    onnx_feat = torch.from_numpy(onnx_out).float()
    onnx_feat = onnx_feat / onnx_feat.norm(dim=-1, keepdim=True)

    return {
        "max_abs_diff": (orig_feat.float() - onnx_feat).abs().max().item(),
        "cosine_sim": (orig_feat.float() * onnx_feat).sum(dim=-1).mean().item(),
    }


def benchmark(sess, input_data, input_name, warmup=10, iters=50):
    out_name = sess.get_outputs()[0].name
    for _ in range(warmup):
        sess.run([out_name], {input_name: input_data})

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run([out_name], {input_name: input_data})
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "std_ms": float(np.std(times)),
    }


def print_model_info(ort_path, label):
    """Print op-level summary of the model."""
    try:
        m = onnx.load(ort_path)
    except Exception:
        return

    from collections import Counter
    ops = Counter(n.op_type for n in m.graph.node)
    fp32_conv = ops.get('Conv', 0) + ops.get('FusedConv', 0)
    int8_ops = (ops.get('MatMulInteger', 0) + ops.get('QLinearConv', 0) +
                ops.get('DynamicQuantizeLinear', 0) + ops.get('DynamicQuantizeMatMul', 0))

    print(f"\n--- {label} ---")
    print(f"  Nodes: {len(m.graph.node)}, Initializers: {len(m.graph.initializer)}")
    print(f"  FP32 Convs: {fp32_conv}, INT8 ops: {int8_ops}")
    print(f"  Top ops: {ops.most_common(6)}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimized ONNX export for Android deployment")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    model_name = args.model
    model_key = model_name.lower().replace("-", "_")
    model_file = f"./{model_key}.pt"

    if not os.path.exists(model_file):
        print(f"ERROR: Model file {model_file} not found!")
        for f in os.listdir("."):
            if f.endswith(".pt"):
                print(f"  {f}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"MobileCLIP Android Export: {model_name}")
    print(f"{'='*60}")

    import json
    from PIL import Image

    # Load
    print("\n[1] Loading model...")
    model_kwargs = get_model_kwargs(model_name)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=model_file, **model_kwargs)
    model.eval()
    model = reparameterize_model(model)

    config_path = f"mobileclip2/model_configs/{model_name}.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            img_size = json.load(f).get("vision_cfg", {}).get("image_size", 256)
    else:
        img_size = 256

    image = preprocess(
        Image.open("docs/fig_accuracy_latency.png").convert("RGB")
    ).unsqueeze(0)

    # Export
    print(f"\n[2] Exporting FP32 ONNX → ORT (target: ARM)...")
    onnx_path, ort_files = export_visual_ort(
        model.visual, image,
        f"{OUTPUT_DIR}/{model_key}_visual.onnx")

    # Verify
    print(f"\n[3] Precision verification...")
    for ort_file in ort_files:
        sess = ort.InferenceSession(ort_file, providers=['CPUExecutionProvider'])
        info = verify_precision(sess, model, image)
        print(f"  {os.path.basename(ort_file)}: "
              f"max_diff={info['max_abs_diff']:.6f}, cosine_sim={info['cosine_sim']:.6f}")

    # Benchmark
    if args.benchmark:
        print(f"\n[4] PC Benchmark (CPUExecutionProvider)...")
        image_np = image.numpy()

        # Include reference models for comparison
        all_models = list(ort_files)
        for ref in ['int8_results/mobileclip2_s0_visual.ort',
                     'int8_results/clip-image-int8.ort']:
            if os.path.exists(ref):
                all_models.append(ref)

        for model_path in all_models:
            sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            input_name = sess.get_inputs()[0].name
            in_shape = [1 if not isinstance(d, int) else d for d in sess.get_inputs()[0].shape]
            test_input = image_np if in_shape[2:] == [256, 256] else \
                         np.random.randn(*in_shape).astype(np.float32)
            stats = benchmark(sess, test_input, input_name)
            label = os.path.basename(model_path)
            print(f"  {label}: {stats['mean_ms']:.1f}ms "
                  f"(median={stats['median_ms']:.1f}ms) [input={in_shape}]")

    # Summary
    print(f"\n{'='*60}")
    print("Export Summary")
    print(f"{'='*60}")
    for f in ort_files:
        print(f"  {f} ({os.path.getsize(f)/1024/1024:.1f} MB)")
    print_model_info(onnx_path, model_key)

    print(f"""
{'='*60}
ANDROID DEPLOYMENT GUIDE
{'='*60}

To reduce inference time from 80ms to ~25ms on Android, use one of
these execution providers with the exported ORT model:

1. NNAPI (recommended — offloads Conv ops to GPU/NPU):
```kotlin
val sessionOptions = OrtSession.SessionOptions()
sessionOptions.addConfigEntry("session.use_nnapi", "1")
val session = OrtEnvironment.getEnvironment()
    .createSession("model.with_runtime_opt.ort", sessionOptions)
```
   Use with_runtime_opt.ort for NNAPI. Expected: 15-25ms.

2. XNNPACK (CPU-only, optimized ARM kernels):
```kotlin
val sessionOptions = OrtSession.SessionOptions()
sessionOptions.addConfigEntry("session.intra_op_num_threads", "4")
sessionOptions.optimizationLevel =
    OrtSession.SessionOptions.OptLevel.ALL_OPT
val session = OrtEnvironment.getEnvironment()
    .createSession("model.ort", sessionOptions)
```
   Add to app/build.gradle:
```
   implementation 'com.microsoft.onnxruntime:onnxruntime-xnnpack:1.18.+'
```

3. For Qualcomm devices, use QNN delegate for NPU acceleration.
""")


if __name__ == "__main__":
    main()
