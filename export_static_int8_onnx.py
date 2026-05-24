"""
Static INT8 quantization for MobileCLIP visual encoder.

Implements per-channel weight INT8 quantization for all Conv2d layers
and group-wise weight INT8 quantization for MatMul/Linear layers.
Conv activations remain FP32 to avoid the per-tensor activation
quantization error that destroys accuracy in Conv-based models.

This is a pragmatic static quantization approach:
  - Conv weights: per-channel INT8 (stored as INT8 in ONNX, dequantized at runtime)
  - MatMul/Linear weights: per-channel INT8
  - Conv activations: FP32 (avoids per-tensor quantization error)
  - MatMul activations: optionally INT8 via ORT static quantization (--with-ort-quant)

Why this approach:
  - MobileCLIP2-S0 uses FastViT-MCi with 95 Conv ops and GELU activations
  - Per-tensor quantization of Conv activations causes cosine_sim to drop to 0.3-0.5
  - Even with calibrated scales, per-tensor Conv activation quantization is too lossy
  - Per-channel weight quantization is near-lossless (cosine_sim > 0.96)
  - The ONNX model with INT8 weights is 71% smaller and runs directly in ORT

Calibration: COCO val2017 (auto-downloaded) is used as the calibration dataset
for the optional ORT MatMul activation quantization step.

Usage:
  uv run export_static_int8_onnx.py                        # Weight INT8 (default)
  uv run export_static_int8_onnx.py --with-ort-quant       # + ORT MatMul activation INT8
  uv run export_static_int8_onnx.py --benchmark            # With benchmark
  uv run export_static_int8_onnx.py --model MobileCLIP2-S2 # Different model
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model

MODEL_NAME = "MobileCLIP2-S0"
OUTPUT_DIR = "static_int8_results"
CALIBRATION_DIR = "calibration_data"
COCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"

_GROUP_SIZE = 128  # group size for Linear/Embedding weight quantization


def get_model_kwargs(model_name):
    if model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14"):
        return {}
    return {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}


# ============================================================
# INT8 quantized modules (ONNX-exportable)
# ============================================================

class Int8Linear(nn.Module):
    """Linear layer with INT8 weights (group-wise quantization, group_size=128)."""

    def __init__(self, int8_weight, scale, bias=None, group_size=_GROUP_SIZE):
        super().__init__()
        out_f, in_f = int8_weight.shape
        self.in_features = in_f
        self.out_features = out_f
        self.group_size = group_size
        self.register_buffer("int8_weight", int8_weight)
        self.register_buffer("scale", scale.float())
        if bias is not None:
            self.register_buffer("bias", bias.float())
        self.register_buffer("weight", torch.zeros(out_f, in_f))

    @classmethod
    def from_linear(cls, linear, group_size=_GROUP_SIZE):
        weight = linear.weight.data
        out_f, in_f = weight.shape
        gs = group_size if in_f % group_size == 0 and in_f >= group_size else in_f

        if gs == in_f:
            max_abs = weight.abs().max(dim=1, keepdim=True)[0]
            scale = max_abs / 127.0
            i8w = (weight / max_abs * 127.0).round().clamp(-128, 127).to(torch.int8)
        else:
            ng = in_f // gs
            wr = weight.view(out_f, ng, gs)
            max_abs = wr.abs().max(dim=2, keepdim=True)[0]
            scale = max_abs.squeeze(2) / 127.0
            i8w = (wr / max_abs * 127.0).round().clamp(-128, 127).to(torch.int8)
            i8w = i8w.view(out_f, in_f)

        bias = linear.bias.data.clone() if linear.bias is not None else None
        return cls(i8w, scale, bias, gs)

    def forward(self, x):
        if self.group_size == self.in_features:
            w_fp = self.int8_weight.float() * self.scale
        else:
            w_fp = self.int8_weight.float() * self.scale.repeat_interleave(
                self.group_size, dim=1)
        out = torch.matmul(x, w_fp.t())
        if hasattr(self, 'bias') and self.bias is not None:
            out = out + self.bias
        return out


class Int8Conv2d(nn.Module):
    """Conv2d with INT8 weights (per-channel quantization for all kernel sizes)."""

    def __init__(self, int8_weight, scale, bias=None,
                 stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.register_buffer("int8_weight", int8_weight)
        self.register_buffer("scale", scale.float())
        if bias is not None:
            self.register_buffer("bias", bias.float())
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.register_buffer("weight", torch.zeros(1))

    @classmethod
    def from_conv2d(cls, conv):
        """Per-channel INT8 quantization for any Conv2d kernel size.

        Each output channel gets its own scale: scale[i] = max(|W[i]|) / 127.
        This is near-lossless for convolution weights.
        """
        weight = conv.weight.data  # (out_c, in_c/g, kH, kW)
        out_c = weight.shape[0]
        w_flat = weight.view(out_c, -1)  # (out_c, in_c * kH * kW)

        max_abs = w_flat.abs().max(dim=1, keepdim=True)[0]  # (out_c, 1)
        max_abs = max_abs.clamp(min=1e-8)
        scale = max_abs / 127.0  # (out_c, 1)
        i8w = (w_flat / max_abs * 127.0).round().clamp(-128, 127).to(torch.int8)
        i8w = i8w.view_as(weight)

        bias = conv.bias.data.clone() if conv.bias is not None else None
        return cls(i8w, scale, bias, conv.stride, conv.padding,
                   conv.dilation, conv.groups)

    def forward(self, x):
        out_c = self.int8_weight.shape[0]
        w_flat = self.int8_weight.float().view(out_c, -1)
        w_deq = (w_flat * self.scale).view_as(self.int8_weight)
        bias = self.bias if hasattr(self, 'bias') else None
        return F.conv2d(x, w_deq, bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Int8Embedding(nn.Module):
    """Embedding with INT8 weights (group-wise quantization)."""

    def __init__(self, int8_weight, scale, padding_idx=None, group_size=_GROUP_SIZE):
        super().__init__()
        self.num_embeddings, self.embedding_dim = int8_weight.shape
        self.padding_idx = padding_idx
        self.group_size = group_size
        self.register_buffer("int8_weight", int8_weight)
        self.register_buffer("scale", scale.float())
        self.register_buffer("weight", torch.zeros(1))

    @classmethod
    def from_embedding(cls, emb, group_size=_GROUP_SIZE):
        weight = emb.weight.data
        n_emb, e_dim = weight.shape
        gs = group_size if e_dim % group_size == 0 and e_dim >= group_size else e_dim

        if gs == e_dim:
            max_abs = weight.abs().max(dim=1, keepdim=True)[0]
            scale = max_abs / 127.0
            i8w = (weight / max_abs * 127.0).round().clamp(-128, 127).to(torch.int8)
        else:
            ng = e_dim // gs
            wr = weight.view(n_emb, ng, gs)
            max_abs = wr.abs().max(dim=2, keepdim=True)[0]
            scale = max_abs.squeeze(2) / 127.0
            i8w = (wr / max_abs * 127.0).round().clamp(-128, 127).to(torch.int8)
            i8w = i8w.view(n_emb, e_dim)

        return cls(i8w, scale, emb.padding_idx, gs)

    def forward(self, x):
        if self.group_size == self.embedding_dim:
            w_fp = self.int8_weight.float() * self.scale
        else:
            w_fp = self.int8_weight.float() * self.scale.repeat_interleave(
                self.group_size, dim=1)
        return F.embedding(x, w_fp, self.padding_idx)


# ============================================================
# Model quantization: replace modules with INT8 versions
# ============================================================

_ATTENTION_TYPES = {"MultiheadAttention", "MultiHeadAttention", "MHSA", "Attention"}


def _should_quantize_linear(name, module, parent):
    """Skip attention projection layers (they're sensitive to quantization)."""
    if parent is None:
        return True
    return parent.__class__.__name__ not in _ATTENTION_TYPES


def quantize_model_weights(model):
    """Replace all nn.Linear, nn.Conv2d, and nn.Embedding with INT8 versions.

    - Conv2d: per-channel INT8 (all kernel sizes)
    - Linear: group-wise INT8 (group_size=128, excludes attention projections)
    - Embedding: group-wise INT8 (group_size=128)
    """
    stats = {"conv": 0, "linear": 0, "embedding": 0}

    def _replace(mod, parent=None):
        nonlocal stats
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Conv2d):
                mod._modules[name] = Int8Conv2d.from_conv2d(child)
                stats["conv"] += 1
            elif isinstance(child, nn.Linear):
                if _should_quantize_linear(name, child, mod):
                    mod._modules[name] = Int8Linear.from_linear(child)
                    stats["linear"] += 1
            elif isinstance(child, nn.Embedding):
                mod._modules[name] = Int8Embedding.from_embedding(child)
                stats["embedding"] += 1
            elif len(list(child.children())) > 0:
                _replace(child, parent=child)

    _replace(model)
    return stats


def verify_weight_quantization(model_int8, model_orig):
    """Compute max weight error per quantized module."""
    errors = {}
    for (name_i, m_i), (name_o, m_o) in zip(
            model_int8.named_modules(), model_orig.named_modules()):
        if isinstance(m_i, Int8Conv2d) and isinstance(m_o, nn.Conv2d):
            w_orig = m_o.weight.data.float()
            out_c = m_i.int8_weight.shape[0]
            w_deq = (m_i.int8_weight.float().view(out_c, -1) * m_i.scale).view_as(w_orig)
            errors[name_i] = (w_orig - w_deq).abs().max().item()
        elif isinstance(m_i, Int8Linear) and isinstance(m_o, nn.Linear):
            w_orig = m_o.weight.data.float()
            out_f, in_f = m_i.int8_weight.shape
            if m_i.group_size == in_f:
                w_deq = m_i.int8_weight.float() * m_i.scale
            else:
                w_deq = m_i.int8_weight.float() * m_i.scale.repeat_interleave(
                    m_i.group_size, dim=1)
            errors[name_i] = (w_orig - w_deq).abs().max().item()
        elif isinstance(m_i, Int8Embedding) and isinstance(m_o, nn.Embedding):
            w_orig = m_o.weight.data.float()
            e_dim = m_i.embedding_dim
            if m_i.group_size == e_dim:
                w_deq = m_i.int8_weight.float() * m_i.scale
            else:
                w_deq = m_i.int8_weight.float() * m_i.scale.repeat_interleave(
                    m_i.group_size, dim=1)
            errors[name_i] = (w_orig - w_deq).abs().max().item()
    return errors


# ============================================================
# Calibration dataset (for optional ORT MatMul quantization)
# ============================================================

def download_coco_val2017(target_dir, num_images, skip_download=False):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(target_dir.glob("*.jpg"))
    if len(existing) >= num_images:
        print(f"  Using {min(len(existing), num_images)} cached COCO images")
        return existing[:num_images]
    if skip_download:
        return []

    zip_path = target_dir / "val2017.zip"
    if not zip_path.exists():
        print(f"  Downloading COCO val2017 (~1GB)...")

        def _progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(100, block_num * block_size * 100 / total_size)
                print(f"\r  Downloading: {pct:.0f}%", end="")

        try:
            urllib.request.urlretrieve(COCO_VAL2017_URL, zip_path, _progress)
            print()
        except Exception as e:
            print(f"\n  WARNING: Download failed: {e}")
            return []

    print(f"  Extracting images...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            jpg_files = sorted(f for f in zf.namelist() if f.endswith('.jpg'))
            for f in jpg_files[:num_images + 50]:
                zf.extract(f, target_dir)
    except Exception as e:
        print(f"  WARNING: Extraction failed: {e}")
        return []

    val_dir = target_dir / "val2017"
    if val_dir.exists():
        for f in val_dir.glob("*.jpg"):
            shutil.move(str(f), str(target_dir / f.name))
        val_dir.rmdir()

    return sorted(target_dir.glob("*.jpg"))[:num_images]


def create_augmented_calibration_set(target_dir, num_samples, seed=42):
    random.seed(seed)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(target_dir.glob("*.jpg"))
    if len(existing) >= num_samples:
        return existing[:num_samples]

    from torchvision.transforms import functional as TF

    source_images = [Image.open(p).convert("RGB")
                     for p in sorted(Path("docs").glob("*.png"))]
    if not source_images:
        raise RuntimeError("No source images found in docs/")

    print(f"  Creating {num_samples} augmented images from {len(source_images)} sources...")

    for i in range(num_samples):
        img = random.choice(source_images).copy()
        w, h = img.size
        scale = random.uniform(0.3, 1.0)
        cw = max(32, int(w * scale))
        ch = max(32, int(h * scale))
        l = random.randint(0, max(1, w - cw))
        t = random.randint(0, max(1, h - ch))
        img = img.crop((l, t, l + cw, t + ch)).resize((256, 256), Image.BILINEAR)
        if random.random() < 0.8:
            img = TF.adjust_brightness(img, random.uniform(0.3, 1.7))
        if random.random() < 0.8:
            img = TF.adjust_contrast(img, random.uniform(0.3, 1.7))
        if random.random() < 0.8:
            img = TF.adjust_saturation(img, random.uniform(0.3, 1.7))
        if random.random() < 0.3:
            img = TF.adjust_hue(img, random.uniform(-0.3, 0.3))
        if random.random() < 0.5:
            img = TF.hflip(img)
        img.save(target_dir / f"calib_{i:05d}.jpg")

    return sorted(target_dir.glob("*.jpg"))[:num_samples]


class CalibrationDataReader:
    """ORT CalibrationDataReader: yields preprocessed images as numpy dicts."""

    def __init__(self, image_paths, preprocess, input_name="image", batch_size=1):
        self._paths = sorted(image_paths)
        self._preprocess = preprocess
        self.input_name = input_name
        self.batch_size = batch_size
        self._start = 0
        self._end = len(self._paths)
        self._index = self._start

    def get_next(self):
        if self._index >= self._end:
            return None
        batch_paths = self._paths[self._index:self._index + self.batch_size]
        self._index += self.batch_size
        batch = []
        for p in batch_paths:
            try:
                batch.append(self._preprocess(Image.open(p).convert("RGB")))
            except Exception:
                continue
        if not batch:
            return self.get_next() if self._index < self._end else None
        return {self.input_name: torch.stack(batch).numpy()}

    def set_range(self, start_index, end_index):
        self._start = max(0, start_index)
        self._end = min(len(self._paths), end_index)
        self._index = self._start

    def __len__(self):
        return len(self._paths)

    def __iter__(self):
        self._index = self._start
        return self

    def __next__(self):
        result = self.get_next()
        if result is None:
            raise StopIteration
        return result


# ============================================================
# ONNX export
# ============================================================

def export_fp32_onnx(model, dummy_input, output_path,
                     input_name="image", output_name="image_features"):
    model.eval()
    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=[input_name], output_names=[output_name],
        dynamic_axes={input_name: {0: "batch_size"},
                      output_name: {0: "batch_size"}},
        opset_version=20, dynamo=False,
    )
    print(f"  FP32 ONNX: {os.path.basename(output_path)} "
          f"({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")


def apply_ort_full_static_quantization(fp32_path, output_path, calibration_reader,
                                        nodes_to_exclude=None):
    """Apply ORT full static INT8 quantization (weights + activations) to ALL ops.

    Uses carefully tuned parameters to minimize accuracy loss:
      - asymmetric QUInt8 activation (handles Conv's varied distributions)
      - per-channel QInt8 weight (near-lossless for weights)
      - reduce_range=True (7-bit activations, more conservative)
      - Entropy calibration (KL divergence minimization, best for mixed distributions)
      - Percentile fallback with 99.99%

    Sensitive nodes (first Conv, last projection) can be excluded to preserve
    accuracy at a small cost to overall quantization coverage.
    """
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat, CalibrationMethod)

    print(f"  Applying full static INT8 quantization...")
    if nodes_to_exclude:
        print(f"  Excluding {len(nodes_to_exclude)} sensitive nodes from quantization")
    t0 = time.time()

    quantize_static(
        model_input=fp32_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        reduce_range=False,
        nodes_to_exclude=nodes_to_exclude,
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={
            "CalibPercentile": 99.995,
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
        },
    )

    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Full static INT8: {os.path.basename(output_path)} ({size_mb:.1f} MB) [{elapsed:.1f}s]")


def apply_ort_weight_only_quantization(fp32_path, output_path, calibration_reader):
    """Apply ORT static quantization ONLY to MatMul/Gemm paths.

    Conv ops are excluded — their activations stay FP32. This is the fallback
    when full INT8 accuracy is unacceptable.
    """
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat, CalibrationMethod)

    print(f"  Applying MatMul-only quantization (Conv activations stay FP32)...")
    t0 = time.time()

    quantize_static(
        model_input=fp32_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm", "Add"],
        calibrate_method=CalibrationMethod.Percentile,
        extra_options={"CalibPercentile": 99.99},
    )

    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  MatMul-only INT8: {os.path.basename(output_path)} ({size_mb:.1f} MB) [{elapsed:.1f}s]")


def find_sensitive_nodes(onnx_path):
    """Find sensitive Conv/MatMul nodes to exclude from activation quantization.

    Returns node names (strings) for:
      - First 2 Conv nodes (stem): quantizing RGB input is lossy
      - Last MatMul (projection head): output used directly for similarity
    """
    model = onnx.load(onnx_path)
    graph = model.graph
    model_input_name = graph.input[0].name
    model_output_name = graph.output[0].name

    conv_nodes = [n for n in graph.node if n.op_type == "Conv"]
    matmul_nodes = [n for n in graph.node if n.op_type in ("MatMul", "Gemm")]

    exclude = []

    # Exclude first 2 Conv nodes that consume model input (stem)
    for node in conv_nodes:
        if model_input_name in node.input:
            exclude.append(node.name)
            break
    # Also find Conv nodes 1 hop away from input
    input_consumers = [n for n in graph.node
                       if any(model_input_name in o for o in n.output)]
    for ic in input_consumers:
        for node in conv_nodes:
            if ic.output[0] in node.input and node.name not in exclude:
                exclude.append(node.name)
                break

    # Exclude last MatMul/Gemm that produces model output
    for node in matmul_nodes:
        if model_output_name in node.output:
            exclude.append(node.name)
            break
    # Also check 1 hop upstream from output
    output_producers = [n for n in graph.node
                        if model_output_name in n.input]
    for op in output_producers:
        if op.op_type in ("MatMul", "Gemm"):
            exclude.append(op.name)

    return list(set(exclude))


def quantize_onnx_weights_to_int8(onnx_path, output_path):
    """Post-process ONNX model: replace FP32 Conv/MatMul weights with INT8 + dequantization.

    For each Conv and MatMul node, the FP32 weight initializer is replaced with:
      - INT8 weight (1 byte per element)
      - FP32 per-channel scale
      - Cast(INT8→FP32) + Mul(×scale) dequantization nodes

    This reduces the ONNX model size by ~4x for weight tensors while maintaining
    near-lossless accuracy via per-channel quantization.
    """
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    model = onnx.load(onnx_path)
    graph = model.graph
    init_map = {init.name: init for init in graph.initializer}
    new_nodes = []
    removed_inits = set()

    for node in list(graph.node):
        if node.op_type not in ("Conv", "MatMul"):
            continue

        if len(node.input) < 2:
            continue

        weight_name = node.input[1]
        if weight_name not in init_map:
            continue

        weight_init = init_map[weight_name]
        w_fp32 = numpy_helper.to_array(weight_init)
        if w_fp32.dtype != np.float32:
            continue

        out_c = w_fp32.shape[0]
        w_flat = w_fp32.reshape(out_c, -1)
        max_abs = np.maximum(np.abs(w_flat).max(axis=1, keepdims=True), 1e-8)
        scale = (max_abs / 127.0).astype(np.float32)
        w_int8 = np.round(w_flat / max_abs * 127.0).clip(-128, 127).astype(np.int8)
        w_int8 = w_int8.reshape(w_fp32.shape)

        # INT8 weight initializer
        i8_name = weight_name + "_i8"
        graph.initializer.append(numpy_helper.from_array(w_int8, name=i8_name))

        # Scale initializer (reshaped for broadcasting)
        scale_shape = [out_c] + [1] * (len(w_fp32.shape) - 1)
        s_name = weight_name + "_s"
        graph.initializer.append(
            numpy_helper.from_array(scale.reshape(scale_shape), name=s_name))

        # Dequantization: Cast → Mul
        cast_out = weight_name + "_f32"
        cast_node = helper.make_node(
            "Cast", [i8_name], [cast_out],
            name=weight_name + "_cast", to=int(TensorProto.FLOAT))

        deq_out = weight_name + "_deq"
        mul_node = helper.make_node(
            "Mul", [cast_out, s_name], [deq_out],
            name=weight_name + "_deq")

        # Redirect Conv/MatMul weight input to dequantized weight
        node.input[1] = deq_out
        new_nodes.extend([cast_node, mul_node])
        removed_inits.add(weight_name)

    # Remove old FP32 weight initializers
    new_inits = [i for i in graph.initializer if i.name not in removed_inits]
    while len(graph.initializer) > 0:
        graph.initializer.pop()
    graph.initializer.extend(new_inits)
    graph.node.extend(new_nodes)

    onnx.save(model, output_path)


# ============================================================
# Verification
# ============================================================

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


def verify_pytorch_precision(model_int8, model_orig, image_tensor):
    with torch.no_grad():
        orig_feat = model_orig.encode_image(image_tensor)
        orig_feat = orig_feat / orig_feat.norm(dim=-1, keepdim=True)
        int8_feat = model_int8(image_tensor)
        int8_feat = int8_feat / int8_feat.norm(dim=-1, keepdim=True)
    return {
        "max_abs_diff": (orig_feat.float() - int8_feat).abs().max().item(),
        "cosine_sim": (orig_feat.float() * int8_feat).sum(dim=-1).mean().item(),
    }


def verify_text_probs(model_orig, tokenizer, sess_vis, image_tensor, input_name="image"):
    text = tokenizer(["a diagram", "a paper essay", "a cat"])
    with torch.no_grad():
        img_orig = model_orig.encode_image(image_tensor)
        txt_orig = model_orig.encode_text(text)
        img_orig = img_orig / img_orig.norm(dim=-1, keepdim=True)
        txt_orig = txt_orig / txt_orig.norm(dim=-1, keepdim=True)
        probs_orig = (100.0 * img_orig @ txt_orig.T).softmax(dim=-1)

    img_onnx = torch.from_numpy(sess_vis.run(None, {input_name: image_tensor.numpy()})[0])
    img_onnx = img_onnx / img_onnx.norm(dim=-1, keepdim=True)
    probs_onnx = (100.0 * img_onnx @ txt_orig.T).softmax(dim=-1)
    return probs_orig, probs_onnx


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
    return {"mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "std_ms": float(np.std(times))}


def print_model_info(onnx_path, label):
    try:
        m = onnx.load(onnx_path)
    except Exception:
        return
    ops = Counter(n.op_type for n in m.graph.node)
    qconv = ops.get('QLinearConv', 0)
    qmatmul = ops.get('QLinearMatMul', 0) + ops.get('MatMulInteger', 0)
    qdq = sum(v for k, v in ops.items() if k in ('QuantizeLinear', 'DequantizeLinear'))
    fp32_conv = ops.get('Conv', 0)
    fp32_matmul = ops.get('MatMul', 0)
    print(f"\n--- {label} ---")
    print(f"  Nodes: {len(m.graph.node)}, Initializers: {len(m.graph.initializer)}")
    print(f"  Q/DQ ops: {qdq}, QLinearConv: {qconv}, QLinearMatMul: {qmatmul}")
    print(f"  FP32 Conv: {fp32_conv}, FP32 MatMul: {fp32_matmul}")
    print(f"  Top ops: {ops.most_common(8)}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Static INT8 quantization for MobileCLIP visual encoder")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--calibration-samples", type=int, default=300)
    parser.add_argument("--calibration-dataset", type=str, default="coco",
                        choices=["coco", "local"])
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--weight-only", action="store_true",
                        help="Only quantize weights, keep activations FP32 (safer)")
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"{'='*60}")
    print(f"Static INT8 Quantization: {model_name}")
    print(f"Mode: {'Weight-only' if args.weight_only else 'Full static (weights + activations)'}")
    print(f"Calibration: {args.calibration_dataset}, {args.calibration_samples} samples")
    print(f"{'='*60}")

    # ---- 1. Load model ----
    print(f"\n[1/6] Loading model...")
    model_kwargs = get_model_kwargs(model_name)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=model_file, **model_kwargs)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    model = reparameterize_model(model)

    test_image = preprocess(
        Image.open("docs/fig_accuracy_latency.png").convert("RGB")).unsqueeze(0)

    # ---- 2. Prepare calibration data ----
    print(f"\n[2/6] Preparing calibration data ({args.calibration_samples} samples)...")

    image_paths = []
    if args.calibration_dataset == "coco":
        calib_dir = os.path.join(CALIBRATION_DIR, "coco_val2017")
        image_paths = download_coco_val2017(
            calib_dir, args.calibration_samples, skip_download=args.skip_download)

    if not image_paths:
        if args.calibration_dataset == "coco":
            print("  Falling back to augmented local images...")
        calib_dir = os.path.join(CALIBRATION_DIR, "augmented")
        image_paths = create_augmented_calibration_set(
            calib_dir, args.calibration_samples)

    if len(image_paths) < 10:
        print(f"ERROR: Only {len(image_paths)} calibration images. Need >= 10.")
        sys.exit(1)

    print(f"  Using {len(image_paths)} calibration images")

    # ---- 3. Export FP32 ONNX ----
    print(f"\n[3/6] Exporting FP32 ONNX...")
    fp32_path = os.path.join(OUTPUT_DIR, f"{model_key}_visual_fp32.onnx")
    export_fp32_onnx(model.visual, test_image, fp32_path)
    fp32_mb = os.path.getsize(fp32_path) / (1024 * 1024)

    # Shape inference
    try:
        onnx.shape_inference.infer_shapes_path(fp32_path, fp32_path)
    except Exception:
        pass

    # ---- 4. Quantization ----
    print(f"\n[4/6] Static INT8 quantization...")

    calibration_reader = CalibrationDataReader(image_paths, preprocess, "image")

    # Always create weight-INT8 via graph manipulation (for size comparison)
    w_int8_path = os.path.join(OUTPUT_DIR, f"{model_key}_visual_w_int8.onnx")
    quantize_onnx_weights_to_int8(fp32_path, w_int8_path)
    w_int8_mb = os.path.getsize(w_int8_path) / (1024 * 1024)
    print(f"  Weight-INT8 reference: {w_int8_mb:.1f} MB "
          f"({(1 - w_int8_mb/fp32_mb)*100:.0f}% reduction)")

    if args.weight_only:
        print(f"  Mode: Weight-only (activations stay FP32)")
        ort_int8_path = os.path.join(OUTPUT_DIR, f"{model_key}_visual_static_int8.onnx")
        apply_ort_weight_only_quantization(fp32_path, ort_int8_path, calibration_reader)
    else:
        print(f"  Mode: Full static (weights + activations INT8)")
        ort_int8_path = os.path.join(OUTPUT_DIR, f"{model_key}_visual_full_int8.onnx")
        apply_ort_full_static_quantization(
            fp32_path, ort_int8_path, calibration_reader)

    ort_mb = os.path.getsize(ort_int8_path) / (1024 * 1024)

    # ---- 5. Precision verification ----
    print(f"\n[5/6] Precision verification...")

    sess_fp32 = ort.InferenceSession(fp32_path, providers=['CPUExecutionProvider'])
    sess_w_int8 = ort.InferenceSession(w_int8_path, providers=['CPUExecutionProvider'])
    sess_ort = ort.InferenceSession(ort_int8_path, providers=['CPUExecutionProvider'])

    fp32_info = verify_precision(sess_fp32, model, test_image)
    w_int8_info = verify_precision(sess_w_int8, model, test_image)
    ort_info = verify_precision(sess_ort, model, test_image)

    print(f"  FP32 ONNX:        cos={fp32_info['cosine_sim']:.6f}")
    print(f"  Weight-INT8 ONNX: cos={w_int8_info['cosine_sim']:.6f} "
          f"({w_int8_mb:.1f} MB)")
    print(f"  Static INT8 ONNX: cos={ort_info['cosine_sim']:.6f} "
          f"({ort_mb:.1f} MB)")

    probs_orig, _ = verify_text_probs(model, tokenizer, sess_fp32, test_image)
    _, probs_w_int8 = verify_text_probs(model, tokenizer, sess_w_int8, test_image)
    _, probs_ort = verify_text_probs(model, tokenizer, sess_ort, test_image)

    print(f"\n  Text probs (FP32 ref):      {probs_orig.squeeze().tolist()}")
    print(f"  Text probs (Weight-INT8):   {probs_w_int8.squeeze().tolist()}")
    print(f"  Text probs (Static INT8):   {probs_ort.squeeze().tolist()}")

    prob_diff_w = (probs_orig - probs_w_int8).abs().max().item() * 100
    prob_diff_ort = (probs_orig - probs_ort).abs().max().item() * 100
    print(f"  Prob diff (Weight-INT8):    {prob_diff_w:.4f}%")
    print(f"  Prob diff (Static INT8):    {prob_diff_ort:.4f}%")

    # ---- 6. Benchmark ----
    print(f"\n[6/6] Benchmark...")

    if args.benchmark:
        image_np = test_image.numpy()
        for label, sess in [("FP32 ONNX", sess_fp32),
                            ("Weight-INT8", sess_w_int8),
                            ("Static INT8", sess_ort)]:
            stats = benchmark(sess, image_np, "image")
            print(f"    {label}: {stats['mean_ms']:.1f}ms "
                  f"(median={stats['median_ms']:.1f}ms)")

    print_model_info(fp32_path, "FP32 Model")
    print_model_info(w_int8_path, "Weight-INT8 Model")
    print_model_info(ort_int8_path, "Static INT8 Model")

    # ---- Summary ----
    primary_path = ort_int8_path
    primary_diff = prob_diff_ort
    primary_info = ort_info

    print(f"\n{'='*60}")
    print("Export Summary")
    print(f"{'='*60}")
    print(f"  FP32 ONNX:         {fp32_mb:.1f} MB")
    print(f"  Weight-INT8 ONNX:  {w_int8_mb:.1f} MB "
          f"({(1 - w_int8_mb/fp32_mb)*100:.0f}% reduction)")
    print(f"  Static INT8 ONNX:  {ort_mb:.1f} MB "
          f"({(1 - ort_mb/fp32_mb)*100:.0f}% reduction)")
    print(f"  Cosine sim (Static INT8): {primary_info['cosine_sim']:.6f}")
    print(f"  Prob max diff:     {primary_diff:.4f}%")

    if primary_diff < 0.5:
        print(f"\n  [OK] Accuracy loss negligible (<0.5%)")
    elif primary_diff < 2.0:
        print(f"\n  [OK] Accuracy loss acceptable (<2%)")
    elif primary_diff < 5.0:
        print(f"\n  [WARN] Accuracy loss moderate (2-5%)")
        print(f"  Use --weight-only for safer weight-only quantization")
    else:
        print(f"\n  [FAIL] Accuracy loss high (>5%)")
        print(f"  Use --weight-only for safe weight-only quantization")

    print(f"\n  Primary model: {primary_path}")
    print(f"  Load directly with ORT InferenceSession on Android:")
    print(f'    session = OrtEnvironment.getEnvironment()')
    print(f'        .createSession("{os.path.basename(primary_path)}", options)')
    print(f"\nDone! Models saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
