from pathlib import Path
import torch
import torch.nn as nn
import open_clip
import numpy as np
from mobileclip.modules.common.mobileone import reparameterize_model
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process
from onnxruntime.quantization.registry import IntegerOpsRegistry
import onnx
from onnxruntime.tools.optimize_onnx_model import optimize_model
import os
# 对比推理精度
import onnxruntime as ort

# https://huggingface.co/apple/MobileCLIP2-S2/tree/main
model_name = "MobileCLIP2-S2"
model_file = model_name.lower().replace("-", "_")

model_kwargs = {}
if not (
    model_name.endswith("S3")
    or model_name.endswith("S4")
    or model_name.endswith("L-14")
):
    model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained=f"./{model_file}.pt", **model_kwargs
)
tokenizer = open_clip.get_tokenizer(model_name)

model.eval()
model = reparameterize_model(model)

import PIL.Image
from PIL import Image

# image = PIL.Image.open("docs/fig_accuracy_latency.png").convert("RGB")
image = preprocess(
    Image.open("docs/fig_accuracy_latency.png").convert("RGB")
).unsqueeze(0)
text = tokenizer(["a diagram", "a paper essay", "a cat"])

visual_model = model.visual
text_model = model.text

with torch.no_grad():
    image_features_orig = model.encode_image(image)
    text_features_orig = model.encode_text(text)
    image_features_orig /= image_features_orig.norm(dim=-1, keepdim=True)
    text_features_orig /= text_features_orig.norm(dim=-1, keepdim=True)
    text_probs_orig = (100.0 * image_features_orig @ text_features_orig.T).softmax(dim=-1)

print("Original model text probs:", text_probs_orig)


def export_model_to_onnx(export_model, dummy_input, onnx_path, input_names, output_names, dynamic_axes):
    export_model.eval()
    torch.onnx.export(
        export_model,
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=20,
        dynamo=False,
    )
    print(f"Exported FP32 ONNX to {onnx_path}")


# 第一步：导出 FP32 ONNX 模型（作为量化的基础）
visual_fp32_path = f"./{model_file}_visual_fp32_for_quant.onnx"
text_fp32_path = f"./{model_file}_text_fp32_for_quant.onnx"

export_model_to_onnx(
    visual_model,
    (image,),
    visual_fp32_path,
    input_names=["image"],
    output_names=["image_features"],
    dynamic_axes={"image": {0: "batch_size"}, "image_features": {0: "batch_size"}},
)

export_model_to_onnx(
    text_model,
    (text,),
    text_fp32_path,
    input_names=["text"],
    output_names=["text_features"],
    dynamic_axes={"text": {0: "batch_size"}, "text_features": {0: "batch_size"}},
)

# 第二步：Pre-process FP32 ONNX (shape inference + optimization)

# visual_fp32_optimized_path = f"./{model_file}_visual_fp32_optimized.onnx"
# text_fp32_optimized_path = f"./{model_file}_text_fp32_optimized.onnx"

# Optimization
optimize_model(Path(visual_fp32_path), Path(visual_fp32_path))
optimize_model(Path(text_fp32_path), Path(text_fp32_path))

# Shape inference
onnx.shape_inference.infer_shapes_path(visual_fp32_path, visual_fp32_path)
onnx.shape_inference.infer_shapes_path(text_fp32_path, text_fp32_path)

print(f"Pre-processed FP32 ONNX (shape inference + optimization) saved to {visual_fp32_path} and {text_fp32_path}")

# 第三步：使用 ONNX Runtime 进行动态量化，生成真正的 INT8 模型

# 预处理
quant_pre_process(visual_fp32_path, visual_fp32_path)
quant_pre_process(text_fp32_path, text_fp32_path)

visual_int8_path = f"./int8_results/{model_file}_visual.onnx"
text_int8_path = f"./int8_results/{model_file}_text.onnx"

if not os.path.exists("int8_results"):
    os.makedirs("int8_results")

op_types_to_quantize = IntegerOpsRegistry.copy()
op_types_to_quantize.pop("Conv")

quantize_dynamic(
    model_input=visual_fp32_path,
    model_output=visual_int8_path,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=op_types_to_quantize.keys(),
)

print(f"Exported true INT8 ONNX to {visual_int8_path}")

quantize_dynamic(
    model_input=text_fp32_path, 
    model_output=text_int8_path,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=op_types_to_quantize.keys(),
)

print(f"Exported true INT8 ONNX to {text_int8_path}")

print("\n" + "="*60)
print("模型体积对比:")
print("="*60)

for path in [visual_fp32_path, visual_int8_path, text_fp32_path, text_int8_path]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{path}: {size_mb:.2f} MB")


print("\n" + "="*60)
print("推理精度对比:")
print("="*60)

# FP32 ONNX 推理
sess_visual_fp32 = ort.InferenceSession(visual_fp32_path)
sess_text_fp32 = ort.InferenceSession(text_fp32_path)

image_np = image.numpy()
text_np = text.numpy()

image_features_fp32_onnx = sess_visual_fp32.run(None, {"image": image_np})[0]
text_features_fp32_onnx = sess_text_fp32.run(None, {"text": text_np})[0]

image_features_fp32_onnx = torch.from_numpy(image_features_fp32_onnx)
text_features_fp32_onnx = torch.from_numpy(text_features_fp32_onnx)

image_features_fp32_onnx /= image_features_fp32_onnx.norm(dim=-1, keepdim=True)
text_features_fp32_onnx /= text_features_fp32_onnx.norm(dim=-1, keepdim=True)
text_probs_fp32_onnx = (100.0 * image_features_fp32_onnx @ text_features_fp32_onnx.T).softmax(dim=-1)

# INT8 ONNX 推理
sess_visual_int8 = ort.InferenceSession(visual_int8_path)
sess_text_int8 = ort.InferenceSession(text_int8_path)

image_features_int8_onnx = sess_visual_int8.run(None, {"image": image_np})[0]
text_features_int8_onnx = sess_text_int8.run(None, {"text": text_np})[0]

image_features_int8_onnx = torch.from_numpy(image_features_int8_onnx)
text_features_int8_onnx = torch.from_numpy(text_features_int8_onnx)

image_features_int8_onnx /= image_features_int8_onnx.norm(dim=-1, keepdim=True)
text_features_int8_onnx /= text_features_int8_onnx.norm(dim=-1, keepdim=True)
text_probs_int8_onnx = (100.0 * image_features_int8_onnx @ text_features_int8_onnx.T).softmax(dim=-1)

print("PyTorch FP32 text probs:", text_probs_orig)
print("ONNX FP32 text probs:", text_probs_fp32_onnx)
print("ONNX INT8 text probs:", text_probs_int8_onnx)
print("Max diff (PyTorch FP32 vs ONNX FP32): %.5f" % (text_probs_orig - text_probs_fp32_onnx).abs().max().item())
print("Max diff (ONNX FP32 vs ONNX INT8): %.5f" % (text_probs_fp32_onnx - text_probs_int8_onnx).abs().max().item())
print("Max diff (PyTorch FP32 vs ONNX INT8): %.5f" % (text_probs_orig - text_probs_int8_onnx).abs().max().item())

# 清理临时文件
os.remove(visual_fp32_path)
os.remove(text_fp32_path)
print("\n临时 FP32 文件已清理")
