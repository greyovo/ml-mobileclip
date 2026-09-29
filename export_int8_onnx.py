import sys
sys.path.insert(0, "third_party/open_clip/src")

import torch
import torch.nn as nn
import open_clip
from PIL import Image
from mobileclip.modules.common.mobileone import reparameterize_model
from onnxruntime import quantization
from onnxruntime.transformers.float16 import convert_float_to_float16
import onnx
from pathlib import Path
import subprocess

model_name = "MobileCLIP2-S0"
model_file = model_name.lower().replace("-", "_")
visual_fp32 = Path(f"./{model_file}_visual_fp32.onnx")
visual_fp16 = Path(f"./{model_file}_visual_fp16.onnx")
visual_output = Path(f"./{model_file}_visual.onnx")
text_fp32 = Path(f"./{model_file}_text_fp32.onnx")
text_preprocessed = Path(f"./{model_file}_text_preprocessed.onnx")
text_output = Path(f"./{model_file}_text.onnx")

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

# Model needs to be in eval mode for inference because of batchnorm layers unlike ViTs
model.eval()

# For inference/model exporting purposes, please reparameterize first
model = reparameterize_model(model)

image = preprocess(
    Image.open("docs/fig_accuracy_latency.png").convert("RGB")
).unsqueeze(0)
text = tokenizer("a diagram")

# 分离 text 和 visual
visual_model = model.visual
text_model: nn.Module = model.text

# 打印模型输入维度
print("Input dim of visual model", image.shape)
print("Input dim of text model", text.shape)

# 导出 visual 模型
torch.onnx.export(
    visual_model,
    (image,),
    f=visual_fp32,
    external_data=False,
    verify=True,
)

# 导出 text 模型
torch.onnx.export(
    text_model,
    (text,),
    f=text_fp32,
    external_data=False,
    verify=True,
)

print("Exported FP32 ONNX models")

# The visual encoder is stored with FP16 weights but FP32 input/output, which
# keeps the Flutter Float32List I/O contract while reducing the asset size.
fp32_visual = onnx.load(visual_fp32)
onnx.save(convert_float_to_float16(fp32_visual, keep_io_types=False), visual_fp16)
subprocess.run(
    [
        "uv",
        "run",
        "python",
        "add_fp32_io_to_fp16_model.py",
        str(visual_fp16),
        str(visual_output),
    ],
    check=True,
)

# Dynamic INT8 is smaller and accurate enough for the text encoder. Its int64
# token input and FP32 embedding output are intentionally retained.
quantization.quant_pre_process(
    text_fp32,
    text_preprocessed,
)
quantization.quantize_dynamic(
    text_preprocessed,
    text_output,
    weight_type=quantization.QuantType.QInt8,
)

print(f"Created {visual_output} and {text_output}")

## with torch.no_grad(), torch.cuda.amp.autocast():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)

#     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# print("Label probs:", text_probs)
