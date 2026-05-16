import torch
import torch.nn as nn
import litert_torch
import numpy as np
import os
from pathlib import Path
import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model
import PIL.Image
from PIL import Image

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

# 准备测试数据
image = preprocess(
    Image.open("docs/fig_accuracy_latency.png").convert("RGB")
).unsqueeze(0)
text = tokenizer(["a diagram", "a paper essay", "a cat"])

# PyTorch FP32 推理基准
with torch.no_grad():
    image_features_orig = model.encode_image(image)
    text_features_orig = model.encode_text(text)
    image_features_orig /= image_features_orig.norm(dim=-1, keepdim=True)
    text_features_orig /= text_features_orig.norm(dim=-1, keepdim=True)
    text_probs_orig = (100.0 * image_features_orig @ text_features_orig.T).softmax(dim=-1)

print("PyTorch FP32 text probs:", text_probs_orig)

# 创建输出目录
output_dir = Path("litert_results")
output_dir.mkdir(exist_ok=True)


# ============================================================
# Text 模型包装器
# LiteRT 的 tfl.arg_max 不支持 int64 输入/输出，
# 而原始 TextTransformer 使用 text.argmax(dim=-1) 进行 pooling，
# 该操作返回 int64 索引。
# 解决方案：使用 output_tokens 模式跳过 argmax pooling，
# 将 token id 转为 int32 以避免 int64 问题。
# ============================================================
class TextEncoderWrapper(nn.Module):
    """包装 TextTransformer，使输出兼容 LiteRT 转换。"""

    def __init__(self, text_model):
        super().__init__()
        self.text_model = text_model
        self.text_model.output_tokens = True

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        text_int32 = text.to(torch.int32)
        pooled, tokens = self.text_model(text_int32)
        return pooled


text_wrapper = TextEncoderWrapper(model.text)
text_wrapper.eval()


# ============================================================
# 第一步：导出 Visual 模型为 FP32 LiteRT
# ============================================================
visual_path = str(output_dir / f"{model_file}_visual.tflite")

print("\n[1/3] Converting visual model to LiteRT...")

visual_edge_model = litert_torch.convert(model.visual, (image,))
visual_edge_model.export(visual_path)
print(f"Exported visual model to {visual_path}")

# ============================================================
# 第二步：导出 Text 模型为 FP32 LiteRT
# ============================================================
text_path = str(output_dir / f"{model_file}_text.tflite")

print("\n[2/3] Converting text model to LiteRT...")

text_edge_model = litert_torch.convert(text_wrapper, (text.to(torch.int32),))
text_edge_model.export(text_path)
print(f"Exported text model to {text_path}")

# ============================================================
# 第三步：推理精度对比
# ============================================================
print("\n" + "=" * 60)
print("模型体积:")
print("=" * 60)

for label, path in [("Visual", visual_path), ("Text", text_path)]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {label}: {size_mb:.2f} MB")

print("\n" + "=" * 60)
print("推理精度对比:")
print("=" * 60)

# Visual LiteRT 推理
visual_loaded = litert_torch.Model.load(visual_path)
image_features_litert = visual_loaded(image.numpy())
image_features_litert = torch.from_numpy(np.array(image_features_litert)).float()
image_features_litert /= image_features_litert.norm(dim=-1, keepdim=True)

# Text LiteRT 推理
text_loaded = litert_torch.Model.load(text_path)
text_features_litert = text_loaded(text.to(torch.int32).numpy())
text_features_litert = torch.from_numpy(np.array(text_features_litert)).float()
text_features_litert /= text_features_litert.norm(dim=-1, keepdim=True)

# 计算相似度
text_probs_litert = (100.0 * image_features_litert @ text_features_litert.T).softmax(dim=-1)

print("  PyTorch FP32 text probs: ", text_probs_orig)
print("  LiteRT FP32 text probs:  ", text_probs_litert)
print(
    f"  Max diff: {(text_probs_orig - text_probs_litert).abs().max().item():.5f}"
)
