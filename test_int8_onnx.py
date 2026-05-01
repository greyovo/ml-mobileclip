import sys
sys.path.insert(0, "open_clip/src")

import numpy as np
import onnxruntime as ort
from PIL import Image
import open_clip
import torch

model_name = "MobileCLIP2-S2"
model_file = model_name.lower().replace("-", "_")

# 加载 tokenizer 和 preprocess（用于准备输入）
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

image_path = "docs/fig_accuracy_latency.png"
text = "a diagram"

# 准备输入
image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
text_tokens = tokenizer(text)

print("Image input shape:", image.shape, "dtype:", image.dtype)
print("Text input shape:", text_tokens.shape, "dtype:", text_tokens.dtype)

# Image encoder int8
sess_image = ort.InferenceSession(f"./{model_file}_visual_int8.onnx")
input_name_image = sess_image.get_inputs()[0].name
print("Image model input name:", input_name_image, "shape:", sess_image.get_inputs()[0].shape)

image_np = image.numpy()
out_image = sess_image.run(None, {input_name_image: image_np})
print("Image output shape:", out_image[0].shape)

# Text encoder int8
sess_text = ort.InferenceSession(f"./{model_file}_text_int8.onnx")
input_name_text = sess_text.get_inputs()[0].name
print("Text model input name:", input_name_text, "shape:", sess_text.get_inputs()[0].shape, "type:", sess_text.get_inputs()[0].type)

text_np = text_tokens.numpy()
# int64 -> int32 if needed
if sess_text.get_inputs()[0].type == "tensor(int32)":
    text_np = text_np.astype(np.int32)
out_text = sess_text.run(None, {input_name_text: text_np})
print("Text output shape:", out_text[0].shape)

# L2 normalize
image_feat = out_image[0]
text_feat = out_text[0]
image_feat = image_feat / np.linalg.norm(image_feat, axis=-1, keepdims=True)
text_feat = text_feat / np.linalg.norm(text_feat, axis=-1, keepdims=True)

similarity = (image_feat @ text_feat.T) * 100.0
print("ONNX int8 similarity:", similarity)

# 对比原始 PyTorch 输出
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("PyTorch similarity:", (100.0 * image_features @ text_features.T).numpy())
print("PyTorch text_probs:", text_probs.numpy())

# 计算 ONNX 与 PyTorch 的 cosine 相似度差异
onnx_cos = similarity / 100.0
pt_cos = (image_features @ text_features.T).numpy()
print("Cosine diff (ONNX - PyTorch):", onnx_cos - pt_cos)

# 测试 fp32 ONNX 输出
print("\n--- FP32 ONNX ---")
sess_image_fp32 = ort.InferenceSession(f"./{model_file}_visual.onnx")
sess_text_fp32 = ort.InferenceSession(f"./{model_file}_text.onnx")
out_image_fp32 = sess_image_fp32.run(None, {sess_image_fp32.get_inputs()[0].name: image_np})
text_np_fp32 = text_tokens.numpy()
if sess_text_fp32.get_inputs()[0].type == "tensor(int32)":
    text_np_fp32 = text_np_fp32.astype(np.int32)
out_text_fp32 = sess_text_fp32.run(None, {sess_text_fp32.get_inputs()[0].name: text_np_fp32})

image_feat_fp32 = out_image_fp32[0]
text_feat_fp32 = out_text_fp32[0]
image_feat_fp32 = image_feat_fp32 / np.linalg.norm(image_feat_fp32, axis=-1, keepdims=True)
text_feat_fp32 = text_feat_fp32 / np.linalg.norm(text_feat_fp32, axis=-1, keepdims=True)
sim_fp32 = (image_feat_fp32 @ text_feat_fp32.T) * 100.0
print("ONNX fp32 similarity:", sim_fp32)
print("Cosine diff (FP32 ONNX - PyTorch):", sim_fp32/100.0 - pt_cos)
