import torch
import torch.nn as nn
import open_clip
import numpy as np
from mobileclip.modules.common.mobileone import reparameterize_model

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

image = torch.randn(1, 3, 256, 256)
text = tokenizer(["a diagram", "a dog", "a cat"])

visual_model = model.visual
text_model = model.text

with torch.no_grad():
    image_features_orig = model.encode_image(image)
    text_features_orig = model.encode_text(text)
    image_features_orig /= image_features_orig.norm(dim=-1, keepdim=True)
    text_features_orig /= text_features_orig.norm(dim=-1, keepdim=True)
    text_probs_orig = (100.0 * image_features_orig @ text_features_orig.T).softmax(dim=-1)

print("Original model text probs:", text_probs_orig)


class ManualInt8Linear(nn.Module):
    """手动实现的 INT8 量化线性层，参考 bnb 的量化方式"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('int8_weight', torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8))
        self.register_buffer('scale', torch.ones(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None
        # 添加 weight 属性以兼容 get_weight_dtype 等检查
        # 使用一个假的 fp32 weight 作为占位符
        self.register_buffer('weight', torch.zeros(out_features, in_features, dtype=torch.float32))

    @classmethod
    def from_linear(cls, linear_module):
        """从现有的 Linear 模块创建 INT8 版本"""
        int8_module = cls(linear_module.in_features, linear_module.out_features, linear_module.bias is not None)
        # 量化权重：按行量化
        weight = linear_module.weight.data  # (out_features, in_features)
        # 计算每行的 scale
        max_abs = weight.abs().max(dim=1, keepdim=True)[0]
        scale = (max_abs / 127.0).squeeze(1)
        # 量化到 int8
        int8_weight = (weight / (max_abs / 127.0)).round().clamp(-128, 127).to(torch.int8)
        int8_module.int8_weight.copy_(int8_weight)
        int8_module.scale.copy_(scale)
        # 更新占位 weight
        int8_module.weight.copy_(weight)
        if linear_module.bias is not None:
            int8_module.bias.copy_(linear_module.bias.data)
        return int8_module

    def forward(self, x):
        # 反量化并计算
        weight_fp = self.int8_weight.float() * self.scale.unsqueeze(1)
        out = torch.matmul(x, weight_fp.t())
        if self.bias is not None:
            out = out + self.bias
        return out


def replace_linear_manual_int8(model, include_modules=None):
    if include_modules is None:
        include_modules = ["c_fc", "c_proj", "qkv_proj", "out_proj", "in_proj"]
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear_manual_int8(module, include_modules)

        if isinstance(module, torch.nn.Linear) and name in include_modules:
            model._modules[name] = ManualInt8Linear.from_linear(module)
    return model


# 使用手动 INT8 量化
visual_model_int8 = replace_linear_manual_int8(visual_model)
text_model_int8 = replace_linear_manual_int8(text_model)

visual_model_int8.eval()
text_model_int8.eval()

with torch.no_grad():
    image_features_int8 = visual_model_int8(image)
    text_features_int8 = text_model_int8(text)
    image_features_int8 /= image_features_int8.norm(dim=-1, keepdim=True)
    text_features_int8 /= text_features_int8.norm(dim=-1, keepdim=True)
    text_probs_int8 = (100.0 * image_features_int8 @ text_features_int8.T).softmax(dim=-1)

print("INT8 model text probs:", text_probs_int8)
print("Max diff in image features:", (image_features_orig - image_features_int8).abs().max().item())
print("Max diff in text features:", (text_features_orig - text_features_int8).abs().max().item())
print("Max diff in text probs:", (text_probs_orig - text_probs_int8).abs().max().item())


def export_model_to_onnx(export_model, dummy_input, onnx_path, input_names, output_names, dynamic_axes):
    export_model.eval()
    torch.onnx.export(
        export_model,
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported ONNX to {onnx_path}")


export_model_to_onnx(
    visual_model_int8,
    (image,),
    f"./{model_file}_visual_int8.onnx",
    input_names=["image"],
    output_names=["image_features"],
    dynamic_axes={"image": {0: "batch_size"}, "image_features": {0: "batch_size"}},
)

export_model_to_onnx(
    text_model_int8,
    (text,),
    f"./{model_file}_text_int8.onnx",
    input_names=["text"],
    output_names=["text_features"],
    dynamic_axes={"text": {0: "batch_size"}, "text_features": {0: "batch_size"}},
)

import onnxruntime as ort

sess_visual = ort.InferenceSession(f"./{model_file}_visual_int8.onnx")
sess_text = ort.InferenceSession(f"./{model_file}_text_int8.onnx")

image_np = image.numpy()
text_np = text.numpy()

image_features_onnx = sess_visual.run(None, {"image": image_np})[0]
text_features_onnx = sess_text.run(None, {"text": text_np})[0]

image_features_onnx = torch.from_numpy(image_features_onnx)
text_features_onnx = torch.from_numpy(text_features_onnx)

image_features_onnx /= image_features_onnx.norm(dim=-1, keepdim=True)
text_features_onnx /= text_features_onnx.norm(dim=-1, keepdim=True)
text_probs_onnx = (100.0 * image_features_onnx @ text_features_onnx.T).softmax(dim=-1)

print("ONNX INT8 model text probs:", text_probs_onnx)
print("Max diff in image features (PyTorch INT8 vs ONNX INT8):", (image_features_int8 - image_features_onnx).abs().max().item())
print("Max diff in text features (PyTorch INT8 vs ONNX INT8):", (text_features_int8 - text_features_onnx).abs().max().item())
print("Max diff in text probs (PyTorch INT8 vs ONNX INT8):", (text_probs_int8 - text_probs_onnx).abs().max().item())
