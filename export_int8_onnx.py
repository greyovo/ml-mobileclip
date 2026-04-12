import torch
import torch.nn as nn
import open_clip
from PIL import Image
from mobileclip.modules.common.mobileone import reparameterize_model
from onnxruntime import quantization

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
    f=f"./{model_file}_visual.onnx",
    external_data=False,
    verify=True,
)

# 导出 text 模型
torch.onnx.export(
    text_model,
    (text,),
    f=f"./{model_file}_text.onnx",
    external_data=False,
    verify=True,
)

print("Export ONNX done")


# 量化为 int8

# 预处理
quantization.quant_pre_process(
    f"./{model_file}_visual.onnx",
    f"./{model_file}_visual_int8.onnx",
)

quantization.quant_pre_process(
    f"./{model_file}_text.onnx",
    f"./{model_file}_text_int8.onnx",
)


quantization.quantize_dynamic(
    f"./{model_file}_visual_int8.onnx",
    f"./{model_file}_visual_int8.onnx",
)

quantization.quantize_dynamic(
    f"./{model_file}_text_int8.onnx",
    f"./{model_file}_text_int8.onnx",
)

print("Quantization done")

## with torch.no_grad(), torch.cuda.amp.autocast():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)

#     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# print("Label probs:", text_probs)
