import torch
from PIL import Image
import open_clip

import os
from pathlib import Path

import onnxruntime as ort

from PIL import Image
model_name = "MobileCLIP2-S2"
model_file = model_name.lower().replace("-", "_")

model_kwargs = {}
if not (
    model_name.endswith("S3")
    or model_name.endswith("S4")
    or model_name.endswith("L-14")
):
    model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

model_text = f"{model_file}_text_int8.onnx"
# model_text = "text_model.onnx"
model_image = f"{model_file}_visual_int8.onnx"

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained=f"./{model_file}.pt", **model_kwargs
)

def test_image():
    # Image
    image_path = "docs/fig_accuracy_latency.png"
    ort_session = ort.InferenceSession(model_image)
    input_name = ort_session.get_inputs()[0].name
    
    image_input = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    print(image_input.dtype)
    outputs = ort_session.run(None, {input_name: image_input.numpy()})
    print(f"image '{image_path}' feat:", len(outputs[0][0]), ", ...ommited")



def tset_text():
    # Text
    text = "A Diagram"
    # sess_opt = ort.SessionOptions()
    # sess_opt.log_severity_level = 0  # Verbose
    ort_session = ort.InferenceSession(model_text)
    tokenizer = open_clip.get_tokenizer(model_name)
    token_input: torch.Tensor = tokenizer(text)
    token_input = token_input.to(torch.int64)
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: token_input.numpy()})
    print(f"text: '{text}' feat:", len(outputs[0][0]), ", ...ommited")
    
if __name__ == "__main__":
    test_image()
    tset_text()


# with torch.no_grad(), torch.cuda.amp.autocast():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)

#     print("image_features:", image_features)
#     print("text_features:", text_features)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)

#     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# print("Label probs:", text_probs)
