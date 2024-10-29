import torch
from PIL import Image
import mobileclip
from torchsummary import summary

from mobileclip.clip import CLIP
from mobileclip.text_encoder import TextTransformer
import os
from pathlib import Path

import onnxruntime as ort

from PIL import Image


model_text = "mobileclip-text-encoder.onnx"
# model_text = "text_model.onnx"
model_image = "mobileclip-image-encoder.onnx"

model, _, preprocess = mobileclip.create_model_and_transforms(
    "mobileclip_s0", pretrained="./models/mobileclip_s0.pt"
)

def test_image():
    # Image
    image_path = "docs/diagram-square.png"
    ort_session = ort.InferenceSession(model_image)
    input_name = ort_session.get_inputs()[0].name
    
    pil = Image.open(image_path)
    
    converted_rgb = pil.convert("RGB")
    
    image_input = preprocess(converted_rgb).unsqueeze(0)
    print(image_input.dtype)
    outputs = ort_session.run(None, {input_name: image_input.numpy()})
    print(f"image '{image_path}' feat:", outputs[0])



def tset_text():
    # Text
    text = "A Diagram"
    # sess_opt = ort.SessionOptions()
    # sess_opt.log_severity_level = 0  # Verbose
    ort_session = ort.InferenceSession(model_text)
    tokenizer = mobileclip.get_tokenizer("mobileclip_s0")
    token_input: torch.Tensor = tokenizer(text)
    token_input = token_input.to(torch.int32)
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: token_input.numpy()})
    print(f"text: '{text}' feat:", outputs[0])
    
if __name__ == "__main__":
    test_image()


# with torch.no_grad(), torch.cuda.amp.autocast():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)

#     print("image_features:", image_features)
#     print("text_features:", text_features)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)

#     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# print("Label probs:", text_probs)
