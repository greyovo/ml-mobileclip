from typing import Any, Dict, Optional
from mobileclip.modules.common.mobileone import reparameterize_model
import torch
from PIL import Image
import mobileclip
from PIL import Image
from torch import Tensor
import torch as nn
from mobileclip.clip import CLIP
from mobileclip.text_encoder import TextTransformer
from mobileclip.image_encoder import MCi

class CLIP_encode_image(CLIP):
    """Class for encoding images using the image encoder from CLIP."""

    def __init__(self, cfg: Dict, output_dict: bool = False, *args, **kwargs) -> None:
        super().__init__(cfg, output_dict, *args, **kwargs)

    def forward(self, image: Optional[torch.Tensor] = None) -> Any:
        image_embeddings = (
            self.encode_image(image, normalize=True) if image is not None else None
        )
        return image_embeddings

model, _, preprocess = mobileclip.create_model_and_transforms(
    "mobileclip_s0", pretrained="./models/mobileclip_s0.pt"
)
assert isinstance(model, CLIP)

# model.eval()
# print(summary(model, (3, 224, 224)))
tokenizer = mobileclip.get_tokenizer("mobileclip_s0")

text_encoder: TextTransformer = model.text_encoder
image_encoder: MCi = model.image_encoder

image = preprocess(
    Image.open("docs/fig_accuracy_latency.png").convert("RGB")
).unsqueeze(0)
text = tokenizer(["A Diagram", "a dog", "a cat"])


def export_text_encoder():
    text_encoder.eval()
    text = "A Diagram"
    input_tensor: Tensor = tokenizer(text)
    input_tensor = input_tensor.to(torch.int32)
    model_text = "mobileclip-text-encoder.onnx"
    torch.onnx.export(text_encoder, input_tensor, model_text)
    # traced_model = torch.jit.trace(text_encoder, input_tensor)
    # text_encoder.modules
    # replace argmax to argmax32
    # for name, module in text_encoder.named_modules():
    #     try:
    #         sub_module = model.get_submodule(name)
    #         print(f"Module Name: {name}, Sub module Type: {sub_module.__class__.__name__}")
    #     except:
    #         print(f"Module Name: {name}, Module Type: {module.__class__.__name__}")


def export_image_encoder():
    image_encoder.eval()
    image = Image.open("docs/fig_accuracy_latency.png").convert("RGB")
    input_tensor: Tensor = preprocess(image).unsqueeze(0)
    model_image = "mobileclip-image-encoder.onnx"
    torch.save(image_encoder, "mobileclip-image.pth")
    # torch.onnx.export(image_encoder, input_tensor, model_image)


def example():
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        print("image_features:", image_features)
        print("text_features:", text_features)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        print("Label probs:", text_probs)


if __name__ == "__main__":
    export_image_encoder()
    # export_text_encoder()
    # python -m onnxruntime.tools.check_onnx_model_mobile_usability mobileclip-image-encoder.onnx
    # python -m onnxruntime.tools.check_onnx_model_mobile_usability mobileclip-text-encoder.onnx
