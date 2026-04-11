import torch
from PIL import Image
import mobileclip
from torchsummary import summary

from mobileclip.clip import CLIP
from mobileclip.text_encoder import TextTransformer

model, _, preprocess = mobileclip.create_model_and_transforms(
    "mobileclip_s0", pretrained="./models/mobileclip_s0.pt"
)
assert isinstance(model, CLIP)

model.eval()
# print(summary(model, (3, 224, 224)))
tokenizer = mobileclip.get_tokenizer("mobileclip_s0")

image = preprocess(
    Image.open("docs/fig_accuracy_latency.png").convert("RGB")
).unsqueeze(0)
text = tokenizer(["A Diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    print("image_features:", image_features)
    print("text_features:", text_features)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
