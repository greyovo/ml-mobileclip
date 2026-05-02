import torch
import torch.nn as nn
import open_clip
import numpy as np
from mobileclip.modules.common.mobileone import reparameterize_model
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import os
import glob
import random

from gen_random_sentences import generate_sentence

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

from PIL import Image

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
        opset_version=14,
        dynamo=False,
    )
    print(f"Exported FP32 ONNX to {onnx_path}")


visual_fp32_path = f"./{model_file}_visual_fp32_for_static_quant.onnx"
text_fp32_path = f"./{model_file}_text_fp32_for_static_quant.onnx"

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


image_paths = sorted(glob.glob("dataset/images/*.JPEG"))
print(f"Found {len(image_paths)} images for calibration")


def load_calibration_images(num_samples):
    selected = random.sample(image_paths, min(num_samples, len(image_paths)))
    tensors = []
    for path in selected:
        img = preprocess(Image.open(path).convert("RGB"))
        tensors.append(img)
    return torch.stack(tensors)


def generate_calibration_texts(num_samples):
    sentences = [generate_sentence() for _ in range(num_samples)]
    return tokenizer(sentences)


class ImageCalibrationDataReader(CalibrationDataReader):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.images = load_calibration_images(num_samples).numpy()
        self.idx = 0

    def get_next(self):
        if self.idx < self.images.shape[0]:
            item = {"image": self.images[self.idx:self.idx+1]}
            self.idx += 1
            return item
        return None

    def rewind(self):
        self.idx = 0


class TextCalibrationDataReader(CalibrationDataReader):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.texts = generate_calibration_texts(num_samples).numpy()
        self.idx = 0

    def get_next(self):
        if self.idx < self.texts.shape[0]:
            item = {"text": self.texts[self.idx:self.idx+1]}
            self.idx += 1
            return item
        return None

    def rewind(self):
        self.idx = 0


def run_static_quantization(num_iters):
    print("\n" + "=" * 60)
    print(f"Running static quantization with {num_iters} calibration iterations")
    print("=" * 60)

    visual_int8_path = f"./{model_file}_visual_static_int8_iter{num_iters}.onnx"
    text_int8_path = f"./{model_file}_text_static_int8_iter{num_iters}.onnx"

    image_reader = ImageCalibrationDataReader(num_iters)
    text_reader = TextCalibrationDataReader(num_iters)

    quantize_static(
        model_input=visual_fp32_path,
        model_output=visual_int8_path,
        calibration_data_reader=image_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"Exported static INT8 visual ONNX to {visual_int8_path}")

    quantize_static(
        model_input=text_fp32_path,
        model_output=text_int8_path,
        calibration_data_reader=text_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"Exported static INT8 text ONNX to {text_int8_path}")

    sess_visual_int8 = ort.InferenceSession(visual_int8_path)
    sess_text_int8 = ort.InferenceSession(text_int8_path)

    image_np = image.numpy()
    text_np = text.numpy()

    image_features_int8 = sess_visual_int8.run(None, {"image": image_np})[0]
    text_features_int8 = sess_text_int8.run(None, {"text": text_np})[0]

    image_features_int8 = torch.from_numpy(image_features_int8)
    text_features_int8 = torch.from_numpy(text_features_int8)

    image_features_int8 /= image_features_int8.norm(dim=-1, keepdim=True)
    text_features_int8 /= text_features_int8.norm(dim=-1, keepdim=True)
    text_probs_int8 = (100.0 * image_features_int8 @ text_features_int8.T).softmax(dim=-1)

    max_diff_fp32 = (text_probs_orig - text_probs_int8).abs().max().item()
    print(f"ONNX Static INT8 ({num_iters} iters) text probs:", text_probs_int8)
    print(f"Max diff (PyTorch FP32 vs Static INT8 {num_iters}): %.5f" % max_diff_fp32)

    size_visual = os.path.getsize(visual_int8_path) / (1024 * 1024)
    size_text = os.path.getsize(text_int8_path) / (1024 * 1024)
    print(f"Visual model size: {size_visual:.2f} MB")
    print(f"Text model size: {size_text:.2f} MB")

    return {
        "iters": num_iters,
        "max_diff": max_diff_fp32,
        "text_probs": text_probs_int8,
        "visual_size_mb": size_visual,
        "text_size_mb": size_text,
        "visual_path": visual_int8_path,
        "text_path": text_int8_path,
    }


results = []
for iters in [100, 150, 200]:
    result = run_static_quantization(iters)
    results.append(result)

print("\n" + "=" * 60)
print("Summary: Static Quantization Comparison")
print("=" * 60)
print(f"{'Iters':<10} {'Max Diff':<15} {'Visual MB':<12} {'Text MB':<12}")
for r in results:
    print(f"{r['iters']:<10} {r['max_diff']:<15.5f} {r['visual_size_mb']:<12.2f} {r['text_size_mb']:<12.2f}")

best = min(results, key=lambda x: x["max_diff"])
print(f"\nBest configuration: {best['iters']} iterations with max diff = {best['max_diff']:.5f}")

for r in results:
    if r["iters"] != best["iters"]:
        os.remove(r["visual_path"])
        os.remove(r["text_path"])
        print(f"Removed intermediate files for {r['iters']} iterations")

print(f"\nKept best model files:")
print(f"  {best['visual_path']}")
print(f"  {best['text_path']}")

os.remove(visual_fp32_path)
os.remove(text_fp32_path)
print("\n临时 FP32 文件已清理")
