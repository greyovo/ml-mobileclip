import sys
sys.path.insert(0, "open_clip/src")

import torch
import torch.nn as nn
import open_clip
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from mobileclip.modules.common.mobileone import reparameterize_model
from onnxruntime import quantization
import os
import numpy as np
import random
import argparse

model_name = "MobileCLIP2-S2"
model_file = model_name.lower().replace("-", "_")

parser = argparse.ArgumentParser(description="Static INT8 ONNX export with calibration")
parser.add_argument("--batch_size", type=int, default=200, help="Calibration batch size (try 200 or 300)")
parser.add_argument("--per_channel", action="store_true", default=True, help="Enable per-channel weight quantization")
parser.add_argument("--no_per_channel", action="store_true", default=False, help="Disable per-channel weight quantization")
args = parser.parse_args()
PER_CHANNEL = not args.no_per_channel
BATCH_SIZE = args.batch_size

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

image = preprocess(
    Image.open("docs/fig_accuracy_latency.png").convert("RGB")
).unsqueeze(0)
text = tokenizer("a diagram")

visual_model = model.visual
text_model: nn.Module = model.text

print("Input dim of visual model", image.shape)
print("Input dim of text model", text.shape)

torch.onnx.export(
    visual_model,
    (image,),
    f=f"./{model_file}_visual.onnx",
    external_data=False,
    verify=True,
)

torch.onnx.export(
    text_model,
    (text,),
    f=f"./{model_file}_text.onnx",
    external_data=False,
    verify=True,
)

print("Export ONNX done")

quantization.quant_pre_process(
    f"./{model_file}_visual.onnx",
    f"./{model_file}_visual_preproc.onnx",
)

quantization.quant_pre_process(
    f"./{model_file}_text.onnx",
    f"./{model_file}_text_preproc.onnx",
)

print("Preprocess done")


def generate_image_variants(base_img, num_variants):
    """生成图片的多种增强变体（支持组合增强）"""
    variants = []
    w, h = base_img.size

    AUGMENTATIONS = [
        "crop", "rotate", "noise", "blur", "brightness", "contrast",
        "flip", "color_jitter", "sharpen", "solarize", "posterize",
        "equalize", "autocontrast", "cutout", "perspective", "elastic",
        "invert", "grayscale_mix", "jpeg_compress", "original"
    ]

    for _ in range(num_variants):
        variant = base_img.copy()

        num_augs = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
        selected_augs = random.sample(AUGMENTATIONS, num_augs)

        for aug_type in selected_augs:
            if aug_type == "crop":
                crop_w = int(w * random.uniform(0.6, 1.0))
                crop_h = int(h * random.uniform(0.6, 1.0))
                left = random.randint(0, max(w - crop_w, 0))
                top = random.randint(0, max(h - crop_h, 0))
                variant = variant.crop((left, top, left + crop_w, top + crop_h))
                variant = variant.resize((w, h), Image.BILINEAR)

            elif aug_type == "rotate":
                angle = random.uniform(-30, 30)
                variant = variant.rotate(angle, fillcolor=(128, 128, 128))

            elif aug_type == "noise":
                img_array = np.array(variant).astype(np.float32)
                noise = np.random.normal(0, random.uniform(5, 25), img_array.shape)
                img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                variant = Image.fromarray(img_array)

            elif aug_type == "blur":
                radius = random.uniform(0.5, 3.0)
                variant = variant.filter(ImageFilter.GaussianBlur(radius=radius))

            elif aug_type == "brightness":
                enhancer = ImageEnhance.Brightness(variant)
                factor = random.uniform(0.5, 1.5)
                variant = enhancer.enhance(factor)

            elif aug_type == "contrast":
                enhancer = ImageEnhance.Contrast(variant)
                factor = random.uniform(0.5, 1.5)
                variant = enhancer.enhance(factor)

            elif aug_type == "flip":
                if random.random() > 0.5:
                    variant = variant.transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    variant = variant.transpose(Image.FLIP_TOP_BOTTOM)

            elif aug_type == "color_jitter":
                enhancer = ImageEnhance.Color(variant)
                factor = random.uniform(0.3, 1.7)
                variant = enhancer.enhance(factor)

            elif aug_type == "sharpen":
                enhancer = ImageEnhance.Sharpness(variant)
                factor = random.uniform(0.0, 2.0)
                variant = enhancer.enhance(factor)

            elif aug_type == "solarize":
                threshold = random.randint(50, 200)
                variant = ImageOps.solarize(variant, threshold=threshold)

            elif aug_type == "posterize":
                bits = random.randint(2, 6)
                variant = ImageOps.posterize(variant, bits=bits)

            elif aug_type == "equalize":
                variant = ImageOps.equalize(variant)

            elif aug_type == "autocontrast":
                variant = ImageOps.autocontrast(variant, cutoff=random.uniform(0, 5))

            elif aug_type == "cutout":
                img_array = np.array(variant)
                cut_w = int(w * random.uniform(0.05, 0.3))
                cut_h = int(h * random.uniform(0.05, 0.3))
                cx = random.randint(0, max(w - cut_w, 0))
                cy = random.randint(0, max(h - cut_h, 0))
                fill_val = random.randint(0, 255)
                img_array[cy:cy+cut_h, cx:cx+cut_w, :] = fill_val
                variant = Image.fromarray(img_array)

            elif aug_type == "perspective":
                img_array = np.array(variant).astype(np.float32)
                dx = random.uniform(-0.1, 0.1) * w
                dy = random.uniform(-0.1, 0.1) * h
                from PIL import Image as PILImage
                coeffs = [
                    dx, dy,
                    w - dx, dy,
                    w + dx, h - dy,
                    -dx, h + dy
                ]
                try:
                    variant = variant.transform((w, h), Image.PERSPECTIVE,
                                               list(map(float, [
                                                   1, 0, dx * 0.3,
                                                   0, 1, dy * 0.3,
                                                   0.0001, 0.0001
                                               ])),
                                               Image.BICUBIC)
                except Exception:
                    pass

            elif aug_type == "elastic":
                img_array = np.array(variant).astype(np.float32)
                dx = np.random.uniform(-5, 5, img_array.shape[:2]).astype(np.float32)
                dy = np.random.uniform(-5, 5, img_array.shape[:2]).astype(np.float32)
                from PIL import Image as PILImage
                dx_img = PILImage.fromarray(dx.astype(np.uint8))
                dy_img = PILImage.fromarray(dy.astype(np.uint8))
                dx_blurred = np.array(dx_img.filter(ImageFilter.GaussianBlur(radius=3))).astype(np.float32) * 2
                dy_blurred = np.array(dy_img.filter(ImageFilter.GaussianBlur(radius=3))).astype(np.float32) * 2
                h_idx, w_idx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                new_h = np.clip(h_idx + dy_blurred, 0, h - 1).astype(np.int32)
                new_w = np.clip(w_idx + dx_blurred, 0, w - 1).astype(np.int32)
                img_array = img_array[new_h, new_w]
                variant = Image.fromarray(img_array.astype(np.uint8))

            elif aug_type == "invert":
                variant = ImageOps.invert(variant)

            elif aug_type == "grayscale_mix":
                gray = variant.convert("L").convert("RGB")
                alpha = random.uniform(0.2, 0.8)
                img_arr = np.array(variant).astype(np.float32)
                gray_arr = np.array(gray).astype(np.float32)
                mixed = (img_arr * alpha + gray_arr * (1 - alpha)).astype(np.uint8)
                variant = Image.fromarray(mixed)

            elif aug_type == "jpeg_compress":
                import io
                buf = io.BytesIO()
                quality = random.randint(10, 75)
                variant.save(buf, format="JPEG", quality=quality)
                buf.seek(0)
                variant = Image.open(buf).copy()

        variants.append(variant)

    return variants


def generate_random_image(size=(256, 256)):
    """生成随机图片（噪声、渐变、几何图形等）"""
    img_type = random.choice([
        "noise", "gradient", "solid", "stripes", "checkerboard",
        "circles", "gaussian_blob", "perlin_like", "random_blocks"
    ])

    if img_type == "noise":
        arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

    elif img_type == "gradient":
        arr = np.zeros((*size, 3), dtype=np.uint8)
        direction = random.choice(["h", "v", "d", "radial"])
        color1 = np.random.randint(0, 256, 3).astype(np.float32)
        color2 = np.random.randint(0, 256, 3).astype(np.float32)
        for i in range(size[0]):
            for j in range(size[1]):
                if direction == "h":
                    t = j / size[1]
                elif direction == "v":
                    t = i / size[0]
                elif direction == "d":
                    t = (i + j) / (size[0] + size[1])
                else:
                    cx, cy = size[1] / 2, size[0] / 2
                    t = min(1.0, np.sqrt((i - cy)**2 + (j - cx)**2) / (max(size) / 2))
                arr[i, j] = color1 * (1 - t) + color2 * t

    elif img_type == "solid":
        color = np.random.randint(0, 256, 3)
        arr = np.full((*size, 3), color, dtype=np.uint8)

    elif img_type == "stripes":
        arr = np.zeros((*size, 3), dtype=np.uint8)
        stripe_width = random.randint(5, 30)
        color = np.random.randint(0, 256, 3)
        for i in range(0, size[0], stripe_width * 2):
            arr[i:i+stripe_width, :] = color

    elif img_type == "checkerboard":
        arr = np.zeros((*size, 3), dtype=np.uint8)
        square_size = random.randint(10, 40)
        color1 = np.random.randint(0, 256, 3)
        color2 = np.random.randint(0, 256, 3)
        for i in range(0, size[0], square_size):
            for j in range(0, size[1], square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    arr[i:i+square_size, j:j+square_size] = color1
                else:
                    arr[i:i+square_size, j:j+square_size] = color2

    elif img_type == "circles":
        arr = np.random.randint(0, 50, (*size, 3), dtype=np.uint8)
        for _ in range(random.randint(2, 8)):
            cx, cy = random.randint(0, size[1]), random.randint(0, size[0])
            r = random.randint(10, min(size) // 3)
            color = np.random.randint(50, 256, 3)
            y, x = np.ogrid[:size[0], :size[1]]
            mask = (x - cx)**2 + (y - cy)**2 <= r**2
            arr[mask] = color

    elif img_type == "gaussian_blob":
        arr = np.zeros((*size, 3), dtype=np.float32)
        for c in range(3):
            cx = random.randint(0, size[1])
            cy = random.randint(0, size[0])
            sigma = random.uniform(20, 80)
            y, x = np.ogrid[:size[0], :size[1]]
            blob = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
            arr[:, :, c] = blob * random.uniform(100, 255)
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    elif img_type == "perlin_like":
        arr = np.zeros((*size, 3), dtype=np.float32)
        for c in range(3):
            freq = random.uniform(0.01, 0.05)
            phase_x = random.uniform(0, 2 * np.pi)
            phase_y = random.uniform(0, 2 * np.pi)
            y, x = np.ogrid[:size[0], :size[1]]
            arr[:, :, c] = (np.sin(x * freq + phase_x) + np.sin(y * freq + phase_y) + 2) / 4 * 255
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    elif img_type == "random_blocks":
        arr = np.zeros((*size, 3), dtype=np.uint8)
        block_size = random.randint(8, 32)
        for i in range(0, size[0], block_size):
            for j in range(0, size[1], block_size):
                color = np.random.randint(0, 256, 3)
                arr[i:min(i+block_size, size[0]), j:min(j+block_size, size[1])] = color

    return Image.fromarray(arr.astype(np.uint8))


class ImageCalibrationDataReader(quantization.CalibrationDataReader):
    def __init__(self, num_samples=300):
        self.num_samples = num_samples
        self.current_index = 0
        self.preprocess = preprocess

        self.image_data = []

        base_img = Image.open("docs/fig_accuracy_latency.png").convert("RGB")
        self.image_data.append(preprocess(base_img).unsqueeze(0).numpy())

        num_variants = min(int(num_samples * 0.4), num_samples - 1)
        variants = generate_image_variants(base_img, num_variants)
        for v in variants:
            self.image_data.append(preprocess(v).unsqueeze(0).numpy())

        remaining = num_samples - len(self.image_data)
        for _ in range(remaining):
            rand_img = generate_random_image(size=(256, 256))
            self.image_data.append(preprocess(rand_img).unsqueeze(0).numpy())

        print(f"Image calibration data prepared: {len(self.image_data)} samples")

    def get_next(self):
        if self.current_index < len(self.image_data):
            data = {"x": self.image_data[self.current_index]}
            self.current_index += 1
            return data
        return None

    def rewind(self):
        self.current_index = 0


# 随机单词库（用于生成校准文本）
WORD_LIST = [
    "apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew",
    "kiwi", "lemon", "mango", "nectarine", "orange", "papaya", "quince", "raspberry",
    "strawberry", "tangerine", "watermelon", "blueberry", "blackberry", "coconut",
    "dog", "cat", "bird", "fish", "horse", "elephant", "lion", "tiger", "bear",
    "wolf", "fox", "rabbit", "deer", "monkey", "panda", "zebra", "giraffe", "kangaroo",
    "car", "truck", "bus", "bicycle", "motorcycle", "train", "airplane", "boat",
    "house", "building", "bridge", "tower", "castle", "temple", "church", "mosque",
    "mountain", "river", "ocean", "lake", "forest", "desert", "beach", "island",
    "volcano", "waterfall", "canyon", "valley", "meadow", "jungle", "savanna",
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black",
    "white", "gray", "golden", "silver", "bronze", "copper", "crimson", "azure",
    "happy", "sad", "angry", "excited", "calm", "nervous", "brave", "shy", "proud",
    "running", "jumping", "flying", "swimming", "walking", "dancing", "singing",
    "eating", "sleeping", "reading", "writing", "painting", "cooking", "playing",
    "beautiful", "ugly", "big", "small", "tall", "short", "fast", "slow", "hot", "cold",
    "old", "new", "clean", "dirty", "smooth", "rough", "soft", "hard", "loud", "quiet",
    "morning", "afternoon", "evening", "night", "sunrise", "sunset", "dawn", "dusk",
    "spring", "summer", "autumn", "winter", "january", "february", "march", "april",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "music", "art", "science", "math", "history", "geography", "physics", "chemistry",
    "football", "basketball", "tennis", "golf", "swimming", "running", "cycling",
    "pizza", "burger", "pasta", "sushi", "salad", "soup", "steak", "chicken", "rice",
    "doctor", "teacher", "engineer", "artist", "musician", "chef", "pilot", "driver",
    "friend", "family", "neighbor", "stranger", "guest", "host", "leader", "member",
    "city", "country", "village", "town", "capital", "suburb", "downtown", "countryside",
    "phone", "computer", "tablet", "camera", "television", "radio", "watch", "clock",
    "book", "newspaper", "magazine", "letter", "email", "message", "note", "document",
    "road", "street", "highway", "path", "trail", "tunnel", "railway", "runway",
    "flower", "tree", "grass", "bush", "vine", "weed", "seed", "leaf", "root", "branch",
    "cloud", "rain", "snow", "wind", "storm", "thunder", "lightning", "fog", "mist",
    "star", "moon", "planet", "galaxy", "universe", "comet", "asteroid", "meteor",
    "child", "adult", "baby", "teenager", "senior", "parent", "sibling", "cousin",
    "school", "university", "college", "library", "museum", "theater", "cinema", "park",
    "hospital", "clinic", "pharmacy", "restaurant", "cafe", "hotel", "market", "store",
    "shirt", "pants", "dress", "skirt", "jacket", "coat", "shoes", "hat", "gloves",
    "table", "chair", "bed", "sofa", "desk", "shelf", "cabinet", "drawer", "mirror",
    "window", "door", "roof", "wall", "floor", "ceiling", "stairs", "elevator",
    "knife", "fork", "spoon", "plate", "bowl", "cup", "glass", "bottle", "jar",
    "circle", "square", "triangle", "rectangle", "oval", "diamond", "star", "heart",
    "road", "bridge", "tunnel", "dam", "harbor", "port", "airport", "station",
    "king", "queen", "prince", "princess", "president", "minister", "mayor", "judge",
    "war", "peace", "battle", "victory", "defeat", "soldier", "army", "navy", "guard",
    "money", "coin", "bill", "check", "card", "bank", "cash", "price", "cost", "value",
    "love", "hate", "joy", "fear", "hope", "dream", "wish", "goal", "plan", "idea",
    "truth", "lie", "fact", "fiction", "story", "tale", "legend", "myth", "history",
    "water", "fire", "earth", "air", "metal", "wood", "stone", "sand", "clay", "mud",
    "glass", "plastic", "rubber", "leather", "silk", "cotton", "wool", "linen",
    "north", "south", "east", "west", "up", "down", "left", "right", "front", "back",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "second", "third", "last", "next", "previous", "beginning", "end",
    "always", "never", "sometimes", "often", "rarely", "usually", "frequently",
]


def generate_random_text():
    """使用随机单词库生成自然语言风格的文本"""
    pattern = random.choice([
        "simple",          # 3-5 个单词的简单短语
        "descriptive",     # 带形容词的描述
        "action",          # 主谓宾结构
        "prepositional",   # 带介词短语
        "complex",         # 复合句
    ])

    if pattern == "simple":
        words = random.sample(WORD_LIST, random.randint(3, 5))
        return " ".join(words)

    elif pattern == "descriptive":
        adj = random.choice(["a", "an", "the"])
        desc = random.choice(WORD_LIST)
        noun = random.choice(WORD_LIST)
        return f"{adj} {desc} {noun}"

    elif pattern == "action":
        subject = random.choice(WORD_LIST)
        verb = random.choice(WORD_LIST)
        obj = random.choice(WORD_LIST)
        return f"{subject} {verb} {obj}"

    elif pattern == "prepositional":
        noun = random.choice(WORD_LIST)
        prep = random.choice(["on", "in", "at", "under", "over", "beside", "near", "behind"])
        loc = random.choice(WORD_LIST)
        return f"{noun} {prep} {loc}"

    elif pattern == "complex":
        part1 = generate_random_text()
        conj = random.choice(["and", "or", "but", "with", "near"])
        part2 = generate_random_text()
        return f"{part1} {conj} {part2}"


class TextCalibrationDataReader(quantization.CalibrationDataReader):
    def __init__(self, num_samples=300):
        self.num_samples = num_samples
        self.current_index = 0
        self.tokenizer = tokenizer

        self.text_data = []

        sample_texts = [
            "a diagram",
            "a photo of a cat",
            "a photo of a dog",
            "a beautiful landscape",
            "a red apple on a table",
            "an abstract painting",
            "a person riding a bike",
            "a sunset over the ocean",
            "a cup of coffee",
            "a modern building",
            "a child playing in the park",
            "a car driving on the highway",
            "a bird flying in the sky",
            "a flower blooming in spring",
            "a river flowing through mountains",
            "a teacher writing on the board",
            "a musician playing guitar",
            "a chef cooking in the kitchen",
            "a dog running on the beach",
            "a star shining in the night",
        ]
        for txt in sample_texts:
            tokens = tokenizer(txt).numpy()
            self.text_data.append(tokens)

        num_random = min(int(num_samples * 0.5), num_samples - len(self.text_data))
        for _ in range(num_random):
            txt = generate_random_text()
            tokens = tokenizer(txt).numpy()
            self.text_data.append(tokens)

        vocab_size = 49408
        remaining = num_samples - len(self.text_data)
        for _ in range(remaining):
            length = np.random.randint(5, 77)
            random_tokens = np.random.randint(0, vocab_size, size=(1, length))
            random_tokens[0, 0] = 49406
            random_tokens[0, -1] = 49407
            padded = np.zeros((1, 77), dtype=np.int64)
            padded[0, :length] = random_tokens[0, :length]
            self.text_data.append(padded)

        print(f"Text calibration data prepared: {len(self.text_data)} samples")

    def get_next(self):
        if self.current_index < len(self.text_data):
            data = {"text": self.text_data[self.current_index]}
            self.current_index += 1
            return data
        return None

    def rewind(self):
        self.current_index = 0


# 静态量化 visual 模型
print(f"Starting static quantization for visual model (batch={BATCH_SIZE}, per_channel={PER_CHANNEL})...")
image_calib_reader = ImageCalibrationDataReader(num_samples=BATCH_SIZE)

quantization.quantize_static(
    model_input=f"./{model_file}_visual_preproc.onnx",
    model_output=f"./{model_file}_visual_static_int8.onnx",
    calibration_data_reader=image_calib_reader,
    quant_format=quantization.QuantFormat.QOperator,
    activation_type=quantization.QuantType.QUInt8,
    weight_type=quantization.QuantType.QUInt8,
    per_channel=PER_CHANNEL,
)

print("Visual model static quantization done")

# 静态量化 text 模型 (使用 QDQ + QInt8，因为 text 模型包含 int64 输入和 ArgMax，QOperator + QUInt8 不支持)
print(f"Starting static quantization for text model (batch={BATCH_SIZE}, per_channel={PER_CHANNEL})...")
text_calib_reader = TextCalibrationDataReader(num_samples=BATCH_SIZE)

quantization.quantize_static(
    model_input=f"./{model_file}_text_preproc.onnx",
    model_output=f"./{model_file}_text_static_int8.onnx",
    calibration_data_reader=text_calib_reader,
    quant_format=quantization.QuantFormat.QDQ,
    activation_type=quantization.QuantType.QInt8,
    weight_type=quantization.QuantType.QInt8,
    per_channel=PER_CHANNEL,
)

print("Text model static quantization done")

# 验证量化后的模型
import onnxruntime as ort

print("\n--- Verification ---")

# FP32
sess_image_fp32 = ort.InferenceSession(f"./{model_file}_visual.onnx")
sess_text_fp32 = ort.InferenceSession(f"./{model_file}_text.onnx")

image_np = image.numpy()
text_np = text.numpy()
if sess_text_fp32.get_inputs()[0].type == "tensor(int32)":
    text_np = text_np.astype(np.int32)

out_image_fp32 = sess_image_fp32.run(None, {sess_image_fp32.get_inputs()[0].name: image_np})
out_text_fp32 = sess_text_fp32.run(None, {sess_text_fp32.get_inputs()[0].name: text_np})

image_feat_fp32 = out_image_fp32[0] / np.linalg.norm(out_image_fp32[0], axis=-1, keepdims=True)
text_feat_fp32 = out_text_fp32[0] / np.linalg.norm(out_text_fp32[0], axis=-1, keepdims=True)
sim_fp32 = (image_feat_fp32 @ text_feat_fp32.T) * 100.0

# Static INT8
sess_image_int8 = ort.InferenceSession(f"./{model_file}_visual_static_int8.onnx")
sess_text_int8 = ort.InferenceSession(f"./{model_file}_text_static_int8.onnx")

text_np_int8 = text.numpy()
if sess_text_int8.get_inputs()[0].type == "tensor(int32)":
    text_np_int8 = text_np_int8.astype(np.int32)

out_image_int8 = sess_image_int8.run(None, {sess_image_int8.get_inputs()[0].name: image_np})
out_text_int8 = sess_text_int8.run(None, {sess_text_int8.get_inputs()[0].name: text_np_int8})

image_feat_int8 = out_image_int8[0] / np.linalg.norm(out_image_int8[0], axis=-1, keepdims=True)
text_feat_int8 = out_text_int8[0] / np.linalg.norm(out_text_int8[0], axis=-1, keepdims=True)
sim_int8 = (image_feat_int8 @ text_feat_int8.T) * 100.0

print(f"FP32 similarity: {sim_fp32[0][0]:.4f}")
print(f"Static INT8 similarity: {sim_int8[0][0]:.4f}")
print(f"Difference: {abs(sim_fp32[0][0] - sim_int8[0][0]):.4f}")

# 对比 PyTorch
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    pt_sim = (100.0 * image_features @ text_features.T).numpy()

print(f"PyTorch similarity: {pt_sim[0][0]:.4f}")
print(f"FP32 ONNX diff from PyTorch: {abs(sim_fp32[0][0] - pt_sim[0][0]):.4f}")
print(f"Static INT8 diff from PyTorch: {abs(sim_int8[0][0] - pt_sim[0][0]):.4f}")
