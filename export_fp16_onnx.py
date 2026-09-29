#!/usr/bin/env python3
"""Export an official MobileCLIP S-series model to FP16 ONNX.

Usage:
    python export_fp16_onnx.py s0
    python export_fp16_onnx.py s1
    python export_fp16_onnx.py s2

MobileCLIP2-S1 was not released by Apple. Consequently, ``s1`` exports the
official MobileCLIP-S1 model; ``s0`` and ``s2`` export MobileCLIP2 models.
The checkpoint is downloaded from Hugging Face on first use and then reused
from the normal Hugging Face cache.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import onnx
import torch
from onnxruntime.transformers.float16 import convert_float_to_float16


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "third_party" / "open_clip" / "src"))

import open_clip  # noqa: E402
from mobileclip.modules.common.mobileone import reparameterize_model  # noqa: E402


# There is no official MobileCLIP2-S1 architecture or checkpoint. Keep this
# mapping explicit so exported filenames always describe the actual model.
MODELS = {
    "s0": ("MobileCLIP2-S0", "dfndr2b"),
    "s1": ("MobileCLIP-S1", "datacompdr"),
    "s2": ("MobileCLIP2-S2", "dfndr2b"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and export MobileCLIP image/text encoders as FP16 ONNX."
    )
    parser.add_argument("model", choices=MODELS, help="model size: s0, s1, or s2")
    return parser.parse_args()


def export_fp16(
    module: torch.nn.Module,
    sample_input: torch.Tensor,
    output_path: Path,
    input_name: str,
) -> None:
    """Export through FP32 ONNX, then convert all floating-point values to FP16."""
    output_name = f"{input_name}_features"
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}_", suffix="_fp32.onnx", dir=output_path.parent
    ) as temporary_file:
        with torch.inference_mode():
            torch.onnx.export(
                module,
                sample_input,
                temporary_file.name,
                export_params=True,
                external_data=False,
                opset_version=18,
                do_constant_folding=True,
                input_names=[input_name],
                output_names=[output_name],
                dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
            )

        fp32_model = onnx.load(temporary_file.name)
        fp16_model = convert_float_to_float16(fp32_model, keep_io_types=False)
        onnx.checker.check_model(fp16_model)
        onnx.save_model(fp16_model, output_path, save_as_external_data=False)


def main() -> None:
    args = parse_args()
    model_name, pretrained_tag = MODELS[args.model]
    file_prefix = model_name.lower().replace("-", "_")
    visual_output = Path.cwd() / f"{file_prefix}_visual_fp16.onnx"
    text_output = Path.cwd() / f"{file_prefix}_text_fp16.onnx"

    if args.model == "s1":
        print(
            "Note: Apple did not release MobileCLIP2-S1; exporting the official "
            "MobileCLIP-S1 instead."
        )

    print(f"Loading {model_name} (the checkpoint downloads automatically if needed)...")
    model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_tag,
        device="cpu",
        **model_kwargs,
    )
    model.eval()
    model = reparameterize_model(model)
    model.eval()

    model_config = open_clip.get_model_config(model_name)
    image_size = model_config["vision_cfg"]["image_size"]
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    context_length = model_config["text_cfg"]["context_length"]

    image = torch.zeros(1, 3, *image_size, dtype=torch.float32)
    text = torch.zeros(1, context_length, dtype=torch.int64)

    print(f"Exporting visual encoder to {visual_output.name}...")
    export_fp16(model.visual, image, visual_output, "image")
    print(f"Exporting text encoder to {text_output.name}...")
    export_fp16(model.text, text, text_output, "text")

    print("Export complete:")
    print(f"  {visual_output}")
    print(f"  {text_output}")


if __name__ == "__main__":
    main()
