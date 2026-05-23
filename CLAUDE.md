
  1. Project Overview

  This is the official repository for MobileCLIP and MobileCLIP2 -- fast image-text models from Apple (CVPR 2024 and TMLR 2025). The project is focused on multi-modal
  reinforced training for efficient CLIP models, with an emphasis on mobile deployment. The project also contains a cloned open_clip repository as a submodule/dependency.

  ---
  2. Project Structure (Key Files and Directories)

  d:\Develop\Projects\Flutter\ml-mobileclip\
  |
  |-- .gitignore
  |-- .venv/                          # Python virtual environment (uv-managed)
  |-- AGENTS.md                       # Agent instructions (use uv, no help() calls)
  |-- CLAUDE.md                       # References AGENTS.md
  |-- HOW_TO.md                       # Quick-start guide for setup & export
  |-- README.md                       # Main project README
  |-- setup.py                        # Legacy pip-installable setup (mobileclip v0.1.0)
  |-- pyproject.toml                  # Modern project config (uv-managed)
  |-- uv.lock                         # uv lockfile (present, confirms uv usage)
  |-- requirements.txt                # Legacy dependencies
  |
  |-- mobileclip/                     # MobileCLIP v1 source code
  |   |-- __init__.py
  |   |-- clip.py                     # CLIP model wrapper
  |   |-- image_encoder.py            # Image encoder API
  |   |-- text_encoder.py             # Text encoder API
  |   |-- logger.py
  |   |-- configs/                    # V1 model config JSONs
  |   |   |-- mobileclip_b.json
  |   |   |-- mobileclip_s0.json
  |   |   |-- mobileclip_s1.json
  |   |   |-- mobileclip_s2.json
  |   |-- models/                     # Model architecture definitions
  |   |   |-- mci.py                  # MCI (MobileOne-based) image models
  |   |   |-- vit.py                  # ViT image models
  |   |-- modules/                    # Reusable building blocks
  |       |-- common/
  |       |   |-- mobileone.py        # MobileOne re-parameterizable blocks
  |       |   |-- transformer.py      # Transformer blocks
  |       |-- image/
  |       |   |-- replknet.py         # RepLKNet image encoder
  |       |   |-- image_projection.py
  |       |-- text/
  |           |-- repmixer.py         # RepMixer text encoder
  |           |-- tokenizer.py        # Tokenizer
  |
  |-- mobileclip2/                    # MobileCLIP v2 additions
  |   |-- mobileclip2.py              # V2 model definitions (fastvit_mci3, fastvit_mci4)
  |   |-- model_configs/              # V2 model config JSONs
  |   |   |-- MobileCLIP2-S0.json
  |   |   |-- MobileCLIP2-S2.json
  |   |   |-- MobileCLIP2-B.json
  |   |   |-- MobileCLIP2-S3.json
  |   |   |-- MobileCLIP2-S4.json
  |   |   |-- MobileCLIP2-L-14.json
  |   |   |-- MobileCLIP-S3.json       (V1 also has S3/S4/L-14 configs here)
  |   |   |-- MobileCLIP-S4.json
  |   |   |-- MobileCLIP-L-14.json
  |   |-- open_clip_inference_only.patch  # Patch for open_clip integration
  |
  |-- open_clip/                      # Cloned OpenCLIP repo (patched for MobileCLIP2)
  |   |-- pyproject.toml              # open_clip_torch package (pdm-based)
  |   |-- src/                        # OpenCLIP source
  |   |-- ...
  |
  |-- export_fp16_onnx.py             # FP16 visual + INT8 text ONNX export
  |-- export_int8_onnx.py             # Basic FP32 ONNX export + dynamic INT8 quant
  |-- export_true_int8_onnx.py        # True INT8 dynamic quant ONNX export
  |-- export_int8_bnb_onnx.py         # BitsAndBytes INT8 quant + ONNX export
  |-- export_model.sh                 # Shell script to export & copy models
  |
  |-- test_mobileclip_onnx.py         # Test V1 ONNX inference
  |-- test_int8_onnx.py               # Test INT8 ONNX inference (V2)
  |-- test_mobileclip_original.py     # Test V1 PyTorch inference
  |-- test_int8_export.py             # Test custom INT8 Linear export to ONNX
  |-- test_bnb_reconstruct.py         # Test bitsandbytes weight reconstruction
  |
  |-- fp16_results/                   # Exported FP16/INT8 ONNX models
  |   |-- mobileclip2_s2_visual.onnx   (69 MB)
  |   |-- mobileclip2_s2_text.onnx     (61 MB)
  |
  |-- int8_results/                   # Exported INT8 ONNX models
  |   |-- mobileclip2_s2_visual.onnx   (35 MB)
  |   |-- mobileclip2_s2_text.onnx     (61 MB)
  |
  |-- eval/                           # Evaluation scripts
  |   |-- zeroshot_imagenet.py
  |
  |-- results/                        # Benchmark results (JSONL files for all variants)
  |-- docs/                           # Figures and diagrams
  |-- training/                       # Training configs & OpenCLIP patches
  |-- ios_app/                        # iOS demo app (Xcode project)
  |
  |-- ExampleFloatArray.kt            # Kotlin float array (embedding data for Android)
  |-- gen_random_sentences.py         # Utility to generate random sentences
  |-- hf_dataset_example.py           # HuggingFace dataset loading example
  |-- getting_started.ipynb           # Jupyter notebook for getting started
  |-- get_pretrained_models.sh        # Script to download V1 pretrained models

  ---
  3. Available Model Files (.pt)


  Notable: The pyproject.toml does not include torch, open-clip-torch, clip-benchmark, or datasets as direct dependencies. Those come from the requirements.txt or the
  separately installed open_clip package.

  (C) open_clip/pyproject.toml (separate package)

  The cloned OpenCLIP repo uses PDM as its build system and declares its own dependencies (torch>=2.6, torchvision, timm>=1.0.17, safetensors, etc.).

  ---
  6. UV Usage Confirmation

  Yes, uv is being used for dependency management. The evidence:

  1. uv.lock exists at d:\Develop\Projects\Flutter\ml-mobileclip\uv.lock (142 KB) -- this is the uv lockfile with resolved package hashes and versions.
  2. pyproject.toml is present and configured for uv.
  3. HOW_TO.md explicitly instructs:
    - uv sync to install dependencies
    - uv pip install -e . to install open_clip
    - uv run export_true_int8_onnx.py to run scripts
  4. AGENTS.md explicitly states: "Use uv to manage Python dependency environment"
  5. .venv directory is present, consistent with uv's virtual environment management.

  The requires-python = ">=3.13" in pyproject.toml indicates the project targets Python 3.13+.