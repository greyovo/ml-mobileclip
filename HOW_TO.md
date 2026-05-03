# Clone OpenCLIP repository, add MobileCLIP2 models, and install

## install dependencies:
```bash
uv sync
```

## download mobileclip2 models:

https://huggingface.co/apple/MobileCLIP2-S2/tree/main

put them in the root directory of the project.

## clone open_clip repository:

```bash
git clone https://github.com/mlfoundations/open_clip.git
pushd open_clip
git apply ../mobileclip2/open_clip_inference_only.patch
cp -r ../mobileclip2/* ./src/open_clip/
uv pip install -e .
popd
```

# run quantization:

```bash
uv run export_true_int8_onnx.py
```
