"""Benchmark the visual ONNX model used by the Flutter application.

Run from the repository root with:

    uv run --project ml-mobileclip ml-mobileclip/benchmark_visual_onnx.py

The default model is the model actually bundled by the app, rather than the
FP32 export kept in this submodule.  Session construction, first inference and
steady-state inference are reported separately because CoreML compilation can
make the first run substantially more expensive.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "assets/models/mobileclip2_s0_visual.onnx"
DEFAULT_IMAGE = ROOT / "ml-mobileclip/docs/example.png"


def elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000


def preprocess(path: Path, dtype: np.dtype) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        scale = 256 / min(width, height)
        resized_width = max(256, round(width * scale))
        resized_height = max(256, round(height * scale))
        image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        left = (resized_width - 256) // 2
        top = (resized_height - 256) // 2
        image = image.crop((left, top, left + 256, top + 256))
        array = np.asarray(image, dtype=np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...], dtype=dtype)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--provider", choices=("cpu", "coreml"), default="cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--intra-op-threads", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    provider = {
        "cpu": "CPUExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
    }[args.provider]
    if provider not in ort.get_available_providers():
        raise SystemExit(f"{provider} is unavailable; found {ort.get_available_providers()}")

    options = ort.SessionOptions()
    if args.intra_op_threads > 0:
        options.intra_op_num_threads = args.intra_op_threads

    started = time.perf_counter_ns()
    providers = [provider]
    if provider != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=providers)
    session_ms = elapsed_ms(started)
    model_input = session.get_inputs()[0]
    dtype = np.float16 if model_input.type == "tensor(float16)" else np.float32

    preprocess_times: list[float] = []
    tensor = None
    for _ in range(args.iterations):
        started = time.perf_counter_ns()
        tensor = preprocess(args.image, dtype)
        preprocess_times.append(elapsed_ms(started))
    assert tensor is not None
    if args.batch_size > 1:
        tensor = np.repeat(tensor, args.batch_size, axis=0)

    started = time.perf_counter_ns()
    output = session.run(None, {model_input.name: tensor})[0]
    first_run_ms = elapsed_ms(started)
    for _ in range(args.warmup):
        session.run(None, {model_input.name: tensor})

    inference_times: list[float] = []
    for _ in range(args.iterations):
        started = time.perf_counter_ns()
        session.run(None, {model_input.name: tensor})
        inference_times.append(elapsed_ms(started))

    print(f"model: {args.model.resolve()}")
    print(f"model size: {args.model.stat().st_size / 1024 / 1024:.1f} MiB")
    print(f"provider: {session.get_providers()}")
    print(f"input: name={model_input.name} shape={model_input.shape} type={model_input.type}")
    print(f"output: shape={output.shape} dtype={output.dtype}")
    print(f"batch size: {args.batch_size}")
    print(f"session creation: {session_ms:.2f} ms")
    print(f"first inference: {first_run_ms:.2f} ms")
    print(
        "preprocess: "
        f"median={statistics.median(preprocess_times):.2f} ms "
        f"p90={percentile(preprocess_times, 0.9):.2f} ms "
        f"min={min(preprocess_times):.2f} ms"
    )
    print(
        "steady inference: "
        f"median={statistics.median(inference_times):.2f} ms "
        f"p90={percentile(inference_times, 0.9):.2f} ms "
        f"min={min(inference_times):.2f} ms "
        f"mean={statistics.mean(inference_times):.2f} ms"
    )
    print(
        "steady per image: "
        f"median={statistics.median(inference_times) / args.batch_size:.2f} ms"
    )


if __name__ == "__main__":
    main()
