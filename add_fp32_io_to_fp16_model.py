"""Wrap an FP16 ONNX graph with FP32 input/output Cast nodes.

flutter_onnxruntime's public Dart API accepts Float32List efficiently, but has
no Float16List input.  Keeping FP16 weights while exposing FP32 model I/O avoids
an expensive per-inference platform-channel conversion and keeps the bundled
model compact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def replace_names(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
        for index, name in enumerate(node.output):
            if name == old:
                node.output[index] = new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.input)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise SystemExit("Expected a visual model with exactly one input and one output")

    old_input = model.graph.input[0]
    old_output = model.graph.output[0]
    if old_input.type.tensor_type.elem_type != TensorProto.FLOAT16:
        raise SystemExit("Model input is not float16")
    if old_output.type.tensor_type.elem_type != TensorProto.FLOAT16:
        raise SystemExit("Model output is not float16")

    input_name = old_input.name
    output_name = old_output.name
    internal_input = f"{input_name}_fp16_internal"
    internal_output = f"{output_name}_fp16_internal"
    replace_names(model, input_name, internal_input)
    replace_names(model, output_name, internal_output)

    input_shape = [
        dim.dim_param if dim.dim_param else dim.dim_value
        for dim in old_input.type.tensor_type.shape.dim
    ]
    output_shape = [
        dim.dim_param if dim.dim_param else dim.dim_value
        for dim in old_output.type.tensor_type.shape.dim
    ]
    model.graph.input[0].CopyFrom(
        helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
    )
    model.graph.output[0].CopyFrom(
        helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)
    )

    input_cast = helper.make_node(
        "Cast", [input_name], [internal_input], name="picquery_fp32_to_fp16", to=TensorProto.FLOAT16
    )
    output_cast = helper.make_node(
        "Cast", [internal_output], [output_name], name="picquery_fp16_to_fp32", to=TensorProto.FLOAT
    )
    model.graph.node.insert(0, input_cast)
    model.graph.node.append(output_cast)
    onnx.checker.check_model(model)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
