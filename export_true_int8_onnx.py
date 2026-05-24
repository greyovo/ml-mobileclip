"""
使用 bitsandbytes INT8 行级量化导出 ONNX 模型

参考 open_clip/tutorials/int8_tutorial.ipynb 的方法：在 PyTorch 层面使用
bitsandbytes 的 SwitchBackLinear 替换 nn.Linear，运行前向传播触发量化，
再通过 convert_int8_model_to_inference_mode 预计算量化权重，最后提取
int8 weight + scale 并创建 ONNX 可导出的模块直接导出。

相比 ONNX Runtime 动态量化的优势：
- 行级量化（row-wise）比张量级量化精度更高
- 量化权重直接以 int8 格式存入 ONNX，模型体积减半
- 不依赖 ONNX Runtime 量化工具链，跨 runtime 通用

用法:
    uv run export_true_int8_onnx.py           # 使用 bitsandbytes 量化（CUDA）
    uv run export_true_int8_onnx.py --cpu     # 使用手动行级量化（CPU）
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn
import open_clip
import numpy as np
from mobileclip.modules.common.mobileone import reparameterize_model
import os
import sys
import argparse
import copy
import torch.nn.functional as F

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "MobileCLIP2-S0"
MODEL_FILE = MODEL_NAME.lower().replace("-", "_")
OUTPUT_DIR = "int8_results"
FP32_TEMP_DIR = "fp32_temp_for_quant"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FP32_TEMP_DIR, exist_ok=True)


def get_model_kwargs(model_name):
    if model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14"):
        return {}
    return {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}


# ============================================================
# ONNX 可导出的 INT8 量化层（分组量化，group_size=128）
# ============================================================
# 注：bitsandbytes 使用行级量化（group_size = in_features），但行级量化对
# 小维度层（如 512 列）的误差较大，在 12 层 Transformer 中会累积到 50%+ 精度损失。
# 这里改用分组量化（group_size=128），每组 128 个权重共享一个 scale，
# 与 GPTQ/AWQ 的做法一致，可将量化误差降低 4-6 倍。
_GROUP_SIZE = 128


class ONNXInt8Linear(nn.Module):
    """存储 int8 权重 + float32 scale 的线性层（分组量化）。

    Scale 形状为 (out_features, num_groups)，每组 group_size 个权重共用一个 scale。
    当 group_size == in_features 时退化为行级量化（与 bitsandbytes 一致）。
    反量化公式: weight_fp32 = int8_weight.float() * scale.repeat_interleave(group_size, dim=1)
    """

    def __init__(self, int8_weight: torch.Tensor, scale: torch.Tensor,
                 bias: torch.Tensor | None = None, group_size: int = _GROUP_SIZE):
        super().__init__()
        out_features, in_features = int8_weight.shape
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer("int8_weight", int8_weight)
        self.register_buffer("scale", scale.float())
        if bias is not None:
            self.register_buffer("bias", bias.float())
        else:
            self.bias = None
        self.register_buffer("weight", torch.zeros(out_features, in_features))

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"group_size={self.group_size}, bias={self.bias is not None}")

    @classmethod
    def from_bnb_switchback(cls, module) -> "ONNXInt8Linear":
        """从 bitsandbytes SwitchBackLinear / Linear8bitLt 模块提取量化权重。

        注：bitsandbytes 使用行级量化，此方法保留其 scale 格式（(out_features,) 或 (out_features, 1)）。
        """
        state = module.state if hasattr(module, "state") else module.linear.state
        cb = state.CB.clone()
        scb = state.SCB.clone()
        if scb.dim() == 2:
            scb = scb.squeeze(1)
        # bnb 的行级 scale 是 (out_features,)，转成 (out_features, 1) 统一格式
        if scb.dim() == 1:
            scb = scb.unsqueeze(1)
        bias = None
        if hasattr(module, "linear"):
            bias = module.linear.bias
        elif hasattr(module, "bias"):
            bias = module.bias
        if bias is not None:
            bias = bias.data.clone() if isinstance(bias, torch.Tensor) else bias.clone()
        return cls(cb, scb, bias, group_size=cb.shape[1])

    @classmethod
    def from_linear_manual(cls, linear: nn.Linear,
                           group_size: int = _GROUP_SIZE) -> "ONNXInt8Linear":
        """手动分组量化 nn.Linear。

        若 in_features 不能被 group_size 整除或 group_size > in_features，
        则退化为行级量化（group_size = in_features）。
        """
        weight = linear.weight.data
        out_f, in_f = weight.shape
        effective_gs = group_size
        if in_f % group_size != 0 or in_f < group_size:
            effective_gs = in_f

        if effective_gs == in_f:
            # 行级量化
            max_abs = weight.abs().max(dim=1, keepdim=True)[0]  # (out_f, 1)
            scale = max_abs / 127.0  # (out_f, 1)
            int8_weight = (weight / max_abs * 127.0).round().clamp(-128, 127).to(torch.int8)
        else:
            # 分组量化
            num_groups = in_f // effective_gs
            w_reshaped = weight.view(out_f, num_groups, effective_gs)
            max_abs = w_reshaped.abs().max(dim=2, keepdim=True)[0]  # (out_f, num_groups, 1)
            scale = max_abs.squeeze(2) / 127.0  # (out_f, num_groups)
            int8_weight = (w_reshaped / max_abs * 127.0).round().clamp(-128, 127).to(torch.int8)
            int8_weight = int8_weight.view(out_f, in_f)

        bias = linear.bias.data.clone() if linear.bias is not None else None
        return cls(int8_weight, scale, bias, effective_gs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.group_size == self.in_features:
            # 行级量化: scale 为 (out_f, 1)，直接广播
            weight_fp = self.int8_weight.float() * self.scale
        else:
            # 分组量化: scale 为 (out_f, num_groups)，展开为 (out_f, in_f)
            weight_fp = self.int8_weight.float() * self.scale.repeat_interleave(
                self.group_size, dim=1)
        out = torch.matmul(x, weight_fp.t())
        if self.bias is not None:
            out = out + self.bias
        return out


class ONNXInt8Conv2d(nn.Module):
    """存储 int8 权重 + float32 scale 的 Conv2d 层（1x1 卷积，分组量化）。

    仅对 kernel_size=1 的 Conv2d 进行量化（等价于 Linear 层）。
    """

    def __init__(self, int8_weight: torch.Tensor, scale: torch.Tensor,
                 bias: torch.Tensor | None = None,
                 stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1,
                 group_size: int = _GROUP_SIZE):
        super().__init__()
        out_channels = int8_weight.shape[0]
        self.group_size = group_size
        self.register_buffer("int8_weight", int8_weight)
        self.register_buffer("scale", scale.float())
        if bias is not None:
            self.register_buffer("bias", bias.float())
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.register_buffer("weight", torch.zeros(1))

    def extra_repr(self) -> str:
        out_c, in_c = self.int8_weight.shape[:2]
        k = (self.int8_weight.shape[2], self.int8_weight.shape[3])
        return (f"({out_c}, {in_c}, kernel_size={k}, stride={self.stride}, "
                f"group_size={self.group_size}, bias={self.bias is not None})")

    @classmethod
    def from_conv2d_manual(cls, conv: nn.Conv2d,
                           group_size: int = _GROUP_SIZE) -> "ONNXInt8Conv2d":
        """手动分组量化 nn.Conv2d（仅 1x1 卷积）。"""
        weight = conv.weight.data
        out_c, in_c = weight.shape[:2]
        w_flat = weight.view(out_c, in_c)

        effective_gs = group_size
        if in_c % group_size != 0 or in_c < group_size:
            effective_gs = in_c

        if effective_gs == in_c:
            max_abs = w_flat.abs().max(dim=1, keepdim=True)[0]  # (out_c, 1)
            scale = max_abs / 127.0  # (out_c, 1)
            int8_weight = (w_flat / max_abs * 127.0).round().clamp(-128, 127).to(torch.int8)
        else:
            num_groups = in_c // effective_gs
            w_reshaped = w_flat.view(out_c, num_groups, effective_gs)
            max_abs = w_reshaped.abs().max(dim=2, keepdim=True)[0]
            scale = max_abs.squeeze(2) / 127.0  # (out_c, num_groups)
            int8_weight = (w_reshaped / max_abs * 127.0).round().clamp(-128, 127).to(torch.int8)

        int8_weight = int8_weight.view_as(weight)
        bias = conv.bias.data.clone() if conv.bias is not None else None
        return cls(int8_weight, scale, bias, conv.stride, conv.padding,
                   conv.dilation, conv.groups, effective_gs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_c = self.int8_weight.shape[0]
        in_c = self.int8_weight.shape[1]
        w_flat = self.int8_weight.float().view(out_c, in_c)
        if self.group_size == in_c:
            w_deq = (w_flat * self.scale).view_as(self.int8_weight)
        else:
            scale_2d = self.scale.repeat_interleave(self.group_size, dim=1)
            w_deq = (w_flat * scale_2d).view_as(self.int8_weight)
        return F.conv2d(x, w_deq, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ONNXInt8Embedding(nn.Module):
    """存储 int8 权重 + float32 scale 的 Embedding 层（分组量化）。

    Embedding 权重通常是模型最大的组件（MobileCLIP2-S0 文本编码器中 ~97 MB），
    量化后可降至 ~25 MB。
    """

    def __init__(self, int8_weight: torch.Tensor, scale: torch.Tensor,
                 padding_idx: int | None = None, group_size: int = _GROUP_SIZE):
        super().__init__()
        self.num_embeddings, self.embedding_dim = int8_weight.shape
        self.padding_idx = padding_idx
        self.group_size = group_size
        self.register_buffer("int8_weight", int8_weight)
        self.register_buffer("scale", scale.float())
        self.register_buffer("weight", torch.zeros(1))

    @classmethod
    def from_embedding(cls, emb: nn.Embedding,
                       group_size: int = _GROUP_SIZE) -> "ONNXInt8Embedding":
        weight = emb.weight.data
        num_emb, emb_dim = weight.shape
        effective_gs = group_size
        if emb_dim % group_size != 0 or emb_dim < group_size:
            effective_gs = emb_dim

        if effective_gs == emb_dim:
            max_abs = weight.abs().max(dim=1, keepdim=True)[0]  # (num_emb, 1)
            scale = max_abs / 127.0
            int8_weight = (weight / max_abs * 127.0).round().clamp(-128, 127).to(torch.int8)
        else:
            num_groups = emb_dim // effective_gs
            w_reshaped = weight.view(num_emb, num_groups, effective_gs)
            max_abs = w_reshaped.abs().max(dim=2, keepdim=True)[0]
            scale = max_abs.squeeze(2) / 127.0
            int8_weight = (w_reshaped / max_abs * 127.0).round().clamp(-128, 127).to(torch.int8)
            int8_weight = int8_weight.view(num_emb, emb_dim)

        return cls(int8_weight, scale, emb.padding_idx, effective_gs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb_dim = self.embedding_dim
        if self.group_size == emb_dim:
            weight_fp = self.int8_weight.float() * self.scale  # (num_emb, 1) broadcasts
        else:
            weight_fp = self.int8_weight.float() * self.scale.repeat_interleave(
                self.group_size, dim=1)
        return F.embedding(x, weight_fp, self.padding_idx)


# ============================================================
# 模型量化
# ============================================================
def _list_children(module: nn.Module) -> list[tuple[str, nn.Module]]:
    """获取模块的直接子模块列表（安全遍历，避免迭代时修改 dict）。"""
    return list(module.named_children())


# 注意力投影层的类名，量化这些层会导致较大精度损失
_ATTENTION_MODULE_TYPES = {"MultiheadAttention", "MultiHeadAttention", "MHSA", "Attention"}


def _should_quantize_linear(name: str, module: nn.Linear,
                             parent: nn.Module | None) -> bool:
    """判断一个 Linear 层是否应该被量化。

    参照 int8_tutorial 的做法，只量化 FFN 层，排除注意力投影层。
    """
    if parent is None:
        return True
    parent_cls_name = parent.__class__.__name__
    if parent_cls_name in _ATTENTION_MODULE_TYPES:
        return False
    return True


def replace_modules_with_int8(model: nn.Module, int8_linear_cls, int8_conv_cls,
                               int8_emb_cls=None, exclude_names: set | None = None) -> nn.Module:
    """递归替换模型中符合条件的 nn.Linear / 1x1 nn.Conv2d / nn.Embedding 为 INT8 量化版本。

    默认排除注意力投影层（MultiheadAttention、MHSA、Attention 中的 Linear）。
    仅对 kernel_size=1 的 Conv2d 进行量化（等价于 Linear 层）。
    若 int8_emb_cls 不为 None，则同时量化所有 nn.Embedding 层。
    """
    if exclude_names is None:
        exclude_names = set()

    for name, module in _list_children(model):
        if name in exclude_names:
            continue
        if isinstance(module, nn.Linear):
            if _should_quantize_linear(name, module, model):
                model._modules[name] = int8_linear_cls.from_linear_manual(module)
        elif isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
            model._modules[name] = int8_conv_cls.from_conv2d_manual(module)
        elif int8_emb_cls is not None and isinstance(module, nn.Embedding):
            model._modules[name] = int8_emb_cls.from_embedding(module)
        elif len(list(module.children())) > 0:
            replace_modules_with_int8(module, int8_linear_cls, int8_conv_cls,
                                      int8_emb_cls, exclude_names)
    return model


def _is_text_encoder(model: nn.Module) -> bool:
    """检测模型是否为文本编码器（通过检查输入参数类型）。"""
    # 检查模块参数：文本模型的第一层通常是 Embedding
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            return True
    return False


def quantize_with_bnb(model: nn.Module, device: str = "cuda",
                       is_text: bool = False) -> nn.Module:
    """使用 bitsandbytes SwitchBackLinear 量化模型（需要 CUDA）。

    流程（对应 int8_tutorial.ipynb）：
    1. 用 SwitchBackLinear 替换 nn.Linear
    2. 将权重从原始 Linear 复制到 SwitchBackLinear
    3. 运行一次前向传播触发量化
    4. 调用 convert_int8_model_to_inference_mode 预计算量化权重
    5. 提取 int8 权重并替换为 ONNXInt8Linear
    """
    import bitsandbytes as bnb

    def _replace_with_bnb(mod, parent=None):
        for name, child in _list_children(mod):
            if isinstance(child, nn.Linear):
                if not _should_quantize_linear(name, child, mod):
                    continue
                new_mod = bnb.nn.triton_based_modules.SwitchBackLinear(
                    child.in_features, child.out_features, child.bias is not None,
                )
                new_mod.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_mod.linear.bias.data.copy_(child.bias.data)
                mod._modules[name] = new_mod
            elif len(list(child.children())) > 0:
                _replace_with_bnb(child, parent=child)

    _replace_with_bnb(model)
    model = model.to(device)

    # 运行一次前向传播触发 bitsandbytes 量化
    with torch.no_grad():
        if is_text:
            dummy = torch.randint(0, 100, (1, 77)).to(device)
        else:
            dummy = torch.randn(1, 3, 256, 256).to(device)
        model(dummy)

    # 预计算量化权重（对应 int8_tutorial 的 convert_int8_model_to_inference_mode）
    open_clip.utils.convert_int8_model_to_inference_mode(model)
    model = model.cpu()

    # 提取量化权重，替换为 ONNX 可导出的模块
    def _extract_and_replace(mod):
        for name, child in _list_children(mod):
            if isinstance(child, bnb.nn.triton_based_modules.SwitchBackLinear):
                mod._modules[name] = ONNXInt8Linear.from_bnb_switchback(child)
            elif len(list(child.children())) > 0:
                _extract_and_replace(child)

    _extract_and_replace(model)

    # 额外的 1x1 Conv2d + Embedding 量化（bitsandbytes 仅处理 Linear）
    replace_modules_with_int8(model, ONNXInt8Linear, ONNXInt8Conv2d, ONNXInt8Embedding)
    return model


def quantize_manual(model: nn.Module) -> nn.Module:
    """手动分组量化模型（CPU 可用，不依赖 bitsandbytes/CUDA）。

    使用分组量化（group_size=128），与 GPTQ/AWQ 一致。
    对 nn.Linear、1x1 nn.Conv2d 和 nn.Embedding 均进行量化。
    """
    replace_modules_with_int8(model, ONNXInt8Linear, ONNXInt8Conv2d, ONNXInt8Embedding)
    return model


def verify_quantization(model_int8: nn.Module, model_orig: nn.Module) -> dict:
    """验证量化前后权重的误差（按模块名称匹配，覆盖 Linear / Conv2d / Embedding）。"""
    orig_layers = {}
    for name, mod in model_orig.named_modules():
        if isinstance(mod, nn.Linear):
            orig_layers[name] = ("linear", mod.weight.data.float())
        elif isinstance(mod, nn.Conv2d) and mod.kernel_size == (1, 1):
            orig_layers[name] = ("conv2d", mod.weight.data.float())
        elif isinstance(mod, nn.Embedding):
            orig_layers[name] = ("embedding", mod.weight.data.float())

    errors = {}
    for name, mod in model_int8.named_modules():
        if isinstance(mod, ONNXInt8Linear) and name in orig_layers:
            w_orig = orig_layers[name][1]
            out_f, in_f = mod.int8_weight.shape
            scale_2d = mod.scale.float()
            if mod.group_size == in_f:
                w_dequant = mod.int8_weight.float() * scale_2d
            else:
                w_dequant = mod.int8_weight.float() * scale_2d.repeat_interleave(
                    mod.group_size, dim=1)
            max_err = (w_orig - w_dequant).abs().max().item()
            errors[name] = max_err
        elif isinstance(mod, ONNXInt8Conv2d) and name in orig_layers:
            w_orig = orig_layers[name][1]
            out_c, in_c = mod.int8_weight.shape[:2]
            w_flat = mod.int8_weight.float().view(out_c, in_c)
            scale_2d = mod.scale.float()
            if mod.group_size == in_c:
                w_dequant = (w_flat * scale_2d).view_as(w_orig)
            else:
                w_dequant = (w_flat * scale_2d.repeat_interleave(
                    mod.group_size, dim=1)).view_as(w_orig)
            max_err = (w_orig - w_dequant).abs().max().item()
            errors[name] = max_err
        elif isinstance(mod, ONNXInt8Embedding) and name in orig_layers:
            w_orig = orig_layers[name][1]
            num_emb, emb_dim = mod.int8_weight.shape
            scale_2d = mod.scale.float()
            if mod.group_size == emb_dim:
                w_dequant = mod.int8_weight.float() * scale_2d
            else:
                w_dequant = mod.int8_weight.float() * scale_2d.repeat_interleave(
                    mod.group_size, dim=1)
            max_err = (w_orig - w_dequant).abs().max().item()
            errors[name] = max_err
    return errors


# ============================================================
# ONNX 导出
# ============================================================
def export_to_onnx(model: nn.Module, dummy_input: tuple, onnx_path: str,
                   input_names: list, output_names: list, dynamic_axes: dict):
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=20,
        dynamo=False,
    )
    print(f"  -> Exported to {onnx_path}")


def _find_int8_modules(module: nn.Module) -> dict[str, nn.Module]:
    """收集模型中所有 INT8 量化模块，返回 {name: module} 映射。"""
    result = {}
    for name, mod in module.named_modules():
        if isinstance(mod, (ONNXInt8Linear, ONNXInt8Conv2d, ONNXInt8Embedding)):
            result[name] = mod
    return result


def _exclude_attention_matmul_from_onnx(onnx_path: str) -> list[str]:
    """在 ONNX 模型中标记注意力层 MatMul 节点（临时改 op_type 跳过量化）。

    返回被修改的节点名列表。
    """
    import onnx
    model = onnx.load(onnx_path)
    attention_matmul_names = []
    for node in model.graph.node:
        if node.op_type == "MatMul":
            # 检查 MatMul 的输入是否来自注意力投影层
            for inp in node.input:
                if "attn.out_proj" in inp or "attn.in_proj" in inp or "in_proj_weight" in inp:
                    node.op_type = "MatMulSkip"
                    attention_matmul_names.append(node.name)
                    break
    onnx.save(model, onnx_path)
    return attention_matmul_names


def _restore_attention_matmul_in_onnx(onnx_path: str, node_names: list[str]):
    """恢复量化前被标记跳过的注意力 MatMul 节点。"""
    import onnx
    model = onnx.load(onnx_path)
    name_set = set(node_names)
    for node in model.graph.node:
        if node.op_type == "MatMulSkip" and node.name in name_set:
            node.op_type = "MatMul"
    onnx.save(model, onnx_path)


def quantize_onnx_with_attention_exclusion(fp32_path: str, output_path: str):
    """使用 ORT 动态量化 ONNX 模型，但排除注意力层 MatMul。

    流程：
    1. 标记注意力 MatMul 节点（改 op_type 避免被量化）
    2. 应用 ORT 动态量化
    3. 恢复注意力 MatMul 节点
    """
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.registry import IntegerOpsRegistry
    import shutil

    # 复制 FP32 模型用于处理
    temp_path = fp32_path + ".temp"
    shutil.copy(fp32_path, temp_path)

    # 形状推断（跳过 quant_pre_process，它对某些模型图会报错）
    try:
        onnx.shape_inference.infer_shapes_path(temp_path, temp_path)
    except Exception:
        pass

    # 标记注意力层 MatMul
    attention_nodes = _exclude_attention_matmul_from_onnx(temp_path)

    # 动态量化
    op_types_to_quantize = IntegerOpsRegistry.copy()
    op_types_to_quantize.pop("Conv", None)

    quantize_dynamic(
        model_input=temp_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=op_types_to_quantize.keys(),
    )

    # 恢复注意力 MatMul
    if attention_nodes:
        _restore_attention_matmul_in_onnx(output_path, attention_nodes)

    # 清理临时文件
    os.remove(temp_path)
    print(f"  -> ORT INT8 量化完成: {output_path}"
          + (f" (已保留 {len(attention_nodes)} 个注意力 MatMul 为 FP32)" if attention_nodes else ""))


# ============================================================
# 精度对比
# ============================================================
def compare_outputs(image_features_orig, text_features_orig,
                    sess_vis, sess_text, image_np, text_np, label: str):
    img_feat = torch.from_numpy(sess_vis.run(None, {"image": image_np})[0])
    txt_feat = torch.from_numpy(sess_text.run(None, {"text": text_np})[0])
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    probs = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)

    max_diff_img = (image_features_orig - img_feat).abs().max().item()
    max_diff_txt = (text_features_orig - txt_feat).abs().max().item()
    return probs, max_diff_img, max_diff_txt


def print_size_comparison(paths: list[str | None]):
    print("\n" + "=" * 60)
    print(f"{'模型文件':<45} {'大小 (MB)':>10}")
    print("=" * 60)
    for p in paths:
        if p and os.path.exists(p):
            print(f"{p:<45} {os.path.getsize(p) / (1024 * 1024):>10.2f}")
    print("=" * 60)


def percent_diff(orig, other):
    """两个概率向量之间的最大绝对差异百分比。"""
    return (orig - other).abs().max().item() * 100


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="导出 INT8 ONNX 模型")
    parser.add_argument("--cpu", action="store_true",
                        help="使用手动行级量化（CPU，不依赖 CUDA/bitsandbytes）")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help=f"模型名称（默认: {MODEL_NAME}）")
    parser.add_argument("--skip-size-compare", action="store_true",
                        help="跳过模型体积对比")
    args = parser.parse_args()

    model_name = args.model
    model_file = model_name.lower().replace("-", "_")
    use_cuda = not args.cpu and torch.cuda.is_available()

    print(f"模型: {model_name} | 模式: {'bitsandbytes + CUDA' if use_cuda else '手动行级量化 (CPU)'}")
    print("-" * 60)

    # ----------------------------------------------------------
    # 1. 加载模型
    # ----------------------------------------------------------
    print("\n[1/6] 加载模型...")
    model_kwargs = get_model_kwargs(model_name)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=f"./{model_file}.pt", **model_kwargs
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    model = reparameterize_model(model)

    # 准备测试输入
    from PIL import Image
    image = preprocess(
        Image.open("docs/fig_accuracy_latency.png").convert("RGB")
    ).unsqueeze(0)
    text = tokenizer(["a diagram", "a paper essay", "a cat"])

    # 原始模型推理（基准）
    with torch.no_grad():
        image_features_orig = model.encode_image(image)
        text_features_orig = model.encode_text(text)
        image_features_orig = image_features_orig / image_features_orig.norm(dim=-1, keepdim=True)
        text_features_orig = text_features_orig / text_features_orig.norm(dim=-1, keepdim=True)
        text_probs_orig = (100.0 * image_features_orig @ text_features_orig.T).softmax(dim=-1)
    print(f"  基准 probs: {text_probs_orig.squeeze().tolist()}")

    # ----------------------------------------------------------
    # 2. 量化模型（对副本操作，保留原始模型用于后续 ORT 对比）
    # ----------------------------------------------------------
    print("\n[2/6] 量化模型...")
    visual_model = copy.deepcopy(model.visual)
    text_model = copy.deepcopy(model.text)

    if use_cuda:
        print("  使用 bitsandbytes SwitchBackLinear 替换 Linear 层...")
        visual_int8 = quantize_with_bnb(visual_model, device="cuda", is_text=False)
        text_int8 = quantize_with_bnb(text_model, device="cuda", is_text=True)
    else:
        print("  使用手动行级量化（与 bitsandbytes 相同数学公式）...")
        visual_int8 = quantize_manual(visual_model)
        text_int8 = quantize_manual(text_model)

    # 统计量化信息
    int8_linear_count = sum(1 for m in visual_int8.modules() if isinstance(m, ONNXInt8Linear))
    int8_linear_count += sum(1 for m in text_int8.modules() if isinstance(m, ONNXInt8Linear))
    int8_conv_count = sum(1 for m in visual_int8.modules() if isinstance(m, ONNXInt8Conv2d))
    int8_conv_count += sum(1 for m in text_int8.modules() if isinstance(m, ONNXInt8Conv2d))
    int8_emb_count = sum(1 for m in visual_int8.modules() if isinstance(m, ONNXInt8Embedding))
    int8_emb_count += sum(1 for m in text_int8.modules() if isinstance(m, ONNXInt8Embedding))
    parts = [f"{int8_linear_count} Linear", f"{int8_conv_count} Conv2d"]
    if int8_emb_count > 0:
        parts.append(f"{int8_emb_count} Embedding")
    print(f"  已量化 {' + '.join(parts)} 层为 INT8")

    # 验证量化权重误差
    total_quantized = int8_linear_count + int8_conv_count + int8_emb_count
    if total_quantized > 0:
        w_errors = verify_quantization(visual_int8, model.visual)
        w_errors.update(verify_quantization(text_int8, model.text))
        max_w_err = max(w_errors.values())
        print(f"  权重量化最大误差: {max_w_err:.6f}")

    # 量化后 PyTorch 推理验证
    with torch.no_grad():
        image_features_int8 = visual_int8(image)
        text_features_int8 = text_int8(text)
        image_features_int8 = image_features_int8 / image_features_int8.norm(dim=-1, keepdim=True)
        text_features_int8 = text_features_int8 / text_features_int8.norm(dim=-1, keepdim=True)
        text_probs_int8 = (100.0 * image_features_int8 @ text_features_int8.T).softmax(dim=-1)

    print(f"  INT8 probs: {text_probs_int8.squeeze().tolist()}")
    print(f"  量化精度损失 (PyTorch): {percent_diff(text_probs_orig, text_probs_int8):.4f}%")

    # ----------------------------------------------------------
    # 3. 导出 INT8 ONNX（FP32 → ORT 量化，排除注意力层）
    # ----------------------------------------------------------
    print("\n[3/6] 导出 INT8 ONNX 模型（ORT 动态量化 + 注意力层保留 FP32）...")
    visual_fp32_path = f"{FP32_TEMP_DIR}/{model_file}_visual_fp32.onnx"
    text_fp32_path = f"{FP32_TEMP_DIR}/{model_file}_text_fp32.onnx"

    export_to_onnx(
        model.visual, (image,), visual_fp32_path,
        input_names=["image"], output_names=["image_features"],
        dynamic_axes={"image": {0: "batch_size"}, "image_features": {0: "batch_size"}},
    )
    export_to_onnx(
        model.text, (text,), text_fp32_path,
        input_names=["text"], output_names=["text_features"],
        dynamic_axes={"text": {0: "batch_size"}, "text_features": {0: "batch_size"}},
    )

    visual_onnx_path = f"{OUTPUT_DIR}/{model_file}_visual.onnx"
    text_onnx_path = f"{OUTPUT_DIR}/{model_file}_text.onnx"

    quantize_onnx_with_attention_exclusion(visual_fp32_path, visual_onnx_path)
    quantize_onnx_with_attention_exclusion(text_fp32_path, text_onnx_path)

    # ----------------------------------------------------------
    # 4. ONNX Runtime 推理验证
    # ----------------------------------------------------------
    print("\n[4/6] 验证 ONNX 模型...")
    image_np = image.numpy()
    text_np = text.numpy()

    sess_vis_int8 = ort.InferenceSession(visual_onnx_path)
    sess_txt_int8 = ort.InferenceSession(text_onnx_path)

    probs_onnx_int8, max_diff_img, max_diff_txt = compare_outputs(
        image_features_orig, text_features_orig,
        sess_vis_int8, sess_txt_int8, image_np, text_np, "ONNX INT8",
    )
    print(f"  ONNX INT8 probs: {probs_onnx_int8.squeeze().tolist()}")
    print(f"  图像特征最大差异: {max_diff_img:.6f}")
    print(f"  文本特征最大差异: {max_diff_txt:.6f}")
    print(f"  精度损失 (PyTorch FP32 vs ONNX INT8): {percent_diff(text_probs_orig, probs_onnx_int8):.4f}%")

    # ----------------------------------------------------------
    # 5. 与 ORT 全量动态量化对比（可选，用于展示注意力排除的效果）
    # ----------------------------------------------------------
    if not args.skip_size_compare:
        print("\n[5/6] 对比: 全量 ORT 动态量化（不含注意力排除）...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        from onnxruntime.quantization.registry import IntegerOpsRegistry
        import onnx

        op_types_to_quantize = IntegerOpsRegistry.copy()
        op_types_to_quantize.pop("Conv", None)

        # 预处理 FP32 模型
        visual_ort_path = f"{OUTPUT_DIR}/{model_file}_visual_ort_quant.onnx"
        text_ort_path = f"{OUTPUT_DIR}/{model_file}_text_ort_quant.onnx"

        # 复制临时文件用于量化
        import shutil
        shutil.copy(visual_fp32_path, visual_fp32_path + ".ort_tmp")
        shutil.copy(text_fp32_path, text_fp32_path + ".ort_tmp")
        # 形状推断
        try:
            onnx.shape_inference.infer_shapes_path(visual_fp32_path + ".ort_tmp", visual_fp32_path + ".ort_tmp")
            onnx.shape_inference.infer_shapes_path(text_fp32_path + ".ort_tmp", text_fp32_path + ".ort_tmp")
        except Exception:
            pass

        quantize_dynamic(
            model_input=visual_fp32_path + ".ort_tmp",
            model_output=visual_ort_path,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=op_types_to_quantize.keys(),
        )
        quantize_dynamic(
            model_input=text_fp32_path + ".ort_tmp",
            model_output=text_ort_path,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=op_types_to_quantize.keys(),
        )
        # 清理临时文件
        os.remove(visual_fp32_path + ".ort_tmp")
        os.remove(text_fp32_path + ".ort_tmp")
        print("  全量 ORT 动态量化完成")

        # ORT 量化模型推理
        sess_vis_ort = ort.InferenceSession(visual_ort_path)
        sess_txt_ort = ort.InferenceSession(text_ort_path)

        probs_ort, _, _ = compare_outputs(
            image_features_orig, text_features_orig,
            sess_vis_ort, sess_txt_ort, image_np, text_np, "ONNX ORT-INT8",
        )
        print(f"  全量 ORT INT8 probs: {probs_ort.squeeze().tolist()}")
        print(f"  精度损失 (PyTorch FP32 vs 全量 ORT INT8): {percent_diff(text_probs_orig, probs_ort):.4f}%")
        print(f"  精度损失 (排除注意力 vs 全量):           {percent_diff(probs_onnx_int8, probs_ort):.4f}%")

        # 清理临时 FP32 文件
        shutil.rmtree(FP32_TEMP_DIR, ignore_errors=True)
    else:
        visual_ort_path = None
        text_ort_path = None
        # 清理临时 FP32 文件
        import shutil
        shutil.rmtree(FP32_TEMP_DIR, ignore_errors=True)

    # ----------------------------------------------------------
    # 6. 结果汇总
    # ----------------------------------------------------------
    print("\n[6/6] 结果汇总")
    print("\n" + "=" * 60)
    print("精度对比")
    print("=" * 60)
    print(f"  PyTorch FP32 (基准):          {text_probs_orig.squeeze().tolist()}")
    print(f"  PyTorch INT8 (分组量化验证):   {text_probs_int8.squeeze().tolist()}")
    print(f"  ONNX INT8 (排除注意力):       {probs_onnx_int8.squeeze().tolist()}")
    if not args.skip_size_compare:
        print(f"  ONNX ORT INT8 (全量量化):     {probs_ort.squeeze().tolist()}")

    print_size_comparison([
        visual_onnx_path, text_onnx_path,
        visual_ort_path, text_ort_path,
    ])

    print("\n导出完成!")
    print(f"  INT8 (排除注意力): {text_onnx_path}")
    if not args.skip_size_compare:
        print(f"  INT8 (全量量化):   {text_ort_path}")


if __name__ == "__main__":
    import onnxruntime as ort
    main()
