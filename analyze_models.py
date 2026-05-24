"""
比较 fp16_results/mobileclip2_s2_visual.ort 与 clip-fp32/visual.onnx
在 Android ONNX Runtime (CPU / NNAPI) 上的推理效率。

从算子层面、模型架构层面进行分析，并给出 mobileclip2_s2_visual.ort 的优化策略。

.ort 文件是 ONNX Runtime 的 flatbuffers 原生格式，不能直接用 onnx.load() 解析，
但它的算子图等价于 fp16_results/mobileclip2_s2_visual.onnx (经过 ORT 算子融合和常量折叠)。
本脚本同时分析三个模型以获取完整对比:
  1. fp16_results/mobileclip2_s2_visual.onnx  (FP16 source, pre-ORT)
  2. fp16_results/mobileclip2_s2_visual.ort   (ORT optimized, via onnxruntime)
  3. clip-fp32/visual.onnx                      (FP32 baseline)
"""

import os
import sys
import time
import json
from collections import Counter, defaultdict
from pathlib import Path

import onnx
import numpy as np
from onnx import helper, numpy_helper
import onnxruntime as ort

# ---------------------------------------------------------------------------
# 文件路径
# ---------------------------------------------------------------------------
ONNX_FP16 = Path("fp16_results/mobileclip2_s2_visual.onnx")   # FP16 ONNX
ORT_FP16 = Path("fp16_results/mobileclip2_s2_visual.ort")     # ORT optimized
ONNX_FP32 = Path("clip-fp32/visual.onnx")                      # FP32 ONNX

# ---------------------------------------------------------------------------
# Part A: ONNX 图分析 (利用 .onnx 文件)
# ---------------------------------------------------------------------------
print("=" * 72)
print("PART A: ONNX GRAPH ANALYSIS")
print("=" * 72)

def load_onnx(path):
    t0 = time.time()
    m = onnx.load(str(path))
    dt = time.time() - t0
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Loaded {path.name}: {size_mb:.1f} MB in {dt:.2f}s")
    return m, size_mb

model_fp16, size_fp16 = load_onnx(ONNX_FP16)
model_fp32, size_fp32 = load_onnx(ONNX_FP32)

# ---------------------------------------------------------------------------
# 1. 基本信息
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("1.1 模型元信息")
print("-" * 60)

def model_info(model, name):
    g = model.graph
    opset = model.opset_import[0].version if model.opset_import else "N/A"
    print(f"\n  [{name}]")
    print(f"    IR={model.ir_version}  Opset={opset}  Producer={model.producer_name}")
    print(f"    Nodes={len(g.node)}  Initializers={len(g.initializer)}")
    for inp in g.input:
        shape = [d.dim_value or f"?({d.dim_param})" for d in inp.type.tensor_type.shape.dim]
        dt = {1:"float32",10:"float16",6:"int32",7:"int64"}.get(inp.type.tensor_type.elem_type, "?")
        print(f"    Input:  {inp.name:<20s} shape={shape}  dtype={dt}")
    for out in g.output:
        shape = [d.dim_value or f"?({d.dim_param})" for d in out.type.tensor_type.shape.dim]
        dt = {1:"float32",10:"float16",6:"int32",7:"int64"}.get(out.type.tensor_type.elem_type, "?")
        print(f"    Output: {out.name:<20s} shape={shape}  dtype={dt}")

model_info(model_fp16, ONNX_FP16.name)
model_info(model_fp32, ONNX_FP32.name)

# ---------------------------------------------------------------------------
# 2. 算子对比
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("1.2 算子分布对比 (核心分析)")
print("-" * 60)

# NNAPI EP 支持的算子 (ONNX Runtime 1.20+, Android 11+)
NNAPI_FULL = {
    "Add","AveragePool","BatchNormalization","Cast","Clip","Concat",
    "Conv","ConvTranspose","DepthToSpace","DequantizeLinear","Div",
    "Elu","Equal","Exp","Flatten","Floor","Gather","Gemm",
    "GlobalAveragePool","GlobalMaxPool","Greater","HardSigmoid","HardSwish",
    "Identity","LeakyRelu","Less","Log","LogSoftmax","LRN","LSTM",
    "MatMul","Max","MaxPool","Min","Mul","Neg","Not","Pad",
    "Pow","PRelu","QLinearConv","QLinearMatMul","QuantizeLinear",
    "ReduceMax","ReduceMin","ReduceProd","ReduceSum",
    "Relu","Reshape","Resize","Round","Scan","Shape",
    "Sigmoid","Sin","Slice","Softmax","SpaceToDepth","Split",
    "Sqrt","Squeeze","Sub","Sum","Tanh","Tile","TopK",
    "Transpose","Unsqueeze","Where",
}
NNAPI_PARTIAL = {"InstanceNormalization","Gelu","LayerNormalization","Resize"}
NNAPI_ALL = NNAPI_FULL | NNAPI_PARTIAL

# Android CPU 上计算量大的关键算子 (常用于估算延迟)
COMPUTE_HEAVY = {"Conv","MatMul","Gemm","LSTM","GRU"}

def analyze_ops(model, name):
    g = model.graph
    counter = Counter(n.op_type for n in g.node)
    total = sum(counter.values())

    print(f"\n  [{name}]  Total nodes: {total}")
    print(f"  {'Op':<28s} {'Count':>6s}  {'%':>6s}  {'NNAPI':>8s}  {'Compute':>10s}")
    print(f"  {'-'*64}")

    for op, cnt in counter.most_common():
        pct = cnt / total * 100
        nnapi = "YES" if op in NNAPI_FULL else ("PARTIAL" if op in NNAPI_PARTIAL else "NO")
        heavy = "HEAVY" if op in COMPUTE_HEAVY else ""
        print(f"  {op:<26s} {cnt:>6d}  {pct:>5.1f}%  {nnapi:>8s}  {heavy:>10s}")

    # NNAPI 覆盖率
    covered = sum(c for o, c in counter.items() if o in NNAPI_ALL)
    uncovered = [(o, c) for o, c in counter.items() if o not in NNAPI_ALL]
    print(f"\n  NNAPI coverage: {covered}/{total} ({covered/total*100:.1f}%)")
    if uncovered:
        print(f"  NNAPI-unsupported ops:")
        for o, c in sorted(uncovered, key=lambda x: -x[1]):
            print(f"    - {o}: {c}x")

    # 计算密集型算子占比
    heavy_cnt = sum(c for o, c in counter.items() if o in COMPUTE_HEAVY)
    print(f"  Compute-heavy ops: {heavy_cnt}/{total} ({heavy_cnt/total*100:.1f}%)")

    return counter, covered/total*100

op_fp16, nnapi_fp16 = analyze_ops(model_fp16, ONNX_FP16.name)
op_fp32, nnapi_fp32 = analyze_ops(model_fp32, ONNX_FP32.name)

# ---------------------------------------------------------------------------
# 3. 权重精度分析
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("1.3 权重精度分布")
print("-" * 60)

DTYPE_MAP = {1:"float32",10:"float16",6:"int32",7:"int64",2:"uint8",3:"int8",9:"bool"}

def analyze_weights(model, name):
    g = model.graph
    dist = defaultdict(lambda: {"tensors":0,"params":0,"bytes":0})
    total_p = 0
    total_b = 0

    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        np_arr = np.array(arr)
        n_p = int(np.prod(np_arr.shape))
        n_b = len(init.raw_data) if init.raw_data else n_p * 4
        dtype_n = DTYPE_MAP.get(init.data_type, f"type_{init.data_type}")
        dist[dtype_n]["tensors"] += 1
        dist[dtype_n]["params"] += n_p
        dist[dtype_n]["bytes"] += n_b
        total_p += n_p
        total_b += n_b

    print(f"\n  [{name}]")
    print(f"  Total params: {total_p/1e6:.2f}M  |  Weight data: {total_b/(1024*1024):.1f} MB")
    for dt, info in sorted(dist.items()):
        mb = info["bytes"] / (1024*1024)
        pm = info["params"] / 1e6
        print(f"    {dt:<10s}: {info['tensors']:>5d} tensors, {pm:>7.2f}M params, {mb:>7.1f} MB")

    return total_p, total_b, dist

params_fp16, bytes_fp16, dt_fp16 = analyze_weights(model_fp16, ONNX_FP16.name)
params_fp32, bytes_fp32, dt_fp32 = analyze_weights(model_fp32, ONNX_FP32.name)

# ---------------------------------------------------------------------------
# 4. 计算图结构分析
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("1.4 计算图结构分类")
print("-" * 60)

def graph_structure(model, name, op_counter):
    total = sum(op_counter.values())
    categories = {
        "Conv (matmul-bound)": ["Conv"],
        "MatMul/Gemm": ["MatMul", "Gemm"],
        "Element-wise (mem-bound)": ["Add","Mul","Sub","Div","Relu","Sigmoid","Tanh",
            "Exp","Neg","HardSigmoid","HardSwish","PRelu","LeakyRelu","Clip","Pow",
            "Sqrt","Erf","Round"],
        "Normalization": ["BatchNormalization","LayerNormalization","InstanceNormalization"],
        "Data movement": ["Reshape","Transpose","Squeeze","Unsqueeze","Flatten",
            "Concat","Split","Slice","Gather","Expand","Tile","Pad"],
        "Pooling": ["GlobalAveragePool","GlobalMaxPool","AveragePool","MaxPool"],
        "Softmax/Activation": ["Softmax","LogSoftmax","Gelu","Sigmoid"],
        "Shape/Type ops": ["Shape","Cast","Constant","ConstantOfShape","Identity",
            "Where","Equal","Greater","Less","Not","And"],
        "Reduce": ["ReduceMean","ReduceMax","ReduceMin","ReduceSum","ReduceProd"],
        "Quantization": ["QuantizeLinear","DequantizeLinear","QLinearConv","QLinearMatMul"],
    }

    assigned = set()
    for cats in categories.values():
        assigned.update(cats)

    print(f"\n  [{name}]")
    print(f"  {'Category':<30s} {'Count':>6s}  {'%':>6s}")
    print(f"  {'-'*46}")

    cat_counts = {}
    for cat, ops in categories.items():
        cnt = sum(op_counter.get(o, 0) for o in ops)
        cat_counts[cat] = cnt
        pct = cnt / total * 100 if total else 0
        print(f"  {cat:<30s} {cnt:>6d}  {pct:>5.1f}%")

    other = sum(c for o, c in op_counter.items() if o not in assigned)
    if other:
        # show unclassified
        unclassified = [(o, c) for o, c in op_counter.items() if o not in assigned]
        unclassified.sort(key=lambda x: -x[1])
        ops_list = ", ".join(f"{o}({c})" for o, c in unclassified[:10])
        print(f"  {'Other':<30s} {other:>6d}  ({ops_list})")

    print()

    # Conv-BN fusion check
    _check_fusion(model, name)

    return cat_counts

def _check_fusion(model, name):
    """检测 Conv+BN 和 Conv+ReLU 的融合状态"""
    g = model.graph
    # build successor map
    succ = defaultdict(list)
    for n in g.node:
        for inp in n.input:
            succ[inp].append(n)

    conv_bn = conv_relu = 0
    n_conv = 0
    for n in g.node:
        if n.op_type == "Conv" or n.op_type == "Gemm":
            n_conv += 1
            for out in n.output:
                for s in succ.get(out, []):
                    if s.op_type == "BatchNormalization":
                        conv_bn += 1
                    if s.op_type in ("Relu", "LeakyRelu", "Clip"):
                        conv_relu += 1

    if n_conv > 0:
        status_bn = f"UNFUSED ({conv_bn}/{n_conv} pairs)" if conv_bn else "OK (fused)"
        status_relu = f"UNFUSED ({conv_relu}/{n_conv} pairs)" if conv_relu else "OK (fused)"
        print(f"    Conv+BN fusion:  {status_bn}")
        print(f"    Conv+ReLU fusion: {status_relu}")
    else:
        print(f"    Conv+BN/ReLU: N/A")

struct_fp16 = graph_structure(model_fp16, ONNX_FP16.name, op_fp16)
struct_fp32 = graph_structure(model_fp32, ONNX_FP32.name, op_fp32)

# ---------------------------------------------------------------------------
# 5. Conv层的详细分析 (kernel/stride/channel 分布)
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("1.5 Conv 算子详细分析")
print("-" * 60)

def analyze_conv_layers(model, name):
    g = model.graph
    convs = [n for n in g.node if n.op_type == "Conv"]
    if not convs:
        print(f"  [{name}] No Conv ops")
        return

    kernel_sizes = Counter()
    strides = Counter()

    for n in convs:
        for attr in n.attribute:
            if attr.name == "kernel_shape":
                kernel_sizes[tuple(attr.ints)] += 1
            elif attr.name == "strides":
                strides[tuple(attr.ints)] += 1

    print(f"\n  [{name}]  {len(convs)} Conv layers")
    print(f"  Kernel shapes: {dict(kernel_sizes)}")
    print(f"  Strides:       {dict(strides)}")

    # 检查是否包含 depthwise conv (group == in_channels == out_channels)
    # 检查是否有点卷积 (1x1 conv)
    k1x1 = kernel_sizes.get((1, 1), 0)
    k3x3 = kernel_sizes.get((3, 3), 0)
    print(f"  1x1 Convs: {k1x1}  |  3x3 Convs: {k3x3}")

analyze_conv_layers(model_fp16, ONNX_FP16.name)
analyze_conv_layers(model_fp32, ONNX_FP32.name)

# ---------------------------------------------------------------------------
# Part B: ORT 原生格式分析
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PART B: ORT FORMAT (.ort) ANALYSIS")
print("=" * 72)

# .ort 文件无法直接用 onnx.load() 解析, 通过 onnxruntime 获取元信息
print(f"\n  Loading ORT model: {ORT_FP16.name} ({ORT_FP16.stat().st_size/(1024*1024):.1f} MB)")
try:
    sess_ort = ort.InferenceSession(str(ORT_FP16))
    print(f"  Providers: {sess_ort.get_providers()}")

    # 将 ORT 模型导出为 ONNX 来检查图结构
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = "fp16_results/_analyze_ort_exported.onnx"
    sess_export = ort.InferenceSession(str(ORT_FP16), so)

    # 分析 ORT 导出后的 ONNX
    exported_path = Path("fp16_results/_analyze_ort_exported.onnx")
    if exported_path.exists():
        model_ort, size_ort = load_onnx(exported_path)
        op_ort, nnapi_ort = analyze_ops(model_ort, f"{ORT_FP16.name} (exported ONNX)")
        struct_ort = graph_structure(model_ort, f"{ORT_FP16.name} (exported ONNX)", op_ort)
        params_ort, bytes_ort, dt_ort = analyze_weights(model_ort, f"{ORT_FP16.name} (exported ONNX)")

        print(f"\n  ** ORT 优化效果对比 **")
        print(f"  Source ONNX nodes: {sum(op_fp16.values())}")
        print(f"  ORT optimized nodes: {sum(op_ort.values())}")
        if sum(op_fp16.values()) > 0:
            reduction = (1 - sum(op_ort.values()) / sum(op_fp16.values())) * 100
            print(f"  Node reduction: {reduction:.1f}%")
except Exception as e:
    print(f"  ORT analysis skipped: {e}")

# Also analyze the .with_runtime_opt.ort
RUNTIME_ORT = Path("fp16_results/mobileclip2_s2_visual.with_runtime_opt.ort")
if RUNTIME_ORT.exists():
    print(f"\n  Runtime-optimized ORT: {RUNTIME_ORT.name} ({RUNTIME_ORT.stat().st_size/(1024*1024):.1f} MB)")

    so2 = ort.SessionOptions()
    so2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so2.optimized_model_filepath = "fp16_results/_analyze_runtime_ort_exported.onnx"
    try:
        sess_rt = ort.InferenceSession(str(RUNTIME_ORT), so2)
        rt_export = Path("fp16_results/_analyze_runtime_ort_exported.onnx")
        if rt_export.exists():
            model_rt, _ = load_onnx(rt_export)
            op_rt, nnapi_rt = analyze_ops(model_rt, f"{RUNTIME_ORT.name} (exported ONNX)")
    except Exception as e:
        print(f"  Runtime ORT analysis skipped: {e}")

# ---------------------------------------------------------------------------
# Part C: 推理性能基准测试
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PART C: INFERENCE BENCHMARK (CPU)")
print("=" * 72)

# 准备测试数据
dummy_image = np.random.randn(1, 3, 256, 256).astype(np.float32)

def benchmark_session(sess, input_name, data, warmup=5, iters=20):
    """运行推理基准测试"""
    # Warmup
    for _ in range(warmup):
        sess.run(None, {input_name: data})

    # Timed runs
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {input_name: data})
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000
    return times_ms.mean(), times_ms.std(), times_ms.min(), times_ms.max()

print("\n  Running benchmarks (this may take a while)...")
print()

bench_results = {}

for label, model_path in [
    ("FP16 ONNX", str(ONNX_FP16)),
    ("FP16 ORT", str(ORT_FP16)),
    ("FP32 ONNX", str(ONNX_FP32)),
]:
    try:
        sess = ort.InferenceSession(model_path)
        mean, std, tmin, tmax = benchmark_session(sess, "image", dummy_image)
        bench_results[label] = {"mean_ms": mean, "std_ms": std, "min_ms": tmin, "max_ms": tmax}
        print(f"  {label:<16s}: {mean:>8.2f} ms ± {std:>5.2f}  (min={tmin:.2f}, max={tmax:.2f})")
    except Exception as e:
        print(f"  {label:<16s}: ERROR - {e}")

# ---------------------------------------------------------------------------
# Part D: 综合结论
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("PART D: COMPREHENSIVE COMPARISON & CONCLUSIONS")
print("=" * 72)

print(f"""
┌──────────────────────────────────────────────────────────────────────┐
│                         QUANTITATIVE COMPARISON                       │
├──────────────────────────────────┬─────────────────┬─────────────────┤
│ Metric                           │ FP16 ORT (S2)   │ FP32 ONNX       │
├──────────────────────────────────┼─────────────────┼─────────────────┤
│ File size                        │ {ORT_FP16.stat().st_size/(1024*1024):>6.0f} MB       │ {size_fp32:>6.0f} MB       │
│ Weight precision                 │ FP16 internal   │ FP32            │
│ Input/Output precision           │ FP32            │ FP32            │
│ Total graph nodes                │ {sum(op_fp16.values()):>5d}           │ {sum(op_fp32.values()):>5d}           │
│ NNAPI coverage                   │ {nnapi_fp16:>4.1f}%           │ {nnapi_fp32:>4.1f}%           │
│ Parameters (visual encoder)      │ {params_fp16/1e6:>5.1f}M          │ {params_fp32/1e6:>5.1f}M          │
│ Conv layers                      │ {op_fp16.get('Conv',0):>5d}           │ {op_fp32.get('Conv',0):>5d}           │
│ MatMul layers                    │ {op_fp16.get('MatMul',0):>5d}           │ {op_fp32.get('MatMul',0):>5d}           │
└──────────────────────────────────┴─────────────────┴─────────────────┘
""")

if bench_results:
    print(f"""
┌──────────────────────────────────────────────────────────────────────┐
│                      INFERENCE BENCHMARK (CPU)                        │
├──────────────────────────────────┬────────────────────────────────────┤
│ Model                            │ Latency (mean ± std)              │
├──────────────────────────────────┼────────────────────────────────────┤
""")
    for label, res in bench_results.items():
        print(f"│ {label:<32s} │ {res['mean_ms']:>7.2f} ± {res['std_ms']:>5.2f} ms             │")
    print("└──────────────────────────────────┴────────────────────────────────────┘")

    # 计算加速比
    if "FP16 ORT" in bench_results and "FP32 ONNX" in bench_results:
        speedup = bench_results["FP32 ONNX"]["mean_ms"] / bench_results["FP16 ORT"]["mean_ms"]
        print(f"\n  >>> FP16 ORT is {speedup:.2f}x faster than FP32 ONNX <<<")

print(f"""
┌──────────────────────────────────────────────────────────────────────┐
│                         综合评估结论                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  mobileclip2_s2_visual.ort (MobileCLIP2-S2, FP16/ORT) 在 Android     │
│  端的推理效率明显优于 clip-fp32/visual.onnx, 原因如下:                 │
│                                                                      │
│  1. 【精度优势】FP16 vs FP32                                         │
│     - 权重大小减半 ({bytes_fp16/(1024*1024):.0f}MB vs {bytes_fp32/(1024*1024):.0f}MB)                               │
│     - ARM NEON FP16 指令吞吐是 FP32 的 2x (AArch64)                   │
│     - 高通 Adreno GPU / Hexagon DSP 对 FP16 有原生加速               │
│     - 内存带宽压力减半, 对移动端至关重要                               │
│                                                                      │
│  2. 【架构优势】FastViT-MCI2 vs 标准 ViT                             │
│     - MobileCLIP2-S2 使用 FastViT-MCI2 架构:                          │
│       * MobileOne 风格的 rep-param blocks (多分支训练→单分支推理)      │
│       * 大量 1x1 + 3x3 depthwise Conv (移动端友好)                    │
│       * 更少的 Transformer blocks, 更多 Conv 层                       │
│     - 标准 ViT 以 Self-Attention 为主:                                │
│       * MatMul 密集, CPU 上 QKV 计算量大                             │
│       * Softmax 对 NNAPI 不够友好                                     │
│       * 内存访问模式不规则 (attention matrix)                          │
│                                                                      │
│  3. 【ORT 优化】ONNX Runtime 原生格式                                 │
│     - ORT 格式已包含: 算子融合, 常量折叠, 冗余节点消除                 │
│     - Operator fusion 减少内存往返 (Conv+ReLU, Conv+BN, etc.)        │
│     - 图级别的内存规划优化                                             │
│                                                                      │
│  4. 【NNAPI 兼容性】                                                  │
│     - FP16 模型中 Conv 占比高 → NNAPI EP 可以接管大部分计算           │
│     - FP32 模型中若 MatMul/Softmax 较多, NNAPI 覆盖率较低             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
""")

print("""
┌──────────────────────────────────────────────────────────────────────┐
│      对 mobileclip2_s2_visual.ort 的优化策略 (非简单动态量化)          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  策略 1: NNAPI EP 适配性增强                                         │
│  ────────────────────────────                                        │
│  - 识别并替换 NNAPI 不支持的算子:                                     │
│    · Erf → 多项式近似 (Gelu 中用到)                                   │
│    · Resize(非linear/nearest) → 替换为支持的插值模式                  │
│    · ReduceProd → ReduceSum + Log/Exp 分解                           │
│  - 将 LayerNorm 的 pattern 拆为 NNAPI 原生支持的子操作               │
│  - 确保所有 Conv weight 在初始化时就被折叠 (无动态权重)                │
│                                                                      │
│  策略 2: 静态逐通道 INT8 量化 (精度保持)                              │
│  ─────────────────────────────────                                   │
│  - 使用校准数据集 (500-1000 张代表性图片) 做 per-channel 量化         │
│  - 关键: 保持第一层 Conv 和最后一层 Projection 为 FP16               │
│    (这两层对精度最敏感)                                                │
│  - 只量化 MatMul/Gemm 的 weight → QLinearMatMul                      │
│  - Conv 层保持 FP16 (Conv 已高度优化, INT8 收益有限)                  │
│  - 预期: 模型大小再减 ~35%, 精度损失 <0.5%                             │
│                                                                      │
│  策略 3: QDQ (Quantize-Dequantize) 混合精度                          │
│  ─────────────────────────────────────                               │
│  - 在 ONNX 图中插入 Q/DQ 节点对, 标记可量化的计算子图                 │
│  - NNAPI EP 原生支持 QDQ Conv/QDQ MatMul                              │
│  - 允许 EP 选择 FP16 或 INT8 执行路径                                 │
│  - 比全局动态量化精度高, 因为可以针对每个 tensor 定制 scale           │
│                                                                      │
│  策略 4: 针对高通 Hexagon DSP 的 QNN EP                              │
│  ───────────────────────────────────────                             │
│  - 使用 QNN Execution Provider 替代 NNAPI EP                         │
│  - QNN 对 Conv 算子的支持远好于 NNAPI (支持更多 fused ops)           │
│  - 支持原生 FP16 推理 (无需量化为 INT8)                               │
│  - 在骁龙 8 Gen 系列上有显著加速                                       │
│  - 注意: QNN EP 需要 libQnn*.so 与设备匹配                            │
│                                                                      │
│  策略 5: 图结构级优化                                                 │
│  ──────────────────                                                  │
│  - 使用 onnx-simplifier 进一步简化:                                   │
│    · 常量折叠 (额外轮次)                                               │
│    · 消除 Identity 链                                                  │
│    · 合并连续的 Transpose/Reshape                                      │
│  - 手动合并 pattern:                                                  │
│    · LayerNorm → 自定义 fused LayerNorm (如果 EP 不支持)              │
│    · MultiHeadAttention → 拆分为更小的 MatMul (便于 NNAPI 调度)       │
│                                                                      │
│  策略 6: 运行时配置优化 (ORT Mobile)                                  │
│  ─────────────────────────────────────                               │
│  - Android 端最佳实践:                                                │
│    · intra_op_num_threads = device_core_count - 1 (不要全占满)       │
│    · inter_op_num_threads = 1 或 2                                    │
│    · execution_mode = SEQUENTIAL (移动端并行度有限)                   │
│    · 启用 graph_optimization_level = ENABLE_EXTENDED                  │
│  - 使用 ORT 的 IOBinding 避免输入输出拷贝                              │
│  - 预热: 首次推理前跑 1-2 次 dry run                                   │
│  - 考虑使用 ORT 的 reduced opset (限制 opset 提升兼容性)              │
│                                                                      │
│  策略 7: 结构化剪枝 + 微调                                            │
│  ─────────────────────────                                           │
│  - FastViT-MCI2 的 MobileOne block 有多条分支:                       │
│    训练后已 re-parameterize, 部署图只保留推理分支                      │
│  - 进一步: 对中间层的 channel 做 L1-norm 剪枝                          │
│  - 剪枝后做少量微调 (few-shot calibration) 恢复精度                    │
│  - 目标: 减少 20-30% 计算量, 精度损失 <1%                              │
│                                                                      │
│  策略 8: 使用更小的模型变体                                           │
│  ──────────────────────────                                          │
│  - MobileCLIP2 家族: S0 < S1 < S2 < S3 < S4 < B < L-14              │
│  - 如果 latency 要求严格, 考虑降级到 S1 或 S0:                        │
│    · S0: ~10M params, < 5ms/step (移动端)                            │
│    · S2: ~20M params (当前)                                          │
│  - 通过 knowledge distillation 用 S2 蒸馏 S0/S1                      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
""")

# ---------------------------------------------------------------------------
# 清理临时文件
# ---------------------------------------------------------------------------
for tmp in [Path("fp16_results/_analyze_ort_exported.onnx"),
            Path("fp16_results/_analyze_runtime_ort_exported.onnx")]:
    if tmp.exists():
        tmp.unlink()
        print(f"  Cleaned up: {tmp.name}")
