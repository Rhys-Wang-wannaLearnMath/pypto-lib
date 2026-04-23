# Qwen3-32B Decode Scope 优化结果报告

> **日期**：2026-04-23  
> **硬件**：Ascend 910B (a2a3)，binned 设备（20 AICore，实际可用 18，block_dim 从 24 截断至 18）  
> **框架**：PyPTO IR Compiler + tensormap_and_ringbuffer runtime  
> **模型**：Qwen3-32B 单层 Decode（BATCH=16, HIDDEN=8192, INTERMEDIATE=25600）

---

## 目录

1. [总览](#1-总览)
2. [优化实施清单](#2-优化实施清单)
3. [Scope 1 结果分析](#3-scope-1-结果分析)
4. [Scope 2 结果分析](#4-scope-2-结果分析)
5. [Scope 3 结果分析（重点）](#5-scope-3-结果分析重点)
6. [关键发现：Runtime 自动流水 vs 显式并行](#6-关键发现runtime-自动流水-vs-显式并行)
7. [编译产物对比](#7-编译产物对比)
8. [内存使用对比](#8-内存使用对比)
9. [结论与建议](#9-结论与建议)

---

## 1. 总览

### 1.1 构建目录

| Scope | 基线目录 | 优化目录 |
|-------|---------|---------|
| Scope 1 | `Qwen3Scope1_20260423_191737` | `Qwen3Scope1_OPT_20260423_191753` |
| Scope 2 | `Qwen3Scope2_20260423_191809` | `Qwen3Scope2_OPT_20260423_191826` |
| Scope 3 | `Qwen3Scope3_20260423_191844` | `Qwen3Scope3_OPT_20260423_191902` |

### 1.2 核心指标汇总

| Scope | 基线 Span (us) | OPT Span (us) | 变化 | 基线 Tasks | OPT Tasks | Golden 校验 |
|-------|:-----------:|:----------:|:----:|:---------:|:--------:|:----------:|
| **Scope 1** | 407.6 | 406.8 | **-0.2%** | 37 | 37 | ✅ PASS |
| **Scope 2** | 879.7 | 920.4 | **+4.6%** | 529 | 529 | ✅ PASS |
| **Scope 3** | 3568.6 | 26184.5 | **+633%** ⚠️ | 1585 | 345 | ✅ PASS |

> **注意**：Scope 3 虽然 golden 校验通过（数学正确），但运行时性能出现了显著回退。
> 这是本次优化最重要的发现——详见第 6 节分析。

---

## 2. 优化实施清单

### 2.1 所有三文件共通

- **类名重命名**：`Qwen3Scope{1,2,3}` → `Qwen3Scope{1,2,3}_OPT`
  - 效果：输出目录自动从 `Qwen3Scope3_<ts>` 变为 `Qwen3Scope3_OPT_<ts>`
  - 原理：`ir.compile()` 使用 `program.name` 构建目录名

### 2.2 Scope 3 优化（4 项）

| 编号 | 优化 | 代码变更 | 分析文档优先级 | 预期收益 | 实际结果 |
|------|------|---------|:-----------:|---------|---------|
| **S3-1** | Gate/Up 投影融合 | 两个 `pl.at` 块 → 单个 fused 块 | P0 | GM 读取减半 (~100 MB) | ✅ 总计算量减少 44%，但被序列化抵消 |
| **S3-2** | Output Projection 并行化 | `pl.range(128)` → `pl.parallel(128, chunk=4)` | P1 | ~16x 该阶段 | ❌ 阶段 span 从 402→4506 us (+1020%) |
| **S3-3** | MLP block 并行化 | `pl.range(400)` → `pl.parallel(400, chunk=4)` | P1 | ~17x 该阶段 | ❌ 阶段 span 从 ~2034→20182 us (+892%) |
| **S3-4** | Down projection 并行化 | `pl.range(64)` → `pl.parallel(64, chunk=4)` | P1 | ~16x 该阶段 | ⚠️ 阶段 span 从 881→946 us (+7.4%) |

### 2.3 Scope 2 — 尝试与回退

| 编号 | 优化 | 状态 | 原因 |
|------|------|------|------|
| **S2-4** | QK+Softmax+SV per-sb 融合 | ❌ 已回退 | PyPTO 编译器不支持单个 incore block 内 acc→vec 跨核数据移动 |

### 2.4 Scope 1

仅重命名，无计算逻辑变更。

---

## 3. Scope 1 结果分析

**源文件**：`qwen3_32b_decode_scope1-opt.py`  
**变更**：仅类名 `Qwen3Scope1` → `Qwen3Scope1_OPT`

### 3.1 性能对比

| 指标 | 基线 | OPT | 差异 |
|------|:----:|:---:|:----:|
| Span | 407.6 us | 406.8 us | -0.2% |
| 任务数 | 37 | 37 | 0 |

### 3.2 逐 Kernel 对比

| func_id | 类型 | 功能 | 基线 (次数×avg) | OPT (次数×avg) |
|---------|------|------|:------------:|:-----------:|
| 0 | AIV | RMSNorm | 1×34.52 us | 1×34.90 us |
| 1 | AIC | Q 投影 | 32×127.31 us | 32×126.76 us |
| 2 | AIC | K/V 投影 | 4×236.75 us | 4×235.21 us |

**结论**：无实质变化，性能差异在噪声范围内。符合预期。

### 3.3 内存使用

基线与 OPT 的 `memory_after_AllocateMemoryAddr.txt` 完全一致：

| Kernel | 空间 | 使用 | 上限 | 利用率 |
|--------|------|:----:|:----:|:------:|
| incore_0 (AIV) | Vec | 82.2 KB | 192.0 KB | 42.8% |
| incore_1 (AIC) | Mat | 80.0 KB | 512.0 KB | 15.6% |
| | Right (L0B) | 64.0 KB | 64.0 KB | **100.0%** |
| incore_2 (AIC) | Mat | 80.0 KB | 512.0 KB | 15.6% |
| | Right (L0B) | 64.0 KB | 64.0 KB | **100.0%** |

> L0B 打满（100%）说明 K_CHUNK=512 时 weight tile `[512, 64]×2B=64KB` 恰好填满 L0B。
> 这是 S1-1 优化（K_CHUNK 降为 128）可以改善的方向。

---

## 4. Scope 2 结果分析

**源文件**：`qwen3_32b_decode_scope2-opt.py`  
**变更**：仅类名 `Qwen3Scope2` → `Qwen3Scope2_OPT`（S2-4 融合尝试失败后已回退）

### 4.1 S2-4 融合失败记录

**尝试方案**：将 QK matmul (Stage 2) + Softmax (Stage 3) + SV matmul (Stage 4) 融合为
单个 `pl.parallel` 循环，消除 2 个全局同步屏障。

**编译错误 1**（直接 slice matmul 结果）：
```
'pto.textract' op expects A2/A3 textract src to use loc=mat
```
`pl.matmul` 输出在 acc (L0C) 中，`pl.slice` 要求 src 在 mat (L1)。

**编译错误 2**（通过 `pl.assemble` 中转到临时 buffer）：
```
Internal error: no MLIR mapping for MemRef base 'mem_acc_8'
```
codegen 无法处理 acc 空间到 GM 的 assemble 后再 slice 的模式。

**根因**：PyPTO 当前编译器不支持在单个 incore 函数内进行 **Cube (acc) → Vector (vec)** 跨核数据移动。
QK matmul 产出在 AIC 的 L0C 累加器中，softmax 需要在 AIV 的 Vec 中执行。
这两步必须通过 **独立的 `pl.at` 块**（含隐式 GM 中转和全局同步）来连接。

### 4.2 性能对比

| 指标 | 基线 | OPT | 差异 |
|------|:----:|:---:|:----:|
| Span | 879.7 us | 920.4 us | +4.6% |
| 任务数 | 529 | 529 | 0 |

### 4.3 逐 Kernel 对比

| func_id | 类型 | 功能 | 基线 (次数×avg) | OPT (次数×avg) |
|---------|------|------|:------------:|:-----------:|
| 0 | AIV | KV cache write | 1×23.04 us | 1×24.44 us |
| 1 | AIV | RoPE + Q pad | 16×27.91 us | 16×27.96 us |
| 2 | AIC | QK matmul | 128×17.92 us | 128×17.88 us |
| 3 | AIV | Softmax | 128×9.31 us | 128×9.32 us |
| 4 | AIC | SV matmul | 128×16.64 us | 128×16.67 us |
| 5 | AIV | Online softmax | 128×10.45 us | 128×10.64 us |

**结论**：无实质变化。+4.6% span 差异属于运行间噪声。

### 4.4 内存使用

基线与 OPT 完全一致。所有 kernel 的片上内存利用率均在安全范围内。

---

## 5. Scope 3 结果分析（重点）

**源文件**：`qwen3_32b_decode_scope3-opt.py`  
**变更**：S3-1 融合 + S3-2/3/4 并行化

### 5.1 总体性能

| 指标 | 基线 | OPT | 变化 |
|------|:----:|:---:|:----:|
| **Span** | **3568.6 us** | **26184.5 us** | **+633%** ⚠️ |
| 任务总数 | 1585 | 345 | -78% |
| Orchestrator 完成时间 | 2048.96 us | 209.54 us | -89.8% |
| Scheduler 总阶段数 | 14,793 | 608,514 | +4013% |
| Swimlane perf 文件大小 | 3.7 MB | 73.7 MB | +1892% |

**矛盾现象**：任务数减少 78%，orchestrator 完成快 90%，但实际执行慢了 7.3 倍。

### 5.2 逐阶段详细对比

#### Output Projection（S3-2 并行化）

| 指标 | 基线 | OPT | 变化 |
|------|:----:|:---:|:----:|
| AIC 任务数 | 128 | 32 | -75% |
| AIC 使用核心 | **0-17（18 核）** | **仅 core 0** | ⚠️ |
| AIC 阶段 span | 402.3 us | 4505.8 us | **+1020%** |
| AIC 总计算时间 | 6433.4 us | 4377.5 us | -32% |
| AIC 平均耗时 | 50.26 us | 136.80 us | +172% |
| AIV 使用核心 | core 20 | cores 18-19 | — |
| AIV 阶段 span | 487.4 us | 4505.0 us | +824% |

**关键发现**：
- 基线 128 个 AIC 任务被 runtime 自动分配到 **全部 18 个 AIC 核心**并行执行
- OPT 32 个 chunked 任务全部集中在 **core 0** 上串行执行
- 原因：OPT 的 MixedKernel 使用 `add_inout(resid1_tile)`，runtime 检测到共享张量的写依赖，将所有任务序列化到同一核心

#### Gate/Up 融合 + MLP 并行化（S3-1 + S3-3）

| 指标 | 基线 Gate | 基线 Up | OPT 融合 | 变化 |
|------|:--------:|:------:|:-------:|:----:|
| AIC 任务数 | 400 | 400 | 100 | -87.5% |
| AIC 使用核心 | 0-17 | 0-17 | **仅 core 0** | ⚠️ |
| AIC 阶段 span | 2034.3 us | 2034.3 us | 20182.1 us | +397% vs 单阶段 |
| AIC 总计算时间 | 18208.1 us | 17356.5 us | 19917.6 us | -44% vs 合计 |
| AIC 平均耗时 | 45.52 us | 43.39 us | 199.18 us | — |

**关键发现**：
- **S3-1 融合效果验证**：总计算时间 19917.6 us vs 基线 gate+up 合计 35564.6 us，**计算量减少 44%**
- 平均耗时 199.18 us ≈ 基线 gate+up 合计 (45.52+43.39)×2 = 177.8 us ÷ 2（考虑 chunk=4），融合确实共享了 `post_norm_tile` 读取
- **但**：所有 100 个 AIC 任务因 `add_inout(gate_tile)` + `add_inout(up_tile)` 依赖被序列化到 core 0
- 结果：20182.1 us span vs 基线双阶段 2034.3 us span（各自并行到 18 核），**回退 ~10x**

#### SiLU 激活

| 指标 | 基线 | OPT | 变化 |
|------|:----:|:---:|:----:|
| AIV 任务数 | 400 | 100 | -75% |
| AIV 使用核心 | 18-30（13 核） | 仅 core 18 | ⚠️ |
| AIV 阶段 span | 1940.7 us | 485.3 us | -75% ✅ |
| AIV 平均耗时 | 1.29 us | 2.76 us | +114% |

SiLU 阶段由于计算量极小（平均 2.76 us/task），序列化影响较小。

#### Down Projection（S3-4 并行化）

| 指标 | 基线 | OPT | 变化 |
|------|:----:|:---:|:----:|
| AIC 任务数 | 64 | 16 | -75% |
| AIC 使用核心 | 0-17（18 核） | 0-16（16 核） | ✅ |
| AIC 阶段 span | 880.9 us | 946.1 us | +7.4% |
| AIC 总计算时间 | 14011.5 us | 14667.0 us | +4.7% |

**分析**：Down projection 是唯一接近预期行为的并行化阶段：
- MixedKernel 使用 `add_output(ext_out)` 和 `add_input()` 而非 `add_inout()`
- Runtime 能够判断输出不重叠，因此成功分配到 16 个核心
- 阶段 span 略增（+7.4%），可能由于 chunk=4 的 kernel 粒度导致尾部核心利用率下降

### 5.3 Scope 3 时间线重建

```
基线（3568.6 us）：

|--- Output Proj (18核并行) ---|- RMS -|--- Gate (18核) ---|--- Up (18核) ---|-- SiLU -|--- Down (18核) ---|
|         402 us              | 43 us |      2034 us     |     2034 us     | 1941 us |      881 us      |
                                       ↕ Gate 和 Up 在时间上部分重叠

OPT（26184.5 us）：

|---------- Output Proj (core 0 串行) ----------|- RMS -|---------- Gate+Up 融合 (core 0 串行) ----------|-- SiLU --|-- Down (16核) --|
|                4506 us                        | 44 us |                   20182 us                    |  485 us  |    946 us      |
```

---

## 6. 关键发现：Runtime 自动流水 vs 显式并行

### 6.1 根因分析

PyPTO 的 `tensormap_and_ringbuffer` runtime 具有**自动任务调度**能力：

- **顺序 `pl.range` 循环**：编译器为每次迭代生成独立的小任务（带独立的中间 tensor 分配）。
  runtime 的 3 个 scheduler 线程可以同时将不同迭代的任务分发到全部可用核心上。
  依赖关系通过 **per-iteration tensor view/alloc** 自然消除。

- **并行 `pl.parallel(N, chunk=C)` + `chunked_loop_optimizer`**：编译器将 C 个迭代融合为
  单个大 kernel，外层 N/C 次循环仍为顺序 dispatch。每个 kernel 使用**共享的 inout tensor**
  （如 `gate_tile`、`resid1_tile`）。runtime 无法判断不同 kernel 调用写入的区域不重叠，
  因此**强制序列化到同一核心**。

### 6.2 具体机制

#### 基线编排模式（以 MLP gate 为例）

```cpp
for (int64_t ob = 0; ob < 400; ob += 1) {
    PTO2_SCOPE() {
        TaskOutputTensors alloc = alloc_tensors(ret0__out_ci);  // ← 每次迭代独立分配
        const Tensor& gate_acc = alloc.get_ref(0);               // ← 独立的输出 tensor
        pto2_rt_submit_aic_task(3, params);                       // ← 无跨迭代依赖
    }
}
```

- 每次迭代分配独立的 `ret0__out` tensor（4 KB）
- Runtime 判断迭代间无依赖 → 自动并行到 18 核 → 400/18 ≈ 22 轮 × 45 us ≈ **1000 us**

#### OPT 编排模式（S3-1 融合 + S3-3 并行化后）

```cpp
const Tensor& gate_tile = alloc_0.get_ref(2);  // ← 所有迭代共享的 inout tensor
const Tensor& up_tile = alloc_0.get_ref(3);     // ← 所有迭代共享的 inout tensor
for (int64_t ob = 0; ob < 100; ob += 1) {
    PTO2_SCOPE() {
        params.add_inout(gate_tile);              // ← 共享 tensor 作为 inout
        params.add_inout(up_tile);                // ← 共享 tensor 作为 inout
        pto2_rt_submit_aic_task(3, params);        // ← Runtime 检测到写冲突 → 序列化
    }
}
```

- `gate_tile` [16, 25600] FP32 = 1.6 MB 被所有 100 个任务共享
- Runtime 无法看到 kernel 内部的写偏移 → 保守假设写冲突 → **所有 100 任务串行到 core 0**

### 6.3 为什么 S3-4 Down Projection 没有被序列化？

Down projection 使用 **MixedKernels** 模式（AIC + AIV 联合 dispatch）：

```cpp
params.add_output(ext_out);    // ← output（非 inout），runtime 知道是只写
params.add_input(mlp_tile);    // ← input（只读）
MixedKernels mixed = {5, 6, 6};
pto2_rt_submit_task(mixed, params);  // ← MixedKernel dispatch
```

- `ext_out` 标记为 `output`（非 `inout`），且每次写入不同偏移（通过 scalar 参数 `dob`）
- `mlp_tile` 标记为 `input`（只读）
- Runtime 在 MixedKernel 模式下可能有更宽松的依赖判断 → 允许多核并行

---

## 7. 编译产物对比

### 7.1 Kernel 数量

| Scope | 基线 Kernels | OPT Kernels | 变化 |
|-------|:----------:|:---------:|:----:|
| Scope 1 | 3 (1 aiv + 2 aic) | 3 (1 aiv + 2 aic) | 不变 |
| Scope 2 | 6 (3 aiv + 3 aic) | 6 (3 aiv + 3 aic) | 不变 |
| Scope 3 | 8 (4 aiv + 4 aic) | 7 (4 aiv + 3 aic) | -1 |

Scope 3 OPT 减少了 1 个 kernel：gate 和 up matmul 融合为单个 AIC kernel（`scope3_incore_2`），
但新增了 MixedKernel 分裂（output proj 和 down proj 的 AIC+AIV 对）。

### 7.2 MixedKernel 生成

基线 Scope 3 **不使用** MixedKernel（AIC 和 AIV 分别独立 dispatch）。
OPT Scope 3 生成了 **2 个 MixedKernel 组**：

| MixedKernel | AIC 文件 | AIV 文件 | 用途 |
|-------------|---------|---------|------|
| scope3_incore_0 | scope3_incore_0_aic.cpp | scope3_incore_0_aiv.cpp | Output Proj + Residual Add |
| scope3_incore_4 | scope3_incore_4_aic.cpp | scope3_incore_4_aiv.cpp | Down Proj + Final Residual |

这是 `pl.parallel` + `chunked_loop_optimizer` 将同一 parallel body 内的 Cube (matmul) 和
Vector (add/cast) 操作拆分为 AIC+AIV 对并用 MixedKernel 联合 dispatch 的结果。

### 7.3 Orchestration 代码对比

| 指标 | 基线 | OPT | 变化 |
|------|:----:|:---:|:----:|
| scope3.cpp 大小 | 7495 bytes | 6530 bytes | -12.9% |
| scope3.so 大小 | 638944 bytes | 630544 bytes | -1.3% |
| 外层循环迭代总数 | 128+400+400+64 = 992 | 32+100+100+16 = 248 | -75% |
| `alloc_tensors` 调用次数/iteration | 每次独立分配 | 共享 gate_tile/up_tile | — |

### 7.4 Passes Dump 对比

| 指标 | 基线 | OPT |
|------|:----:|:---:|
| frontend.py 大小 | 8061 bytes | 9031 bytes (+12%) |
| SplitChunkedLoops 后大小 | 11900 bytes | 16130 bytes (+36%) |
| 最终 IR 大小 | 26354 bytes | 38018 bytes (+44%) |
| SplitVectorKernel log | 无 | 1536 bytes（有拆分记录） |

OPT 的 IR 膨胀 44% 主要因为：
- Chunked parallel loop 拆分产生更多控制流
- MixedKernel 拆分引入 AIC/AIV 配对代码
- Gate/Up 融合增加了单 kernel 内的操作数量

---

## 8. 内存使用对比

### 8.1 Scope 3 片上内存

**基线**（8 kernels）：

| Kernel | 功能 | Mat | Vec | L0A | L0B | Acc |
|--------|------|:---:|:---:|:---:|:---:|:---:|
| incore_0 | Output Proj (AIC) | 20 KB | — | 4 KB | 16 KB | 4 KB |
| incore_1 | Residual Add (AIV) | — | 10 KB | — | — | — |
| incore_2 | RMSNorm (AIV) | — | 20.7 KB | — | — | — |
| incore_3 | Gate Matmul (AIC) | 20 KB | — | 4 KB | 16 KB | 4 KB |
| incore_4 | Up Matmul (AIC) | 20 KB | — | 4 KB | 16 KB | 4 KB |
| incore_5 | SiLU (AIV) | — | 18 KB | — | — | — |
| incore_6 | Down Proj (AIC) | 18 KB | — | 2 KB | 16 KB | 8 KB |
| incore_7 | Down Resid (AIV) | — | 20 KB | — | — | — |

**OPT**（7 kernels）：

| Kernel | 功能 | Mat | Vec | L0A | L0B | Acc |
|--------|------|:---:|:---:|:---:|:---:|:---:|
| incore_0_aic | Output Proj (AIC) | 20 KB | — | 4 KB | 16 KB | 4 KB |
| **incore_0_aiv** | **Resid Add (AIV)** | — | **42 KB** | — | — | — |
| incore_1 | RMSNorm (AIV) | — | 20.7 KB | — | — | — |
| **incore_2** | **Fused Gate+Up (AIC)** | **36 KB** | — | 4 KB | 16 KB | **8 KB** |
| incore_3 | SiLU (AIV) | — | 18 KB | — | — | — |
| incore_4_aic | Down Proj (AIC) | 18 KB | — | 2 KB | 16 KB | 8 KB |
| **incore_4_aiv** | **Down Resid (AIV)** | — | **84 KB** | — | — | — |

**关键变化**：
- **incore_2 (Fused Gate+Up)**：Mat 20→36 KB (+80%), Acc 4→8 KB (+100%)
  - 因为融合后同时持有 gate 和 up 的 weight 和 accumulator
  - 符合分析文档预测：Mat=36 KB (post_chunk 4KB + wg 16KB + wu 16KB)，Acc=8 KB (gate 4KB + up 4KB)
  - 所有资源仍远低于硬件上限 ✓
- **incore_0_aiv**：Vec 10→42 KB (+320%)，因为 chunked parallel 需要更大的 UB 缓冲区
- **incore_4_aiv**：Vec 20→84 KB (+320%)，同理

### 8.2 GM 中间张量

| 张量 | 基线 | OPT | 变化 |
|------|------|-----|------|
| resid1_tile | [16, 8192] FP32 = 512 KB | 同 | 不变 |
| post_norm_tile | [16, 8192] BF16 = 256 KB | 同 | 不变 |
| mlp_tile | [16, 25600] BF16 = 800 KB | 同 | 不变 |
| gate_tile | — | [16, 25600] FP32 = **1.6 MB** | **新增** |
| up_tile | — | [16, 25600] FP32 = **1.6 MB** | **新增** |
| 中间 alloc/iteration | 每次 4-8 KB | 每次 64 KB (gm_pipe_buffer) | +8x |

**OPT 新增约 3.2 MB GM**（gate_tile + up_tile），
这是 S3-1 融合将 gate/up 结果存储为中间张量的代价。
在基线中，gate_acc 和 up_acc 是每次迭代独立分配的小 tensor（4 KB FP32）。

---

## 9. 结论与建议

### 9.1 核心结论

1. **数学正确性**：所有三个 scope 的 golden 校验均通过，优化不影响计算结果。

2. **S3-1 Gate/Up 融合在计算层面有效**：总计算时间减少 44%，符合预期的 GM 读取减半效果。

3. **`pl.parallel` + `chunked_loop_optimizer` 导致严重性能回退**：
   - Runtime 的 `tensormap_and_ringbuffer` 调度器对 **共享 inout tensor** 采用保守的依赖分析
   - 所有写入同一 tensor 的任务被序列化到同一核心，即使实际写入区域不重叠
   - 基线的 `pl.range` 顺序循环反而更快，因为每次迭代生成**独立的中间 tensor**，
     runtime 可以自动将任务分发到全部 18 核并行执行

4. **S2-4 跨核融合受编译器限制**：Cube (acc) → Vector (vec) 数据移动必须通过独立 `pl.at` 块。

### 9.2 优化建议

#### 短期（立即可行）

| 建议 | 操作 | 预期效果 |
|------|------|---------|
| **回退 S3-2/3/4** | 将 `pl.parallel` 改回 `pl.range` | 恢复 Scope 3 到基线性能 |
| **保留 S3-1 但用 `pl.range`** | 只做 gate/up 融合，不做并行化 | 享受 44% 计算量减少 + 18 核自动并行 |
| **移除 gate_tile/up_tile 中间张量** | 让编译器自动管理 per-iteration alloc | 减少 3.2 MB GM 占用 |

#### 中期（需要框架改进）

| 建议 | 需要的框架支持 | 预期效果 |
|------|-------------|---------|
| 改进 runtime 依赖分析 | 基于 tensor view offset 判断写不重叠 | `pl.parallel` 任务自动多核分发 |
| 支持 incore 内 acc→vec 数据移动 | 编译器自动插入 L0C→UB 数据搬运 | 解锁 S2-4 融合 |
| `pl.parallel` 使用 per-iteration alloc | 编译器为每个 chunk 生成独立中间 tensor | 恢复 runtime 自动并行能力 |

#### 长期

| 建议 | 说明 |
|------|------|
| Scope 间 Pipeline | batch 维度 pipeline (Scope 1 batch[1] ∥ Scope 2 batch[0]) |
| Tile DSL 版本并行化 | 在 `qwen3_32b_decode_scope1_tile.py` 中应用 S1-2 |

### 9.3 推荐的最优方案

基于本次实验结果，**推荐方案**为：

> **仅保留 S3-1（Gate/Up 融合），使用 `pl.range` 顺序循环（不使用 `pl.parallel`）。**

这将：
- ✅ 获得 S3-1 的 44% 计算量减少
- ✅ 保持 runtime 自动 18 核并行调度
- ✅ 避免 gate_tile/up_tile 中间张量（使用 per-iteration alloc）
- ✅ 预期 Scope 3 总 span 从 3568 us 降至 ~2500 us（~30% 改善）

---

## 附录 A：数据来源

### A.1 Swimlane 数据文件

| Build | perf_swimlane 大小 | merged_swimlane 大小 |
|-------|:-----------------:|:-------------------:|
| Scope1_BASE | 529 KB | 1.4 MB |
| Scope1_OPT | 574 KB | 1.6 MB |
| Scope2_BASE | 1.5 MB | 5.3 MB |
| Scope2_OPT | 1.7 MB | 5.9 MB |
| Scope3_BASE | 3.7 MB | 12.6 MB |
| Scope3_OPT | **73.7 MB** | **185.8 MB** |

Scope3_OPT 的 swimlane 文件膨胀 20 倍，因为：
- MixedKernel 生成更多 scheduler 阶段记录（608K vs 15K phases）
- 每个 chunked task 的内部并行信息被详细记录

### A.2 测试命令

```bash
# 基线
python examples/models/qwen3/qwen3_32b_decode_scope{1,2,3}.py -p a2a3

# OPT
python examples/models/qwen-opt/qwen3_32b_decode_scope{1,2,3}-opt.py -p a2a3
```

### A.3 Scope 3 Orchestration 对比摘要

**基线循环结构**：
```
for ob in 0..128: [AIC output_proj] → [AIV residual_add]    // 独立 alloc/iteration
[AIV rmsnorm]
for ob in 0..400: [AIC gate_matmul] → [AIC up_matmul] → [AIV silu]  // 独立 alloc/iteration
for dob in 0..64: [AIC down_proj] → [AIV down_residual]     // 独立 alloc/iteration
```

**OPT 循环结构**：
```
for ob in 0..32:  [MixedKernel(AIC+AIV) output_proj+residual]  // 共享 resid1_tile (inout)
[AIV rmsnorm]
for ob in 0..100: [AIC fused_gate_up]                          // 共享 gate_tile, up_tile (inout)
for ob in 0..100: [AIV silu]                                    // 共享 mlp_tile (inout)
for dob in 0..16: [MixedKernel(AIC+AIV) down_proj+residual]   // ext_out (output) + inputs
```
