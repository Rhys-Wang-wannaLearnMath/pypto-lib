# S1-1：K_CHUNK 缩减优化记录

## 概要

| 项目 | 内容 |
|------|------|
| **优化编号** | S1-1 |
| **目标文件** | `examples/models/qwen-opt/qwen3_32b_decode_scope1_opt.py` |
| **原始文件** | `examples/models/qwen-opt/qwen3_32b_decode_scope1.py` |
| **优化类型** | 参数调优 — 分块大小缩减 |
| **涉及 Scope** | Scope 1（RMSNorm + Q/K/V 投影） |
| **正确性风险** | 极低（reduction 维度分块粒度变化，不改变累加结果） |

---

## 1. 优化内容

### 1.1 变更描述

将 Scope 1 Opaque 版本的 `K_CHUNK`（reduction 维度分块大小）从 **512** 降为 **128**，
与 tile 版本 (`qwen3_32b_decode_scope1_tile.py`) 和 `qwen3_tilelet.md` 最优配置对齐。

### 1.2 变更前

```python
# qwen3_32b_decode_scope1.py 第 30 行
K_CHUNK = 512
```

Vector tile `[BATCH_TILE, K_CHUNK]` FP32 = [16, 512] × 4 B = **32 KB**，超限 TILELET 上限 (2 KB) 的 16 倍。

### 1.3 变更后

```python
# qwen3_32b_decode_scope1_opt.py 第 30 行
K_CHUNK = 128
```

Vector tile `[BATCH_TILE, K_CHUNK]` FP32 = [16, 128] × 4 B = **8 KB**，超限 4 倍（与 tile 版本一致）。

### 1.4 代码影响范围

`K_CHUNK` 影响以下循环的迭代次数：
- **RMSNorm squared sum 循环**：`hidden_blocks = HIDDEN // K_CHUNK`，从 16 次增加到 64 次
- **RMSNorm apply 循环**：同上
- **Q 投影 matmul_acc 循环**：从 16 次增加到 64 次
- **K/V 投影 matmul_acc 循环**：从 16 次增加到 64 次

所有循环的**外层结构和内层操作不变**，仅分块粒度变细。

---

## 2. 数学等价性分析

### 2.1 RMSNorm

RMSNorm 计算 `inv_rms = 1 / sqrt(mean(x^2) + eps)`，其中：

```
sum_sq = Σ_{k=0}^{HIDDEN-1} x[b, k]^2
```

K_CHUNK 只影响累加的分组方式：
- K_CHUNK=512：16 组，每组 512 个元素的 partial sum
- K_CHUNK=128：64 组，每组 128 个元素的 partial sum

两者的最终 `sum_sq` 在精确算术下完全相同。浮点累加顺序的变化可能引入
O(eps_machine) 级别的差异，在 rtol/atol=1e-3 的容差内。

### 2.2 Q/K/V 投影

矩阵乘法 `normed_tile × W` 的 reduction 维度按 K_CHUNK 分块累加：

```
result[b, o] = Σ_{k=0}^{HIDDEN-1} normed[b, k] × W[k, o]
            = Σ_{kb=0}^{HIDDEN_BLOCKS-1} Σ_{j=0}^{K_CHUNK-1} normed[b, kb*K_CHUNK+j] × W[kb*K_CHUNK+j, o]
```

K_CHUNK 只影响外层循环次数和每次 matmul 的 K 维度大小。最终累加结果等价。

---

## 3. 硬件约束分析

### 3.1 Opaque 模式下的 TILELET 拆分

在 `pl.FunctionType.Opaque` 模式下，编译器负责自动将用户指定的 chunk 大小拆分为
硬件 TILELET（2 KB for Vec, 变长 for Cube）。

| K_CHUNK | Vec tile 大小 | 编译器拆分次数 | Cube tile [16, K_CHUNK] BF16 |
|---------|-------------|--------------|------------------------------|
| 512 | [16, 512] FP32 = 32 KB | 16 个 TILELET | 16 KB（1 个 TILE） |
| 128 | [16, 128] FP32 = 8 KB | 4 个 TILELET | 4 KB（1 个 TILE） |

K_CHUNK=128 减少了编译器生成微代码的复杂度。

### 3.2 Cube TILE 利用率

Cube 的 TILE 规格为 `[16, 16]` 元素（BF16）。
- K_CHUNK=512：每个 matmul 的 K 维为 512，分 32 次内积
- K_CHUNK=128：每个 matmul 的 K 维为 128，分 8 次内积

两者均能完整利用 cube 单元。

---

## 4. 预期收益

### 4.1 编译器端

- 减少 Vec TILELET 拆分次数：每个 RMSNorm chunk 操作从 16 次减少到 4 次
- 生成更简洁的微代码
- 降低编译时间

### 4.2 运行时

- **Vec 内存局部性改善**：更小的 chunk 更容易驻留在 L1 缓存中
- **循环次数增加**：从 16 次增到 64 次，增加了循环调度开销

**Net effect 需要 profiling 验证**。在 Opaque 模式下编译器的自动 tiling 策略
可能使两种配置产生接近的性能。

### 4.3 代码一致性

与 tile 版本 (`qwen3_32b_decode_scope1_tile.py`) 的 K_CHUNK=128 对齐，
便于跨版本对比和维护。

---

## 5. 验证方法

### 5.1 正确性验证

```bash
# 运行优化版本
python3 examples/models/qwen-opt/qwen3_32b_decode_scope1_opt.py -p a2a3sim

# 使用固定 golden 数据对比
python3 examples/models/qwen-opt/qwen3_32b_decode_scope1.py -p a2a3sim --golden-data golden_baseline/scope1
python3 examples/models/qwen-opt/qwen3_32b_decode_scope1_opt.py -p a2a3sim --golden-data golden_baseline/scope1
```

验证通过条件：`[RUN] PASS`，rtol=1e-3, atol=1e-3。

### 5.2 性能数据对比

```bash
# 带 profiling 运行
python3 examples/models/qwen-opt/qwen3_32b_decode_scope1_opt.py -p a2a3sim --runtime-profiling

# 分析性能数据
python3 tools/perf_analyzer.py build_output/Qwen3Scope1_K128_<timestamp>/ -v
```

关注指标：
- **Kernel 数量**：K_CHUNK 变化不应改变 kernel 数量（Opaque 模式）
- **Vec 执行时间**：可能改善（更好的缓存利用）或不变
- **Cube 执行时间**：预计不变（总计算量相同）

### 5.3 自定义输出目录

优化版本的编译产物输出到 `build_output/Qwen3Scope1_K128_<timestamp>/`。

---

## 6. 附加变更

- 添加 `--golden-data` CLI 参数，支持固定输入对比
- 通过 `RunConfig.compile` 的 `output_dir` 设置自定义输出目录
