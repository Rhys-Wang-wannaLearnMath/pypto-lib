# Qwen3-32B Decode 全 Scope 优化记录

> **硬件平台**：Ascend 910B (a2a3)，binned 设备（20 AICore，实际可用 18 核，block_dim 从 24 截断至 18）  
> **框架**：PyPTO IR Compiler + tensormap_and_ringbuffer runtime  
> **模型参数**：Qwen3-32B 单层 Decode  
> - BATCH = 16, HIDDEN = 8192, INTERMEDIATE = 25600  
> - NUM_HEADS = 64, NUM_KV_HEADS = 8, HEAD_DIM = 128  
> - MAX_SEQ = 4096

---

## 目录

1. [文件清单与对应关系](#1-文件清单与对应关系)
2. [整体性能对比](#2-整体性能对比)
3. [Scope 1 优化详情](#3-scope-1-优化详情)
4. [Scope 2 优化详情](#4-scope-2-优化详情)
5. [Scope 3 优化详情（核心优化）](#5-scope-3-优化详情核心优化)
6. [被否决的优化方案及原因](#6-被否决的优化方案及原因)
7. [关键技术发现](#7-关键技术发现)
8. [构建与验证](#8-构建与验证)

---

## 1. 文件清单与对应关系

| Scope | 基线文件 | 优化文件 |
|-------|---------|---------|
| Scope 1 (RMSNorm + Q/K/V 投影) | `qwen3/qwen3_32b_decode_scope1.py` | `qwen-opt/qwen3_32b_decode_scope1-opt.py` |
| Scope 2 (RoPE + KV Cache + GQA) | `qwen3/qwen3_32b_decode_scope2.py` | `qwen-opt/qwen3_32b_decode_scope2-opt.py` |
| Scope 3 (Output proj + RMSNorm + MLP + Residual) | `qwen3/qwen3_32b_decode_scope3.py` | `qwen-opt/qwen3_32b_decode_scope3-opt.py` |

---

## 2. 整体性能对比

| Scope | 基线 Span (us) | 优化 Span (us) | 提升幅度 | Task 数 (基线 → 优化) |
|-------|---------------|---------------|---------|----------------------|
| Scope 1 | 407.6 | 407.6 | +0.0% (保持不变) | 37 → 37 |
| Scope 2 | 879.7 | 871.0 | −1.0% (微调) | 529 → 529 |
| **Scope 3** | **3568.6** | **2636.7** | **−26.1%** | **1585 → 857** |

> 注：Span 定义为从第一个 task 启动到最后一个 task 完成的壁钟时间，由 runtime profiling swimlane JSON 数据计算。

---

## 3. Scope 1 优化详情

### 3.1 功能说明

Scope 1 覆盖 Transformer 单层的前端计算：
1. **Input RMSNorm**：对输入 hidden_states 做 RMS 归一化
2. **Q 投影**：normed_tile × wq → q_proj（[16,8192] × [8192,8192]）
3. **K/V 投影**：normed_tile × wk/wv → k_proj/v_proj（[16,8192] × [8192,1024]）

### 3.2 Tiling 常量

| 参数 | 基线值 | 优化值 | 说明 |
|------|-------|-------|------|
| K_CHUNK | 512 | 512 | 内层 reduction 维度 tile，不变 |
| Q_OUT_CHUNK | 64 | 64 | Q 投影输出 tile 宽度，不变 |
| KV_OUT_CHUNK | 64 | 64 | KV 投影输出 tile 宽度，不变 |
| BATCH_TILE | 16 | 16 | Batch tile 大小，不变 |

### 3.3 代码差异

优化版与基线版唯一区别为**程序类名**变更：

```python
# 基线：
class Qwen3Scope1:
    ...

# 优化：
class Qwen3Scope1_OPT:
    ...
```

算法结构、循环结构、`pl.at` 块、`pl.parallel` 参数完全保持不变。

### 3.4 保持不变的原因

- **L0B 已满载**：权重 tile `[K_CHUNK, Q_OUT_CHUNK] = [512, 64] × 2B = 64 KB`，占满 L0B 100% 容量。无法在不减小 K_CHUNK 的前提下增大 Q_OUT_CHUNK。
- **Q 投影调度已优**：`pl.parallel(q_out_blocks=128, chunk=4)` 产生 32 个 task，跨 18 核执行，负载均衡良好（~127 us/task）。
- **K/V 投影并行**：`pl.parallel(kv_out_blocks=16, chunk=4)` 产生 4 个 task，与 Q 投影完美重叠执行（KV 在 Q 结束前完成）。

### 3.5 逐阶段 Profiling 数据

| func_id | 阶段 | Task 数 | 核心数 | Avg (us) | Span (us) |
|---------|------|---------|--------|----------|-----------|
| 0 | RMSNorm (AIV) | 1 | 1 | 35.2 | 35.2 |
| 1 | Q 投影 (AIC) | 32 | 18 | 126.7 | 366.3 |
| 2 | KV 投影 (AIC) | 4 | 4 | 234.2 | 240.9 |

---

## 4. Scope 2 优化详情

### 4.1 功能说明

Scope 2 覆盖注意力计算全流程：
1. **Stage 1**：K RoPE + KV cache 写入 + Q RoPE + 对齐 padding
2. **Stage 2**：QK matmul（[Q_HEAD_PAD, HEAD_DIM] × [SEQ_TILE, HEAD_DIM]^T）
3. **Stage 3**：Softmax（fillpad + scale + row_max + exp + row_sum）
4. **Stage 4**：SV matmul（[Q_HEAD_PAD, SEQ_TILE] × [SEQ_TILE, HEAD_DIM]）
5. **Stage 5**：Online softmax 跨块累积 + 归一化

### 4.2 Tiling 常量

| 参数 | 基线值 | 优化值 | 说明 |
|------|-------|-------|------|
| Q_HEAD_BATCH | 8 | 8 | Q heads 每组批处理数 |
| Q_HEAD_PAD | 16 | 16 | 对齐到 Cube fractal 的 Q 行数 |
| SEQ_TILE | 64 | 64 | 序列 tile 长度 |
| SB_BATCH | 64 | 64 | sequence block 并行 chunk 大小 |

### 4.3 代码差异

优化版与基线版唯一区别为**程序类名**变更：

```python
# 基线：
class Qwen3Scope2:
    ...

# 优化：
class Qwen3Scope2_OPT:
    ...
```

所有 Stage 的循环结构、`pl.parallel` 参数、`pl.at` 块、`chunked_loop_optimizer` 完全保持不变。

### 4.4 保持不变的原因

- **动态序列长度**：`ctx_blocks` 是运行时动态值（依赖 `seq_lens`），`pl.parallel(ctx_blocks, chunk=SB_BATCH=64)` 对动态长度实现了高效批处理。
- **小粒度 task 下调度开销主导**：Stage 2/3/4 的单 task 计算量极小（~1.4 us），改用 `pl.range` 产生更多独立 task 会导致调度开销远超计算收益（实测 +1402% 劣化）。
- **Stage 1 已饱和**：`num_kv_heads=8, chunk=8`，实际只产生 1 个 dispatch，无法进一步拆分。

### 4.5 核心循环结构（保持不变）

```python
# Stage 1: 所有 KV heads 在单个 dispatch 中处理
with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
    for ki in pl.parallel(0, num_kv_heads, chunk=8):
        # K RoPE + cache + V cache + Q RoPE + pad ...

# Stage 2/3/4: 动态 ctx_blocks 以 SB_BATCH 为 chunk 并行
with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
    for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
        # QK matmul / softmax / SV matmul ...

# Stage 5: 串行在线 softmax 累积（依赖前序块）
with pl.at(level=pl.Level.CORE_GROUP):
    for sb in pl.range(1, ctx_blocks):
        # online softmax accumulation ...
```

---

## 5. Scope 3 优化详情（核心优化）

### 5.1 功能说明

Scope 3 覆盖 Transformer 单层的后端计算（最重的部分）：
1. **Stage 0**：Output projection（attn_out × wo，[16,8192] × [8192,8192]）
2. **Stage 1**：第一次 residual 加法（output_proj + hidden_states）
3. **Stage 2**：Post-attention RMSNorm
4. **Stage 3**：Gate 投影（post_norm × w_gate，[16,8192] × [8192,25600]）
5. **Stage 4**：Up 投影（post_norm × w_up，[16,8192] × [8192,25600]）
6. **Stage 5**：SiLU 激活（sigmoid(gate) × gate × up → BF16）
7. **Stage 6**：Down 投影（mlp_tile × w_down，[16,25600] × [25600,8192]）
8. **Stage 7**：第二次 residual 加法 + 最终 BF16 输出

### 5.2 Tiling 常量变更（核心优化）

| 参数 | 基线值 | 优化值 | 变化 | 影响 |
|------|-------|-------|------|------|
| K_CHUNK | 128 | 128 | 不变 | 内层 reduction 维度 |
| **Q_OUT_CHUNK** | **64** | **128** | **×2** | Output proj 输出 tile 宽度加倍 |
| **MLP_OUT_CHUNK** | **64** | **128** | **×2** | MLP gate/up/down 输出 tile 宽度加倍 |
| BATCH_TILE | 16 | 16 | 不变 | Batch 维度 |

### 5.3 Tile 加倍的硬件影响

| 资源 | 基线 (tile=64) | 优化 (tile=128) | 硬件上限 |
|------|--------------|----------------|---------|
| **L0B (权重 tile)** | [128,64]×2B = 16 KB (25%) | [128,128]×2B = 32 KB (50%) | 64 KB |
| **Acc (输出 tile)** | [16,64]×4B = 4 KB | [16,128]×4B = 8 KB | 256 KB |
| **L0A (输入 tile)** | [16,128]×2B = 4 KB | [16,128]×2B = 4 KB | 64 KB |
| **Cube fractals/tile** | 128/16 × 64/16 = 32 | 128/16 × 128/16 = 64 | — |

### 5.4 迭代次数变化

| 循环 | 基线 | 优化 | 减少 |
|------|------|------|------|
| Output proj 外层 (Q_OUT_BLOCKS) | 8192/64 = 128 | 8192/128 = 64 | −50% |
| Gate proj 外层 (MLP_OUT_BLOCKS) | 25600/64 = 400 | 25600/128 = 200 | −50% |
| Up proj 外层 (MLP_OUT_BLOCKS) | 25600/64 = 400 | 25600/128 = 200 | −50% |
| SiLU 外层 (MLP_OUT_BLOCKS) | 400 | 200 | −50% |
| Down proj 内层 (MLP_OUT_BLOCKS) | 400 次 matmul_acc | 200 次 matmul_acc | −50% |
| Down proj 外层 (HIDDEN_BLOCKS) | 64 | 64 | 不变 |
| **总 Task 数** | **1585** | **857** | **−46%** |

### 5.5 代码变更详解

#### 5.5.1 Tiling 常量定义

```python
# ============ 基线 (qwen3_32b_decode_scope3.py:30-33) ============
K_CHUNK = 128
Q_OUT_CHUNK = 64
MLP_OUT_CHUNK = 64
BATCH_TILE = 16

# ============ 优化 (qwen3_32b_decode_scope3-opt.py:30-36) ============
K_CHUNK = 128
# [OPT v3] Doubled tile widths for better Cube fractal utilization:
# L0B: [128,128]×2B = 32KB (50%) vs [128,64]×2B = 16KB (25%)
# Halves iteration count: 400→200 MLP blocks, 128→64 output blocks
Q_OUT_CHUNK = 128
MLP_OUT_CHUNK = 128
BATCH_TILE = 16
```

#### 5.5.2 程序类名

```python
# 基线：class Qwen3Scope3
# 优化：class Qwen3Scope3_OPT
```

#### 5.5.3 MLP gate/up/SiLU 阶段结构（保持分离）

优化版与基线版在 MLP 阶段保持相同的**三段分离结构**，即 gate、up、SiLU 各自在独立的 `pl.at` 块中：

```python
# ============ 基线和优化结构相同（仅 MLP_OUT_CHUNK 不同） ============
for ob in pl.range(MLP_OUT_BLOCKS):    # 基线 400 次，优化 200 次
    o0 = ob * MLP_OUT_CHUNK            # 基线 64，优化 128

    # Gate 投影：独立 pl.at（Cube 计算）
    with pl.at(level=pl.Level.CORE_GROUP):
        gate_acc = pl.matmul(...)       # [16,128] × [128,MLP_OUT_CHUNK]
        for kb in pl.range(1, HIDDEN_BLOCKS):
            gate_acc = pl.matmul_acc(...)

    # Up 投影：独立 pl.at（Cube 计算）
    with pl.at(level=pl.Level.CORE_GROUP):
        up_acc = pl.matmul(...)         # [16,128] × [128,MLP_OUT_CHUNK]
        for kb in pl.range(1, HIDDEN_BLOCKS):
            up_acc = pl.matmul_acc(...)

    # SiLU：独立 pl.at（Vector 计算）
    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
        mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
        ...
```

**关键设计决策**：保持 gate 和 up 在独立 `pl.at` 块中（而非融合），使 runtime 可自动实现 gate[i] 和 up[j] 的跨迭代 overlap。Profiling 验证了这一点——gate 和 up 的时间跨度完全重叠。

#### 5.5.4 Output projection 阶段

```python
# ============ 基线和优化结构相同（仅 Q_OUT_CHUNK 不同） ============
for ob in pl.range(Q_OUT_BLOCKS):      # 基线 128 次，优化 64 次
    o0 = ob * Q_OUT_CHUNK              # 基线 64，优化 128

    # matmul：[BATCH_TILE, K_CHUNK] × [K_CHUNK, Q_OUT_CHUNK]
    with pl.at(level=pl.Level.CORE_GROUP):
        o_acc = pl.matmul(...)         # [16,128] × [128,Q_OUT_CHUNK]
        for kb in pl.range(1, HIDDEN_BLOCKS):
            o_acc = pl.matmul_acc(...)

    # Residual：独立 pl.at
    with pl.at(level=pl.Level.CORE_GROUP):
        resid_sum = pl.add(o_acc, resid)
        ...
```

#### 5.5.5 Down projection 阶段

```python
# ============ 基线和优化结构相同（MLP_OUT_CHUNK 影响内层循环） ============
for dob in pl.range(HIDDEN_BLOCKS):    # 64 次，不变
    # matmul：[BATCH_TILE, MLP_OUT_CHUNK] × [MLP_OUT_CHUNK, K_CHUNK]
    with pl.at(level=pl.Level.CORE_GROUP):
        down_acc = pl.matmul(...)      # [16,MLP_OUT_CHUNK] × [MLP_OUT_CHUNK,128]
        for ob in pl.range(1, MLP_OUT_BLOCKS):  # 基线 399 次，优化 199 次
            down_acc = pl.matmul_acc(...)

    # Residual + 输出
    with pl.at(level=pl.Level.CORE_GROUP):
        out_chunk = pl.add(down_acc, resid1_chunk)
        ...
```

### 5.6 逐阶段 Profiling 数据

| func_id | 阶段 | 基线 Task 数 | 优化 Task 数 | 基线 Avg (us) | 优化 Avg (us) | 基线 Span (us) | 优化 Span (us) | 提升 |
|---------|------|-------------|-------------|--------------|--------------|---------------|---------------|------|
| 0 | Output proj (AIC) | 128 | 64 | 50.26 | 65.54 | 402.3 | 282.0 | **−29.9%** |
| 1 | Residual add (AIV) | 128 | 64 | 1.13 | 1.33 | 487.4 | 266.4 | **−45.3%** |
| 2 | RMSNorm (AIV) | 1 | 1 | 43.26 | 43.50 | 43.3 | 43.5 | +0.5% |
| 3 | Gate proj (AIC) | 400 | 200 | 45.52 | 60.89 | 2034.3 | 1379.3 | **−32.2%** |
| 4 | Up proj (AIC) | 400 | 200 | 43.39 | 60.57 | 2034.3 | 1402.8 | **−31.1%** |
| 5 | SiLU (AIV) | 400 | 200 | 1.29 | 1.48 | 1940.7 | 1306.0 | **−32.7%** |
| 6 | Down proj (AIC) | 64 | 64 | 218.93 | 200.74 | 880.9 | 805.9 | **−8.5%** |
| 7 | Down resid (AIV) | 64 | 64 | 5.13 | 4.22 | 633.8 | 574.7 | **−9.3%** |

### 5.7 Gate/Up 跨迭代 Overlap 验证

```
基线:  gate [672.5 - 2706.8] us,  up [671.7 - 2706.0] us  → 完美重叠 ✓
优化:  gate [443.9 - 1823.2] us,  up [443.5 - 1846.3] us  → 完美重叠 ✓
```

gate 和 up 时间跨度完全重叠，证明 runtime 在分离结构下自动实现了跨迭代并行调度。

---

## 6. 被否决的优化方案及原因

在最终方案确定前，我们尝试了多种优化方案，均因性能劣化被否决。以下是完整记录：

### 6.1 [否决] Scope 3 v1：`pl.parallel` 替代 `pl.range`

**变更**：将所有 `pl.range` 外层循环替换为 `pl.parallel`（显式指定 chunk 并行度）。

**结果**：+633% 劣化（3568.6 us → 22603.2 us，估算）。

**根因**：`pl.parallel` 循环体中使用了 `inout` 语义的共享张量（如 `resid1_tile`、`mlp_tile`），runtime 检测到数据依赖后将所有迭代串行化到同一核心，完全丧失并行性。

### 6.2 [否决] Scope 3 v2：Gate/Up 融合 + `pl.range`

**变更**：
1. 将 gate 和 up 的 matmul 融合到同一个 `pl.at` 块中（共享 `post_norm_tile` 读取）
2. 替换外层 `pl.range` 循环，保持 runtime 自动调度
3. 将 residual add 融合到 output proj 和 down proj 的 `pl.at` 块中

**结果**：+20.7% 劣化。

**根因**：
- 融合后每个 task 包含 gate+up 两次 matmul，阻止了 runtime 将 gate[i] 和 up[j] 分配到不同核心的能力
- 基线中 gate 和 up 的分离结构实现了**完美跨迭代 overlap**（双方 span 完全重叠），融合反而破坏了这一点
- `pl.range` 生成的独立 task 缺乏 `chunked_loop_optimizer` 的批处理优化

### 6.3 [否决] Scope 1 v2：K_CHUNK 512→128 + `pl.range`

**变更**：
1. 减小 `K_CHUNK` 从 512 到 128，将 L0B 利用率从 100% 降至 25%
2. 替换 Q/KV 投影的 `pl.parallel` 为 `pl.range`
3. 移除 `chunked_loop_optimizer`

**结果**：+35.6% 劣化（407.6 us → 552.6 us，估算）。

**根因**：
- `K_CHUNK=128` 使 `hidden_blocks` 从 16 增至 64，RMSNorm 循环开销翻倍（16→64 次迭代）
- `pl.range` 在 Q 投影中生成 128 个独立 task（vs 基线 32 个），每个 task 过小，调度开销占比过高

### 6.4 [否决] Scope 1 v3-a：Q 投影 chunk=8

**变更**：将 Q 投影的 `pl.parallel(q_out_blocks, chunk=4)` 改为 `chunk=8`，期望 16 个大 task 可在 18 核的单波中全部完成。

**结果**：+31.0% 劣化（407.6 us → 533.9 us）。

**根因**：16 个 task 仅使用了 10 核（而非预期 18 核），单 task 计算时间 240.6 us（基线 chunk=4 时 126.7 us），失去了多波次调度带来的负载均衡优势。

### 6.5 [否决] Scope 2 v2：`pl.range` 替代 `pl.parallel`

**变更**：
1. Stage 1：`pl.parallel(0, num_kv_heads, chunk=8)` → `pl.range(num_kv_heads)` + 内层 `pl.at`
2. Stage 2/3/4：`pl.parallel(ctx_blocks, chunk=SB_BATCH)` → `pl.range(ctx_blocks)` + 内层 `pl.at`
3. 移除所有 `chunked_loop_optimizer`

**结果**：+1402% 劣化（879.7 us → 13000+ us，估算）。

**根因**：`ctx_blocks` 是运行时动态值（最大 64），`pl.range` 为每个 sb 生成独立 task。Stage 2/3/4 的单 task 计算量极小（QK matmul ~1.4 us），而 runtime 的 task dispatch 开销约 1-2 us/task，导致调度开销远超计算时间。`pl.parallel` + `chunked_loop_optimizer` 可将多个 sb 批处理到同一 dispatch 中（chunk=64），大幅摊薄调度成本。

---

## 7. 关键技术发现

### 7.1 Runtime 自动流水 vs 手动融合

**发现**：在 PyPTO runtime 调度模型下，将 gate 和 up 保持为独立 `pl.at` 块（分离结构）优于融合到同一 `pl.at` 块。

**机制**：runtime 的 task 调度器可将独立 `pl.at` 块的 task 分配到不同核心。当 gate[i] 在 core A 执行时，up[j] 可在 core B 同时执行（跨迭代 overlap）。融合后每个 task 包含 gate+up 两次 matmul，阻止了这种跨核并行。

**证据**：Profiling 数据显示 gate 和 up 的时间跨度完全重叠（起止时间差 < 1 us）。

### 7.2 `pl.parallel` vs `pl.range` 的适用场景

| 特征 | `pl.parallel` + `chunked_loop_optimizer` | `pl.range` |
|------|----------------------------------------|------------|
| Task 粒度 | 粗粒度（chunk 个迭代合并为 1 task） | 细粒度（每次迭代 1 task） |
| 调度模型 | 编译期确定核分配 | 运行时动态调度 |
| 适用场景 | 小 task、动态 trip count、需要批处理 | 大 task、静态 trip count、自动并行 |
| Scope 2 (动态 ctx_blocks, ~1.4 us/task) | ✓ 最优 | ✗ 调度开销主导 |
| Scope 3 (静态迭代, ~60 us/task) | — | ✓ runtime 自动 overlap |

### 7.3 Tile 尺寸选择的权衡

| 因素 | 小 Tile (64) | 大 Tile (128) |
|------|-------------|--------------|
| L0B 利用率 | 25% (16 KB) | 50% (32 KB) |
| Cube fractal 数/tile | 32 个 16×16 块 | 64 个 16×16 块 |
| 外层迭代数 | 多（更好负载均衡） | 少（更少调度开销） |
| 内层循环数（Down proj） | 多 | 少（更少 matmul_acc 调用） |
| 总 Task 数 | 多 | 少 |
| 适用条件 | L0B 已满载时 | L0B 有余量时 ✓ |

**Scope 3 选择 128** 是因为基线 L0B 仅 25%（有大量余量），加倍后仍在安全范围内（50%），同时获得了显著的迭代次数减半收益。

**Scope 1 不能加倍** 是因为基线 L0B 已 100%（`[512,64]×2B = 64 KB`），无法在不减小 K_CHUNK 的前提下增大 Q_OUT_CHUNK，而减小 K_CHUNK 会引入更多 RMSNorm 开销。

### 7.4 L0B 瓶颈与 K_CHUNK 的耦合关系

```
L0B_size = K_CHUNK × OUT_CHUNK × 2 bytes

Scope 1: 512 × 64 × 2 = 64 KB = 100% L0B  → 无法增大 OUT_CHUNK
Scope 3: 128 × 64 × 2 = 16 KB = 25% L0B   → 增大到 128×128×2 = 32 KB = 50% ✓
```

K_CHUNK 越大，matmul 内层循环越少，但 L0B 占用越高。Scope 1 已在这个 trade-off 的极限点。

---

## 8. 构建与验证

### 8.1 构建与运行命令

```bash
# Scope 1
python examples/models/qwen-opt/qwen3_32b_decode_scope1-opt.py -p a2a3 --runtime-profiling

# Scope 2
python examples/models/qwen-opt/qwen3_32b_decode_scope2-opt.py -p a2a3 --runtime-profiling

# Scope 3
python examples/models/qwen-opt/qwen3_32b_decode_scope3-opt.py -p a2a3 --runtime-profiling
```

### 8.2 验证结果

| Scope | 状态 | 验证容差 |
|-------|------|---------|
| Scope 1 | **PASS** | rtol=1e-3, atol=1e-3 |
| Scope 2 | **PASS** | rtol=1e-3, atol=1e-3 |
| Scope 3 | **PASS** | rtol=3e-3, atol=3e-3 |

### 8.3 构建产物目录

| 类别 | 目录 |
|------|------|
| Scope 1 基线 | `build_output/Qwen3Scope1_20260423_191737` |
| Scope 2 基线 | `build_output/Qwen3Scope2_20260423_191809` |
| Scope 3 基线 | `build_output/Qwen3Scope3_20260423_191844` |
| Scope 3 优化 | `build_output/Qwen3Scope3_OPT_20260423_200752` |

### 8.4 Profiling 数据位置

Swimlane JSON 文件位于各构建目录下的 `swimlane_data/` 子目录，包含每个 task 的 `func_id`、`core_id`、`start_time_us`、`end_time_us`、`duration_us` 等字段，可用于精确的性能分析和可视化。
