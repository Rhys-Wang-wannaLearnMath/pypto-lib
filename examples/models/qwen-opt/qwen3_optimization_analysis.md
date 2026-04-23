# Qwen3-32B Decode 优化机会分析报告

基于 Ascend 910B (a2a3) 硬件约束，对 `examples/models/qwen3/` 下 Scope 1/2/3
及完整 decode 程序进行源码级优化机会分析。本文档不涉及代码修改，仅提供分析结论和
优先级排序，供后续优化迭代参考。

> **本文档同时覆盖满配与 binned 两类硬件**：涉及核心数的推演均以
> **双列**（`满配 910B3：24 AIC` / `本机 910B3 Bin6：20 AIC`）展示。
> 本机情况详见 §1.1；binned 设备的自动 `block_dim` 截断逻辑详见
> `docs/npu_hang_diagnosis_and_fix.md`。

---

## 执行摘要

| 优先级 | 编号 | 优化 | 预期收益（满配 24 核 / 本机 18 核） | 实现难度 |
|--------|------|------|---------|---------|
| **P0** | **S3-1** | Gate/Up 投影融合 | GM 读取减半（~100 MB/batch_tile，与核心数无关） | 低 |
| P1 | S2-4 | QK+Softmax+SV per-sb 融合 | 消除 2 个全局同步屏障（与核心数无关） | 中 |
| P1 | S3-2/3/4 | Scope 3 三处顺序循环并行化 | 满配 3–17x / 本机 ~2.7–14x | 低 |
| P2 | S1-2 | Q/K/V 投影并行化 (tile 版本) | 满配 ~5x / 本机 ~4x | 低 |
| P2 | S2-1/2 | Batch / Group 循环并行化 | 短 ctx_len 时收益显著 | 低 |
| P3 | S1-1 | K_CHUNK 标准化 | 视编译器行为 | 低 |
| P3 | X-1 | Scope 1 输出精度降低 | 减 50% 跨 scope 带宽 | 低（需精度验证） |
| P4 | X-3 | 跨 Scope Pipeline | 高 | 高（需框架支持） |

---

## 目录

1. [硬件约束速查](#1-硬件约束速查)
   - [1.1 本机 Binned 设备影响说明](#11-本机-binned-设备影响说明)
2. [文件结构与版本关系](#2-文件结构与版本关系)
3. [Scope 1 — RMSNorm + Q/K/V 投影](#3-scope-1--rmsnorm--qkv-投影)
4. [Scope 2 — RoPE + KV Cache + Attention](#4-scope-2--rope--kv-cache--attention)
5. [Scope 3 — Output Projection + MLP](#5-scope-3--output-projection--mlp)
6. [跨 Scope 优化机会](#6-跨-scope-优化机会)
7. [优化优先级排序](#7-优化优先级排序)
8. [附录：关键数据量参考](#附录关键数据量参考)

---

## 1. 硬件约束速查

以下为 Ascend 910B 处理器的关键资源约束，所有优化方案均需在此框架内验证可行性。
**单核资源（UB/L1/L0A/L0B/L0C）在满配与 binned 设备上完全相同**；仅核心数和
`block_dim` 因 binning 而变化。

| 资源 | 满配 910B3 | 本机 910B3 Bin6 | 约束说明 |
|------|:---:|:---:|---------|
| **TILELET** (Vector) | **2 KB** (2048 B) | 同左 | 每个 vector 操作数（add/mul/exp/rsqrt 等）≤ 2048 B |
| **TILE** (Cube) | **16 KB** (16384 B) | 同左 | 每个 matmul 操作数 ≤ 16384 B |
| **Vec (UB)** | 192 KB | 同左 | AIV（vector 核心）的工作缓冲区 |
| **Mat (L1)** | 512 KB | 同左 | AIC（cube 核心）的矩阵缓冲区 |
| **L0A** | 64 KB | 同左 | Cube 单元的左矩阵输入寄存器文件 |
| **L0B** | 64 KB | 同左 | Cube 单元的右矩阵输入寄存器文件 |
| **Acc (L0C)** | 128 KB | 同左 | Cube 单元的累加器 |
| **AIC 核心数** | **24** | **20** | cube 矩阵计算单元（core_id 0–23 / 0–19） |
| **AIV 核心数** | **24** | **20** | vector 向量计算单元（core_id 24–47 / 24–43） |
| **AICPU 线程** | 4 | 4 | 3 个调度器线程 + 1 个编排器线程 |
| **有效 `block_dim`** | **24** | **18** | 须被 `scheduler_thread_num`(=3) 整除，详见 §1.1 |

> 详见 `docs/performance_analysis_guide.md` 第 5 节"内存使用指标"。

---

## 1.1 本机 Binned 设备影响说明

### 设备辨识

本机实测参数：

| 项 | 值 | 取值来源 |
|----|----|---------|
| 板卡型号 | `IT21HMDA_Bin6` | `npu-smi info` |
| Board ID | `0x62` | — |
| AICore 集群数 | **20** | `rtGetAiCoreCount()` |
| 物理握手核数 | 60（= 20 × 3，每 cluster 1 AIC + 2 AIV 硬件） | device log `handshake_all_cores` |
| Profiler 逻辑核数 | 40（= 20 AIC + 20 AIV，1:1 映射） | `core_to_thread` |

### `block_dim` 自动截断规则

pypto 运行时会在 `execute_on_device()` 入口处自动执行：

1. 调用 `rtGetAiCoreCount()` 查询实际 AIC 数量 `hw_max`
2. 若用户/默认 `block_dim > hw_max`，则截断至
   `new_bd = ⌊hw_max / scheduler_thread_num⌋ × scheduler_thread_num`
3. `scheduler_thread_num = aicpu_thread_num − 1`（一个线程留给 orchestrator）

对本机：`aicpu_thread_num=4` → `scheduler_thread_num=3`，
`hw_max=20` → `new_bd = (20 // 3) × 3 = 18`，因此 **本机一切样例均以 `block_dim=18` 实际运行**。

### 对并行度的直接影响

- **一轮处理块数**：24 → **18**（满配 → 本机）
- **并行化优化的等效加速比**：满配 `⌈N/24⌉` → 本机 `⌈N/18⌉`
- **对收益排序不造成结构性变化**：并行化收益略降（~75%），但排序相对不变

### 对 GM 带宽 / 内存的影响

- 单次并行"扇出"的工作集从 24 份减为 18 份（-25%），瞬时 GM 带宽需求略降
- 与核心数**无关**的优化（如 S3-1 融合、S2-4 屏障消除）在本机与满配收益完全相同
- 硬件约束表（Mat/UB/L0*/Acc 容量）在本机与满配**完全相同**，所有"硬件约束检查"段落无需重算

### 交叉引用

- Hang 诊断与修复细节：`docs/npu_hang_diagnosis_and_fix.md`
- `block_dim` 自动截断代码：`python/pypto/runtime/device_runner.py` 的 `execute_on_device()`

---

## 2. 文件结构与版本关系

`examples/models/qwen3/` 下存在同一模型的多个实现版本，理解其关系是优化分析的前提。

### 2.1 Scope 级文件

| 文件 | 覆盖范围 | 函数类型 | 关键 tiling 常量 |
|------|---------|---------|-----------------|
| `qwen3_32b_decode_scope1.py` | Scope 1：RMSNorm + Q/K/V 投影 | Opaque | K_CHUNK=512, BATCH_TILE=16 |
| `qwen3_32b_decode_scope1_tile.py` | Scope 1（tile DSL 版本） | InCore | K_CHUNK=128, BATCH_TILE=16 |
| `qwen3_32b_decode_scope2.py` | Scope 2：RoPE + KV Cache + Attention | Opaque | SEQ_TILE=64, Q_HEAD_BATCH=8 |
| `qwen3_32b_decode_scope3.py` | Scope 3：Output Proj + MLP | Opaque | K_CHUNK=128, MLP_OUT_CHUNK=64 |

### 2.2 完整层文件

| 文件 | 说明 | 关键差异 |
|------|------|---------|
| `qwen3_32b_decode.py` | 标准 Opaque 完整单层 decode | SCOPE1_K_CHUNK=512, MLP_OUT_CHUNK=256 |
| `qwen3_32b_decode_tile.py` | Tile DSL 完整单层 decode | 显式 load/move/matmul |
| `qwen3_32b_decode_mixed.py` | Mixed（Opaque + 显式初始化） | BATCH_TILE=16, 显式零初始化 |
| `qwen3_32b_decode_tilelet.py` | 双重最大化 TILELET/TILE 利用率 | BATCH_TILE=4, K_CHUNK=128 |

### 2.3 参考文档

| 文件 | 内容 |
|------|------|
| `qwen3_tilelet.md` | 所有 tile 尺寸的硬件约束验证表 |
| `qwen3_npu_test_guide.md` | 真机测试指南 |
| `qwen3_simulator_test_guide.md` | 模拟器测试指南 |

---

## 3. Scope 1 — RMSNorm + Q/K/V 投影

**计算流程**：hidden_states → RMSNorm → normed_tile → Q/K/V matmul → q_proj/k_proj/v_proj

### 3.1 优化机会 S1-1：K_CHUNK 不一致

**问题描述**：

不同文件中 Scope 1 的 K_CHUNK（reduction 维度分块大小）不一致：

| 文件 | K_CHUNK | Vector tile `[BATCH_TILE, K_CHUNK]` FP32 | TILELET 状态 |
|------|---------|----------------------------------------|-------------|
| `qwen3_32b_decode_scope1.py` | 512 | [16, 512] × 4 B = **32 KB** | 超限 16x |
| `qwen3_32b_decode_scope1_tile.py` | 128 | [16, 128] × 4 B = **8 KB** | 超限 4x |
| `qwen3_32b_decode.py` (Scope 1) | 512 | [16, 512] × 4 B = **32 KB** | 超限 16x |
| `qwen3_tilelet.md` 最优配置 | 128 | [4, 128] × 4 B = **2 KB** | **刚好满** ✓ |

**代码位置**：

`qwen3_32b_decode_scope1.py` 第 30 行：
```python
K_CHUNK = 512
```

`qwen3_32b_decode.py` 第 45 行：
```python
SCOPE1_K_CHUNK = 512
```

**分析**：

- Opaque 函数类型（`pl.FunctionType.Opaque`）下，编译器负责自动 tiling，因此 K_CHUNK=512
  不直接违反硬件约束。但编译器需要将 [16, 512] FP32 = 32 KB 拆分为 16 个 2 KB TILELET，
  增加了生成代码的复杂度和潜在的调度开销。

- `qwen3_tilelet.md` 中经过详细分析的最优配置是 BATCH_TILE=4, K_CHUNK=128，实现了
  vector TILELET 和 cube TILE 的**双重 100% 利用率**。

**优化方向**：将 Opaque 版本的 SCOPE1_K_CHUNK 从 512 降为 128，减少编译器自动拆分负担。

**预期收益**：减少编译器微代码复杂度；改善 Vec 内存局部性。

**风险**：K_CHUNK=512 让编译器看到更大的操作粒度（仅 16 次外层循环 vs 64 次），
降低 K_CHUNK 可能增加调度开销。需要 profiling 验证 net effect。

**正确性风险**：**极低**。K_CHUNK 只影响 reduction 维度的分块粒度，不改变累加结果
（sum of products 的分组方式不同，但浮点误差在容差范围内）。

---

### 3.2 优化机会 S1-2：Q/K/V 投影缺少并行化（tile 版本）

**问题描述**：

`qwen3_32b_decode_scope1_tile.py` 的编排函数中，Q/K/V 的 output block 循环是顺序的：

```python
# qwen3_32b_decode_scope1_tile.py 第 185–192 行
for ob in pl.range(q_out_blocks):       # 顺序，q_out_blocks=128
    q0 = ob * Q_OUT_CHUNK
    q_proj = self.q_proj_reduce(normed_tile, wq, b0, q0, q_proj)

for ob in pl.range(kv_out_blocks):      # 顺序，kv_out_blocks=16
    kv0 = ob * KV_OUT_CHUNK
    k_proj = self.kv_proj_reduce(normed_tile, wk, b0, kv0, k_proj)
    v_proj = self.kv_proj_reduce(normed_tile, wv, b0, kv0, v_proj)
```

**问题**：128 个 Q output block 和 16 个 K/V output block 的迭代完全独立（每个 block
写入 q_proj/k_proj/v_proj 的不同列区域），但使用顺序 `pl.range`，编排器只能逐个提交任务。

**优化方向**：改用 `pl.parallel` 或添加 `chunk` 参数。

**预期收益（双列推演）**：

| 阶段 | N blocks | 满配（24 核）轮数 | 本机（18 核）轮数 | 满配加速比 | 本机加速比 |
|------|:---:|:---:|:---:|:---:|:---:|
| Q 投影 | 128 | `⌈128/24⌉` = 6 | `⌈128/18⌉` = **8** | ~21x | **~16x** |
| K/V 投影 | 16 | 1（16<24） | 1（16<18） | — | — |

Q 投影 128 blocks 可并行到所有 AIC；K/V 投影 16 blocks 均在单轮内完成（两种配置下都不饱和）。

**正确性风险**：**极低**。每个 output block 只写入 `q_proj[b0, q0:q0+Q_OUT_CHUNK]`，
区域不重叠。

---

### 3.3 优化机会 S1-3：RMSNorm 两次遍历（无优化空间）

RMSNorm 需要两次遍历 hidden_states：
1. 第一次：计算 `sum of squares`（reduction，loop-carried dependency）
2. 第二次：应用 `inv_rms × gamma`

**结论**：RMSNorm 的数学特性要求先完成全量 squared sum 才能计算 inv_rms，
两次遍历**无法融合**。当前实现已是最优结构。

---

## 4. Scope 2 — RoPE + KV Cache + Attention

**计算流程**：
1. K RoPE + cache write, V cache write, Q RoPE + pad
2. QK matmul（所有 sequence blocks）
3. Softmax（所有 sequence blocks）
4. SV matmul（所有 sequence blocks）
5. Online-softmax accumulation + normalization

### 4.1 优化机会 S2-1：Batch 循环顺序执行

**代码位置**：

```python
# qwen3_32b_decode_scope2.py 第 77 行
for b in pl.range(batch):  # 顺序，batch=16
    ctx_len = pl.tensor.read(seq_lens, [b])
    # ... 整个 attention 逻辑（RoPE + QK + softmax + SV + accumulation）
```

**问题**：16 个 batch item 的 attention 计算完全独立（各自有不同的 seq_len、访问
k_cache/v_cache 的不同区域），但 batch 循环是顺序的。

**优化方向**：`for b in pl.range(batch)` → `for b in pl.range(batch, parallel=True)`
或使用 `chunk`。

**预期收益（双列推演）**：
- 每个 batch item 内部已有 `pl.parallel` 循环（QK/softmax/SV 的 `sb` 维度），
  当 `ctx_blocks` 较大时（长 ctx_len），内部并行度已饱和核心：
  - 满配：`ctx_blocks ≥ 24` 时饱和
  - **本机：`ctx_blocks ≥ 18` 时饱和**（饱和阈值降低，利好短 ctx）
- 当 `ctx_len` 较短（`ctx_blocks < 18/24`）时，内部并行度不足，
  此时 batch 并行可显著提升核心利用率。
- 具体数值：batch=16，若 `ctx_blocks=8`，满配有 16 空闲核心，本机有 10 空闲核心；
  两种配置下 batch 并行均能把有效并行度拉到 `min(batch × ctx_blocks, cores)`。

**风险**：batch 并行需要所有 batch item 的中间张量
（`all_raw_scores`, `all_exp_padded`, `all_oi_tmp`, `all_cur_mi`, `all_cur_li`）
同时存在，GM 内存压力增大。需要评估总中间内存是否可接受。

---

### 4.2 优化机会 S2-2：Group 循环顺序执行

**代码位置**：

```python
# qwen3_32b_decode_scope2.py 第 147 行
for gi in pl.range(total_q_groups):  # 顺序，total_q_groups=8
    kvh = gi // q_groups
    # ... QK matmul, softmax, SV matmul, online softmax accumulation
```

**问题**：8 个 query group 的 attention 完全独立（不同的 KV head group），但顺序执行。

**优化方向**：`for gi in pl.range(total_q_groups)` → 添加 `parallel=True`。

**预期收益（双列推演）**：
- `total_q_groups=8`，两种配置下都不饱和单轮（8 < 18 < 24），因此 8 个 group 在满配和本机都**单轮完成**
- 单独看此优化：满配加速 ~8x，本机加速 ~8x（相同）
- 但与 S2-1 batch 并行组合（batch × group = 16 × 8 = 128 工作单元）时：
  - 满配：`⌈128/24⌉` = 6 轮
  - **本机：`⌈128/18⌉` = 8 轮**
  - 短 ctx_len 时两种配置收益都明显

**正确性风险**：**低**。每个 group 写入 `attn_row` 的不同列区域
（`q_base * head_dim` 起始的 `Q_HEAD_BATCH * head_dim` 列），不重叠。

---

### 4.3 优化机会 S2-3：中间张量按 max_ctx_blocks 分配（无优化空间）

```python
# qwen3_32b_decode_scope2.py 第 154–158 行
max_ctx_blocks = (max_seq + SEQ_TILE - 1) // SEQ_TILE  # = 64
all_raw_scores = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, SEQ_TILE], dtype=pl.FP32)
```

**结论**：PyPTO 的编程模型要求张量维度在编译时确定，无法使用运行时 ctx_blocks
动态分配。这是框架层面的约束，**无法在用户代码层优化**。

---

### 4.4 优化机会 S2-4：QK + Softmax + SV per-sb 融合

**这是 Scope 2 最大的优化机会。**

**问题描述**：

当前实现将 QK matmul (Stage 2)、softmax (Stage 3)、SV matmul (Stage 4) 分成三个
独立的 `with pl.at` 块，每个块内部各自有 `pl.parallel` 循环：

```python
# qwen3_32b_decode_scope2.py 第 159–209 行

# Stage 2: QK matmul — 所有 sb 块
with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
    for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
        raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
        all_raw_scores = pl.assemble(all_raw_scores, raw_scores, [sb * Q_HEAD_PAD, 0])

# Stage 3: softmax — 所有 sb 块
with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
    for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
        # ... scale, row_max, exp, row_sum
        all_exp_padded = pl.assemble(...)
        all_cur_mi = pl.assemble(...)
        all_cur_li = pl.assemble(...)

# Stage 4: SV matmul — 所有 sb 块
with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
    for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
        oi_tmp = pl.matmul(exp_tile, v_tile, out_dtype=pl.FP32)
        all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [sb * Q_HEAD_PAD, 0])
```

**关键观察**：每个 sb block 的 softmax (Stage 3) 只依赖该 sb 的 raw_scores (Stage 2)，
每个 sb block 的 SV matmul (Stage 4) 只依赖该 sb 的 exp_scores (Stage 3)。
三个 stage 之间没有跨 sb 的依赖。

**但**当前实现用三个独立的 `with pl.at` 块，引入了 **2 个全局同步屏障**：
Stage 2 全部完成 → Stage 3 全部完成 → Stage 4 全部完成。

**优化方向**：将三步融合为单个 parallel loop，每个 sb block 内部连续执行 QK → softmax → SV：

```python
with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
    for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
        # QK matmul
        raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
        # Softmax
        scores = pl.mul(pl.fillpad(raw_scores, ...), attn_scale)
        cur_mi = pl.row_max(scores)
        exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
        exp_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
        cur_li = pl.row_sum(pl.cast(exp_bf16, target_type=pl.FP32))
        # SV matmul
        oi_tmp = pl.matmul(exp_bf16, v_tile, out_dtype=pl.FP32)
        # Assemble results
        all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [sb * Q_HEAD_PAD, 0])
        all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [sb * Q_HEAD_BATCH, 0])
        all_cur_li = pl.assemble(all_cur_li, cur_li, [sb * Q_HEAD_BATCH, 0])
```

**预期收益**：
- 消除 2 个全局同步屏障
- 减少 `all_raw_scores` 和 `all_exp_padded` 的 GM 读写（中间结果可在 L1 内流转）
- `all_raw_scores` 张量可能完全不需要分配到 GM

**硬件约束检查**：融合后单个 incore 需要同时持有 QK 和 SV 的操作数：
- QK matmul: A=[Q_HEAD_PAD, head_dim] BF16 = [16,128]×2 = 4 KB, B=[SEQ_TILE, head_dim]^T BF16 = [64,128]×2 = 16 KB
- SV matmul: A=[Q_HEAD_PAD, SEQ_TILE] BF16 = [16,64]×2 = 2 KB, B=[SEQ_TILE, head_dim] BF16 = [64,128]×2 = 16 KB
- K tile 和 V tile 可以共享 L1 空间（QK 完成后释放 K tile，加载 V tile）
- 中间 softmax 在 Vec 中完成，scores [Q_HEAD_BATCH, SEQ_TILE] FP32 = [8,64]×4 = 2 KB (TILELET MAX)
- Mat 总计峰值: ~36 KB << 512 KB ✓

**正确性风险**：**低**。每个 sb block 内部的计算逻辑完全不变，仅消除了 sb block 之间
不必要的全局同步。Stage 5 (online softmax accumulation) 仍然独立执行，
依赖关系不受影响。

**Binned 设备影响**：此优化的收益来自**同步屏障消除**和**L1 中间结果复用**，
与核心数**无关**。本机（18 核）与满配（24 核）收益完全相同；
硬件约束检查表中的 Mat 峰值（~36 KB）在两种配置下都远低于 L1 容量（512 KB）。

---

## 5. Scope 3 — Output Projection + MLP

**计算流程**：
1. Output projection: `attn_out × wo` → FP32 accumulation
2. Residual addition: `o_proj + hidden_states`
3. Post-attention RMSNorm
4. MLP gate projection: `normed × w_gate`
5. MLP up projection: `normed × w_up`
6. SiLU activation: `gate * sigmoid(gate) * up`
7. Down projection: `mlp × w_down`
8. Final residual: `down + resid1`

### 5.1 优化机会 S3-1：Gate/Up 投影融合

**这是整个 Qwen3 decode 程序中最容易实现且收益最明确的优化。**

**问题描述**：

MLP 的 gate 投影和 up 投影在两个独立的 `with pl.at` 块中，分别遍历 `post_norm_tile`
的所有 K_CHUNK 块。同一份输入数据被从 GM 读取**两次**。

```python
# qwen3_32b_decode_scope3.py 第 109–135 行

for ob in pl.range(MLP_OUT_BLOCKS):  # MLP_OUT_BLOCKS=400 (K_CHUNK=128, MLP_OUT_CHUNK=64)
    o0 = ob * MLP_OUT_CHUNK

    # Block A: Gate projection — 遍历所有 HIDDEN_BLOCKS 个 post_norm_tile chunks
    with pl.at(level=pl.Level.CORE_GROUP):
        post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])  # 从 GM 读取
        wg_0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
        gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN_BLOCKS):
            k0 = kb * K_CHUNK
            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])  # 从 GM 读取
            wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
            gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)

    # Block B: Up projection — 再次遍历同样的 post_norm_tile chunks
    with pl.at(level=pl.Level.CORE_GROUP):
        post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])  # 重复读取！
        wu_0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
        up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN_BLOCKS):
            k0 = kb * K_CHUNK
            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])  # 重复读取！
            wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
            up_acc = pl.matmul_acc(up_acc, post_chunk, wu)
```

**问题量化**：
- 每个 output block 的 `post_norm_tile` 读取量 = HIDDEN_BLOCKS × [BATCH_TILE, K_CHUNK] BF16
  = 64 × [16, 128] × 2 B = 64 × 4 KB = 256 KB
- 当前读取 2 次（gate + up）= 512 KB
- MLP_OUT_BLOCKS × 重复读取 = 400 × 256 KB = **100 MB GM 读取带宽浪费**（每个 batch_tile）

**优化方向**：将 gate 和 up 投影融合到同一个 incore scope 中：

```python
for ob in pl.range(MLP_OUT_BLOCKS):
    o0 = ob * MLP_OUT_CHUNK
    with pl.at(level=pl.Level.CORE_GROUP):
        post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
        wg_0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
        wu_0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
        gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
        up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN_BLOCKS):
            k0 = kb * K_CHUNK
            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
            wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
            wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
            gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)
            up_acc = pl.matmul_acc(up_acc, post_chunk, wu)
```

**硬件约束验证**：

| 资源 | 融合后用量 | 硬件上限 | 利用率 |
|------|-----------|---------|--------|
| **Acc (L0C)** | gate_acc [16,64] FP32 = 4 KB + up_acc [16,64] FP32 = 4 KB = **8 KB** | 128 KB | 6.3% ✓ |
| **Mat (L1)** | post_chunk [16,128] BF16 = 4 KB + wg [128,64] BF16 = 16 KB + wu [128,64] BF16 = 16 KB = **36 KB** | 512 KB | 7.0% ✓ |
| **L0A** | post_chunk move = 4 KB（gate 和 up 共用） | 64 KB | 6.3% ✓ |
| **L0B** | wg/wu 交替 move，单次 16 KB | 64 KB | 25.0% ✓ |

**结论**：硬件资源完全充足，融合是安全的。

**正确性风险**：**极低**。gate 和 up 投影使用同一 `post_norm_tile` chunk 的只读数据，
matmul 操作序列完全不变，只是共享了输入数据的 L1 副本。数学上严格等价。

**Binned 设备影响**：此优化的收益来自**GM 读取量减半**（每 output block 从 512 KB 降到 256 KB），
与核心数**无关**。本机（18 核）与满配（24 核）节省量完全相同（每 batch_tile ~100 MB）；
硬件约束验证表中的单核资源使用量（L0A/L0B/L0C/L1）在两种配置下都保持不变。

---

### 5.2 优化机会 S3-2：Output Projection 顺序循环

**代码位置**：

```python
# qwen3_32b_decode_scope3.py 第 67–87 行
for ob in pl.range(Q_OUT_BLOCKS):  # 顺序，Q_OUT_BLOCKS = 8192 // 64 = 128
    o0 = ob * Q_OUT_CHUNK
    with pl.at(level=pl.Level.CORE_GROUP):
        # matmul_acc 遍历所有 K_CHUNK 块
        ...
    with pl.at(level=pl.Level.CORE_GROUP):
        resid_sum = pl.add(o_acc, resid)
        resid1_tile = pl.assemble(resid1_tile, resid_sum, [0, o0])
```

**分析**：128 个 output block 完全独立——每个 block 写入 `resid1_tile` 的不同列区域
`[0:BATCH_TILE, o0:o0+Q_OUT_CHUNK]`，无重叠。

**优化方向**：`pl.range(Q_OUT_BLOCKS)` → `pl.parallel(Q_OUT_BLOCKS, chunk=...)`

**预期收益（双列推演）**：

| 配置 | 核心数 | 轮数 | 理论加速比 | 实际加速比（受调度/带宽限制） |
|------|:---:|:---:|:---:|:---:|
| 满配 | 24 | `⌈128/24⌉` = 6 | ~21x | ~5x |
| **本机** | **18** | `⌈128/18⌉` = **8** | **~16x** | **~4x** |

**注意**：`resid1_tile` 后续被 RMSNorm 读取，需要所有 output block 完成。
parallel 在 output projection 阶段结束后自然形成同步点，因此安全。

---

### 5.3 优化机会 S3-3：MLP Output Block 顺序循环

**代码位置**：

```python
# qwen3_32b_decode_scope3.py 第 109 行
for ob in pl.range(MLP_OUT_BLOCKS):  # 顺序
    # gate matmul, up matmul, SiLU, assemble → mlp_tile[0:BATCH_TILE, o0:o0+MLP_OUT_CHUNK]
```

**分析**：MLP_OUT_BLOCKS 个 output block 独立写入 `mlp_tile` 的不同列区域。

**预期收益（双列推演）**：

| 配置 | 核心数 | MLP_OUT_CHUNK=64 (400 blocks) | MLP_OUT_CHUNK=256 (100 blocks) |
|------|:---:|:---:|:---:|
| 满配 | 24 | `⌈400/24⌉` = 17 轮（~24x） | `⌈100/24⌉` = 5 轮（~20x） |
| **本机** | **18** | `⌈400/18⌉` = **23 轮（~17x）** | `⌈100/18⌉` = **6 轮（~17x）** |

对应顺序循环数：400 / 100 轮。**本机加速比相对满配略降（~75%）**，但绝对收益仍然显著。

---

### 5.4 优化机会 S3-4：Down Projection 顺序循环

**代码位置**：

```python
# qwen3_32b_decode_scope3.py 第 138 行
for dob in pl.range(HIDDEN_BLOCKS):  # 顺序，HIDDEN_BLOCKS = 8192 // 128 = 64
    d0 = dob * K_CHUNK
    # down matmul_acc + residual → out[b0:, d0:d0+K_CHUNK]
```

**分析**：64 个 down output block 独立写入 `out` 的不同列区域。

**预期收益（双列推演）**：

| 配置 | 核心数 | 轮数 | 理论加速比 |
|------|:---:|:---:|:---:|
| 满配 | 24 | `⌈64/24⌉` = 3 | ~21x |
| **本机** | **18** | `⌈64/18⌉` = **4** | **~16x** |

对应顺序 64 轮。

---

### 5.5 优化机会 S3-5：MLP_OUT_CHUNK 不一致

| 文件 | MLP_OUT_CHUNK | Cube tile `[K_CHUNK, MLP_OUT_CHUNK]` BF16 | TILE 状态 |
|------|---------------|------------------------------------------|----------|
| `scope3.py` | 64 | [128, 64] × 2 = 16 KB | ✓ 满 |
| `decode.py` | 256 | [128, 256] × 2 = 64 KB | **超限 4x** |
| `tilelet.md` | 64 | [128, 64] × 2 = 16 KB | ✓ 满 |

**分析**：`decode.py` 使用 MLP_OUT_CHUNK=256 在 Opaque 模式下由编译器自动拆分为
16 KB tiles。较大的 chunk 减少了外层循环次数（100 vs 400），但编译器内部仍需拆分。
Net effect 需要 profiling 验证。

---

## 6. 跨 Scope 优化机会

### 6.1 优化机会 X-1：Scope 1 → Scope 2 精度降低

**当前**：Scope 1 输出 q_proj/k_proj/v_proj 为 FP32，Scope 2 在 RoPE 后才 cast 到 BF16。

**优化方向**：在 Scope 1 直接输出 BF16，节省跨 scope 的 GM 带宽：
- q_proj: [16, 8192] FP32 = 512 KB → BF16 = 256 KB，节省 50%
- k_proj + v_proj: [16, 1024] × 2 × FP32 = 128 KB → BF16 = 64 KB

**风险**：RoPE 在 FP32 下计算更准确（涉及 sin/cos 乘法和加减）。需要精度测试验证
BF16 中间精度是否在 rtol=1e-3 容差内可接受。

---

### 6.2 优化机会 X-2：全局 Zero-Init 开销

`qwen3_32b_decode_mixed.py` 中显式零初始化所有中间 tensor：

```python
# qwen3_32b_decode_mixed.py 第 155–168 行
with pl.at(level=pl.Level.CORE_GROUP):
    for ob in pl.range(Q_OUT_BLOCKS):
        q0 = ob * Q_OUT_CHUNK
        zero_1 = pl.full([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32, value=0.0)
        q_proj = pl.assemble(q_proj, zero_1, [0, q0])
        attn_out = pl.assemble(attn_out, pl.cast(zero_1, target_type=pl.BF16), [0, q0])
        normed_buf = pl.assemble(normed_buf, pl.cast(zero_1, target_type=pl.BF16), [0, q0])
```

**分析**：如果 `pl.create_tensor` 已隐式零初始化（在 Opaque 版本中），这些显式循环
是冗余的。如果不是，则框架层面支持 tensor 创建时自动零初始化可以省去这些开销。

---

### 6.3 优化机会 X-3：跨 Scope Pipeline

**当前**：Scope 1 → Scope 2 → Scope 3 严格顺序执行。

**分析**：由于数据依赖（Scope 2 依赖 Scope 1 的输出，Scope 3 依赖 Scope 2 的输出），
Scope 间无法并行。但如果 batch 维度可以 pipeline：
- Scope 1 处理 batch[1] 的同时，Scope 2 处理 batch[0]
- 需要框架层面支持 cross-scope pipeline scheduling

**可行性**：**高实现难度**。需要框架支持 fine-grained task dependency tracking，
当前 PyPTO 编排器不支持。列为长期优化方向。

---

## 7. 优化优先级排序

按 **预期收益 / 实现风险** 比排序。**收益列已拆为"满配 24 核 / 本机 18 核"双列**。

| 优先级 | 编号 | 优化 | 满配（24 核）收益 | 本机（18 核）收益 | 实现难度 | 正确性风险 | 所需验证 |
|--------|------|------|------------------|------------------|---------|-----------|---------|
| **P0** | **S3-1** | **Gate/Up 投影融合** | **高：GM 读取减半 (~100 MB)** | **同左（与核心数无关）** | **低** | **极低** | golden 对比 |
| P1 | S2-4 | QK+Softmax+SV per-sb 融合 | 高：消除 2 个同步屏障 | 同左（与核心数无关） | 中 | 低 | golden 对比 + 多输入 |
| P1 | S3-3 | MLP Output Block 并行化 | 中高：~17–20x 该阶段 | **中高：~17x 该阶段（chunk=64/256 差异小）** | 低 | 极低 | golden 对比 |
| P1 | S3-2 | Output Projection 并行化 | 中高：~21x 该阶段 | **中高：~16x 该阶段** | 低 | 极低 | golden 对比 |
| P1 | S3-4 | Down Projection 并行化 | 中：~21x 该阶段 | **中：~16x 该阶段** | 低 | 极低 | golden 对比 |
| P2 | S1-2 | Q/K/V 投影并行化 (tile) | 中：~21x Q 投影 | **中：~16x Q 投影（K/V 两种配置都单轮完成）** | 低 | 极低 | golden 对比 |
| P2 | S2-1 | Batch 循环并行化 | 中（短 ctx_len 高） | **中（短 ctx_len 饱和阈值更低：18 vs 24）** | 低 | 低 | 多输入 + 边界测试 |
| P2 | S2-2 | Group 循环并行化 | 中（短 ctx_len 高） | 同左（8 group 两配置单轮完成） | 低 | 低 | 多输入 |
| P3 | S1-1 | K_CHUNK 标准化 | 低~中 | 同左（编译器行为） | 低 | 低 | profiling |
| P3 | X-1 | Scope 1 输出精度降低 | 低~中 | 同左（与核心数无关） | 低 | 中 | 精度验证 |
| P4 | X-3 | 跨 Scope Pipeline | 高 | 同左（框架层） | 高 | 中 | 框架支持 |
| — | S1-3 | RMSNorm 两次遍历 | 无 | — | — | — | — |
| — | S2-3 | max_ctx_blocks 张量分配 | 无（框架限制） | — | — | — | — |

### 7.1 Binned 设备的排序变化观察

- **排序相对不变**：P0/P1/P2 分层与编号次序完全保持
- **加速比规律**：所有并行化类优化（S1-2, S3-2/3/4）在本机降为满配的 **~75%**（= 18/24），
  因为一轮可完成的 block 数从 24 变为 18
- **与核心数无关的优化（S3-1, S2-4, X-1）收益不变**，在 binned 设备上**相对吸引力更高**
- **对 S2-1/S2-2 有潜在利好**：本机内部并行饱和阈值从 `ctx_blocks=24` 降至 `ctx_blocks=18`，
  意味着更多短 ctx 场景可以"单层并行"即饱和，batch 并行相对收益更小

---

## 附录：关键数据量参考

### A.1 Qwen3-32B 模型参数

| 参数 | 值 |
|------|-----|
| BATCH | 16 |
| MAX_SEQ | 4096 |
| HIDDEN | 8192 (= NUM_HEADS × HEAD_DIM = 64 × 128) |
| KV_HIDDEN | 1024 (= NUM_KV_HEADS × HEAD_DIM = 8 × 128) |
| INTERMEDIATE | 25600 |
| NUM_HEADS | 64 |
| NUM_KV_HEADS | 8 |
| HEAD_DIM | 128 |

### A.2 张量大小

| Tensor | Shape | DType | Size |
|--------|-------|-------|------|
| hidden_states | [16, 8192] | BF16 | 256 KB |
| wq | [8192, 8192] | BF16 | 128 MB |
| wk / wv | [8192, 1024] | BF16 | 16 MB each |
| q_proj | [16, 8192] | FP32 | 512 KB |
| k_proj / v_proj | [16, 1024] | FP32 | 64 KB each |
| k_cache / v_cache | [524288, 128] | BF16 | 128 MB each |
| wo | [8192, 8192] | BF16 | 128 MB |
| w_gate / w_up | [8192, 25600] | BF16 | 400 MB each |
| w_down | [25600, 8192] | BF16 | 400 MB |
| post_norm_tile (per batch_tile) | [16, 8192] | BF16 | 256 KB |
| mlp_tile (per batch_tile) | [16, 25600] | BF16 | 800 KB |
| resid1_tile (per batch_tile) | [16, 8192] | FP32 | 512 KB |

### A.3 S3-1 Gate/Up 融合量化收益

| 指标 | 当前 | 融合后 | 节省 |
|------|------|-------|------|
| 每 output block post_norm_tile GM 读取 | 512 KB (2×) | 256 KB (1×) | 256 KB |
| MLP_OUT_BLOCKS=400 总读取 | 200 MB | 100 MB | **100 MB** |
| MLP_OUT_BLOCKS=100 总读取 | 50 MB | 25 MB | **25 MB** |

> 本项收益与核心数**无关**，满配（24）与本机（18）相同。

### A.4 block_dim 对本机的影响速查

本机 `block_dim` 经自动截断后为 **18**，对所有"N blocks 并行"段的推演影响汇总：

| 优化点 | N blocks | 满配（24）轮数 | 本机（18）轮数 | 轮数比 | 本机加速比 / 满配加速比 |
|--------|:---:|:---:|:---:|:---:|:---:|
| S1-2 Q 投影 | 128 | 6 | 8 | 1.33x | 0.75 |
| S1-2 K/V 投影 | 16 | 1 | 1 | 1.0x | 1.0（两配置均单轮） |
| S3-2 Output Proj | 128 | 6 | 8 | 1.33x | 0.75 |
| S3-3 MLP Out (chunk=64) | 400 | 17 | 23 | 1.35x | 0.74 |
| S3-3 MLP Out (chunk=256) | 100 | 5 | 6 | 1.20x | 0.83 |
| S3-4 Down Proj | 64 | 3 | 4 | 1.33x | 0.75 |

**一般规律**：`本机轮数 / 满配轮数 = ⌈N/18⌉ / ⌈N/24⌉`；
当 `N ≡ 0 (mod 24)` 且 `N ≡ 0 (mod 18)` 时，比值严格等于 `4/3 ≈ 1.333`。
其余场景比值在 `1.0 ~ 1.33` 之间波动，取决于余数。

**如何解读本文档的数值**：
- 若你在**满配设备**运行：以"满配"列为准
- 若你在**本机或其它 binned 设备**运行：以"本机"列为准，或套用 `⌈N / block_dim⌉` 公式自行推演
- 核心数**无关的**优化（S3-1 / S2-4 / X-1）无需换算

**若迁移到其它 binned 型号**：将 `rtGetAiCoreCount()` 返回值代入上述公式重算即可。
常见 910B 系列 AICore 数（典型值）：

| 型号 | AIC 集群数 | 对齐后 `block_dim` | 与满配比 |
|------|:---:|:---:|:---:|
| 910B1（满配） | 24 | 24 | 100% |
| 910B3 Bin6（本机） | 20 | 18 | 75% |
| 910B3 Bin8（假设） | 16 | 15 | 62.5% |
| 910B3 Bin10（假设） | 12 | 12 | 50% |

> 具体 bin 型号对应核心数以实际 `rtGetAiCoreCount()` 返回为准。
