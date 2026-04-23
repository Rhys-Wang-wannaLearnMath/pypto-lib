# Qwen3-32B Decode Scope 优化 v3 结果报告

> **日期**：2026-04-23  
> **硬件**：Ascend 910B (a2a3)，binned 设备（20 AICore，实际可用 18，block_dim 从 24 截断至 18）  
> **框架**：PyPTO IR Compiler + tensormap_and_ringbuffer runtime  
> **模型**：Qwen3-32B 单层 Decode（BATCH=16, HIDDEN=8192, INTERMEDIATE=25600）

---

## 1. 性能总览

| Scope | 基线 (us) | OPT v3 (us) | 提升 | Task 数 (基线→优化) |
|-------|-----------|-------------|------|---------------------|
| Scope 1 | 407.6 | 407.6 | +0.0% | 37 → 37 |
| Scope 2 | 879.7 | 871.0 | −1.0% | 529 → 529 |
| **Scope 3** | **3568.6** | **2636.7** | **−26.1%** | 1585 → 857 |

---

## 2. 优化策略与实验历程

### 2.1 尝试过但被否决的方案

| 方案 | 变更 | 结果 | 原因分析 |
|------|------|------|---------|
| v1: `pl.parallel` → `pl.range` (S3) | 替换所有 `pl.parallel` 为 `pl.range` | +633% 性能劣化 | `pl.parallel` 的 shared `inout` 张量导致 runtime 序列化执行 |
| v2: gate/up 融合 + `pl.range` (S3) | 融合 gate+up 到单个 `pl.at`，使用 `pl.range` | +20.7% 劣化 | 融合阻止了 gate[i]/up[j] 跨迭代 overlap（基线 gate+up 完美重叠） |
| v2: K_CHUNK 512→128 (S1) | 减小 K_CHUNK 缓解 L0B 瓶颈 | +35.6% 劣化 | RMSNorm 开销翻倍（64 次→16 次循环），总任务数 4× |
| v2: `pl.range` 替代 `pl.parallel` (S2) | 注意力阶段改用 `pl.range` | +1402% 劣化 | 动态 ctx_blocks 生成大量 1.4us 微任务，调度开销占主导 |
| v3: chunk=8 Q proj (S1) | 增大 chunk 减少调度轮次 | +31.0% 劣化 | 16 个大任务仅占用 10 核，核心利用率下降 |

### 2.2 最终采用的优化方案

#### Scope 3: Tile 尺寸加倍 (OPT v3)

核心变更：`Q_OUT_CHUNK = 64 → 128`, `MLP_OUT_CHUNK = 64 → 128`

| 指标 | 基线 (64) | 优化 (128) | 变化 |
|------|----------|-----------|------|
| L0B 利用率 | [128,64]×2B = 16KB (25%) | [128,128]×2B = 32KB (50%) | +100% |
| MLP 外层迭代 | 400 | 200 | −50% |
| Output proj 迭代 | 128 | 64 | −50% |
| Down proj 内层循环 | 400 次 matmul_acc | 200 次 matmul_acc | −50% |
| 总 Task 数 | 1585 | 857 | −46% |

**关键洞察**：保持 gate/up 分离结构（非融合），让 runtime 自动实现跨迭代 overlap。基线中 gate 和 up 的时间跨度完全重叠 `[672-2706] us`，说明 runtime 已自动实现完美流水。

#### Scope 1 & 2: 保持基线结构

经过多轮实验验证，Scope 1 和 Scope 2 的基线代码已接近最优：
- **Scope 1**: L0B 已满载（100%），无法增大输出 tile；`pl.parallel(chunk=4)` 已实现最优核心调度
- **Scope 2**: `pl.parallel(ctx_blocks, chunk=SB_BATCH)` 高效批处理动态长度序列

---

## 3. Scope 3 逐阶段分析

| 阶段 | 基线 span (us) | OPT v3 span (us) | 提升 | 说明 |
|------|---------------|------------------|------|------|
| Output proj (AIC) | 402.3 | 282.0 | −29.9% | 128→64 任务，avg 50→66 us |
| Residual add (AIV) | 487.4 | 266.4 | −45.3% | 64 任务，与 output proj 流水 |
| RMSNorm (AIV) | 43.3 | 43.5 | +0.5% | 不变，单任务 |
| Gate proj (AIC) | 2034.3 | 1379.3 | −32.2% | 400→200 任务，avg 46→61 us |
| Up proj (AIC) | 2034.3 | 1402.8 | −31.1% | 400→200 任务，完美 overlap 保持 |
| SiLU (AIV) | 1940.7 | 1306.0 | −32.7% | 与 gate+up 流水 |
| Down proj (AIC) | 880.9 | 805.9 | −8.5% | 内层 400→200 matmul_acc |
| Down resid (AIV) | 633.8 | 574.7 | −9.3% | 与 down proj 流水 |

### 3.1 Gate/Up 完美 Overlap 验证

```
基线:  gate [672.5 - 2706.8],  up [671.7 - 2706.0]  → 完美重叠 ✓
OPTv3: gate [443.9 - 1823.2],  up [443.5 - 1846.3]  → 完美重叠 ✓
```

Tile 加倍后，gate/up 仍保持完美 overlap，证明分离结构是正确选择。

---

## 4. 关键发现

### 4.1 Runtime 自动流水 vs 手动融合

**发现**：在 PyPTO runtime 中，将 gate 和 up 保持为独立 `pl.at` 块（分离结构）优于将它们融合到同一 `pl.at` 块，因为：
- 分离时：runtime 可在不同核心上并行调度 gate[i] 和 up[j]（跨迭代重叠）
- 融合时：每个融合任务包含 gate+up 计算，阻止了跨迭代并行

### 4.2 Tile 尺寸选择的权衡

| 指标 | 小 Tile (64) | 大 Tile (128) | 最优选择 |
|------|-------------|--------------|---------|
| L0B 利用率 | 25% | 50% | 128 ✓ |
| Task 数量 | 多（更好负载均衡） | 少（更少调度开销） | 取决于任务粒度 |
| Cube 效率 | 4 个 16×16 fractal | 8 个 16×16 fractal | 128 ✓ |
| 内层循环次数 | 多 | 少（同 tile 内更多数据复用） | 128 ✓ |

### 4.3 Scope 1 L0B 限制

Scope 1 使用 K_CHUNK=512，权重 tile [512,64]×2B = 64KB 已填满 L0B。无法通过增大 Q_OUT_CHUNK 来优化，因为 [512,128]×2B = 128KB 超出限制。减小 K_CHUNK 虽能释放 L0B 空间，但会增加 RMSNorm 和内层循环开销，实测反而更慢。

---

## 5. 结论

1. **Scope 3 实现 26.1% 性能提升**，通过将 tile 尺寸从 64 加倍到 128，减少任务数 46%，同时保持 runtime 自动流水的完美 gate/up overlap
2. **Scope 1 和 Scope 2 已接近硬件极限**，基线结构是最优的
3. **核心教训**：在 PyPTO runtime 中，保持独立的小粒度 `pl.at` 块比手动融合更优，因为 runtime 的自动调度能实现跨迭代完美流水

### 5.1 构建目录

| Scope | 基线目录 | OPT v3 目录 |
|-------|---------|------------|
| Scope 1 | `Qwen3Scope1_20260423_191737` | 结构等同基线 |
| Scope 2 | `Qwen3Scope2_20260423_191809` | 结构等同基线 |
| Scope 3 | `Qwen3Scope3_20260423_191844` | `Qwen3Scope3_OPT_20260423_200752` |
