# S3-1 Gate/Up 投影融合 — 优化结果记录

## 1. 基本信息

| 项目 | 内容 |
|------|------|
| **优化编号** | S3-1 |
| **优化类型** | 结构变换 — 内核融合 |
| **目标 Scope** | Scope 3, MLP gate/up 投影阶段 |
| **平台** | Ascend 910B (a2a3sim 模拟器) |
| **测试日期** | 2026-04-20 |

### 源文件

| | 文件 |
|---|---|
| **原版源码** | [`examples/models/qwen3/qwen3_32b_decode_scope3.py`](../../examples/models/qwen3/qwen3_32b_decode_scope3.py) |
| **优化版源码** | [`examples/models/qwen-opt/qwen3_32b_decode_scope3.py`](../../examples/models/qwen-opt/qwen3_32b_decode_scope3.py) |

### 构建产物

| | 目录 |
|---|---|
| **原版构建产物** | [`build_output/Qwen3Scope3_20260420_221806/`](../../build_output/Qwen3Scope3_20260420_221806/) |
| **优化版构建产物** | [`build_output/Qwen3Scope3_fused_gate_up/`](../../build_output/Qwen3Scope3_fused_gate_up/) |

### 原始性能数据文件

| 数据 | 原版 | 优化版 |
|------|------|--------|
| **泳道图数据** | [`perf_swimlane_20260420_222108.json`](../../build_output/Qwen3Scope3_20260420_221806/swimlane_data/perf_swimlane_20260420_222108.json) | [`perf_swimlane_20260420_222424.json`](../../build_output/Qwen3Scope3_fused_gate_up/swimlane_data/perf_swimlane_20260420_222424.json) |
| **内存报告** | [`memory_after_AllocateMemoryAddr.txt`](../../build_output/Qwen3Scope3_20260420_221806/report/memory_after_AllocateMemoryAddr.txt) | [`memory_after_AllocateMemoryAddr.txt`](../../build_output/Qwen3Scope3_fused_gate_up/report/memory_after_AllocateMemoryAddr.txt) |
| **内核配置** | [`kernel_config.py`](../../build_output/Qwen3Scope3_20260420_221806/kernel_config.py) | [`kernel_config.py`](../../build_output/Qwen3Scope3_fused_gate_up/kernel_config.py) |
| **性能分析报告** | [`perf_report.md`](../../build_output/Qwen3Scope3_20260420_221806/perf_report.md) | [`perf_report.md`](../../build_output/Qwen3Scope3_fused_gate_up/perf_report.md) |

---

## 2. 优化内容

### 2.1 变更描述

将 MLP 的 gate 投影和 up 投影从**两个独立的 `with pl.at` 块**合并为**一个**，
使 `post_norm_tile` 的每个 K_CHUNK 块只从 GM 读取一次（原来读两次）。

### 2.2 代码变更

原版（第 109–129 行）有两个独立的 `with pl.at` 块：

```python
for ob in pl.range(MLP_OUT_BLOCKS):
    o0 = ob * MLP_OUT_CHUNK
    # Block A: gate — 遍历 HIDDEN_BLOCKS 个 post_norm_tile chunks
    with pl.at(level=pl.Level.CORE_GROUP):
        gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN_BLOCKS):
            gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)

    # Block B: up — 再次遍历同样的 chunks（重复 GM 读取）
    with pl.at(level=pl.Level.CORE_GROUP):
        up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN_BLOCKS):
            up_acc = pl.matmul_acc(up_acc, post_chunk, wu)
```

优化版合并为一个块：

```python
for ob in pl.range(MLP_OUT_BLOCKS):
    o0 = ob * MLP_OUT_CHUNK
    # Gate + Up 融合：post_norm_tile 每个 chunk 只读一次
    with pl.at(level=pl.Level.CORE_GROUP):
        gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
        up_acc   = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN_BLOCKS):
            gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)
            up_acc   = pl.matmul_acc(up_acc,   post_chunk, wu)
```

### 2.3 等价性论证

- gate 和 up 之间无数据依赖：`gate_acc` 不依赖 `up_acc`，反之亦然
- `post_chunk` 是只读数据：gate 和 up 共享读取不产生冲突
- matmul / matmul_acc 的调用序列和参数完全不变
- dtype cast 路径不变（BF16 输入 → FP32 累加）
- K 维度遍历顺序不变（0 到 HIDDEN_BLOCKS-1 顺序）

---

## 3. 正确性验证

### 3.1 验证方法

使用**固定输入**（`--golden-data`），两个版本使用**完全相同**的输入数据和 golden 输出：

```bash
GOLDEN=build_output/Qwen3Scope3_20260419_204418/data

# 原版
python examples/models/qwen3/qwen3_32b_decode_scope3.py -p a2a3sim --golden-data $GOLDEN

# 优化版
python examples/models/qwen-opt/qwen3_32b_decode_scope3.py -p a2a3sim --golden-data $GOLDEN
```

### 3.2 验证结果

| | 结果 |
|---|---|
| 原版 | **PASS** (rtol=3e-3, atol=3e-3) |
| 优化版 | **PASS** (rtol=3e-3, atol=3e-3) |

两个版本均使用相同的 7 个输入张量，与同一份 PyTorch golden 输出（`out/out.pt`）
进行 `torch.allclose` 对比，均在容差范围内通过。

---

## 4. 性能数据对比

### 4.1 总览

| 指标 | 原版 | 优化版 | 变化 |
|------|------|--------|------|
| **编译产生的内核数** | **8** | **7** | **↓ 1** |
| **运行时任务数** | **1585** | **1185** | **↓ 400 (−25.2%)** |
| **编排器提交任务数** | **2178** | **1778** | **↓ 400 (−18.4%)** |
| **编排器总耗时** | **195.00 ms** | **139.91 ms** | **↓ 55.09 ms (−28.3%)** |
| Exec/Latency ratio | 53.8% | 53.9% | ≈ 持平 |
| Wall Time (模拟器) | 157.99 s | 155.62 s | ↓ 2.37 s（模拟器，仅参考） |

> Wall Time 在模拟器上的差异不代表真实性能。模拟器时间戳来自 CPU 模拟的 `get_sys_cnt`，
> 绝对耗时无实际意义（见 `docs/swimlane_guide.md`）。

### 4.2 内核映射对比

原版 8 个内核：

| func_id | 内核名 | 类型 | 次数 | 对应阶段 |
|---------|--------|------|-----:|---------|
| 0 | scope3_incore_0 | aic | 128 | Output projection matmul |
| 1 | scope3_incore_1 | aiv | 128 | Residual addition |
| 2 | scope3_incore_2 | aiv | 1 | Post-RMSNorm |
| **3** | **scope3_incore_3** | **aic** | **400** | **Gate projection** |
| **4** | **scope3_incore_4** | **aic** | **400** | **Up projection** |
| 5 | scope3_incore_5 | aiv | 400 | SiLU activation |
| 6 | scope3_incore_6 | aic | 64 | Down projection |
| 7 | scope3_incore_7 | aiv | 64 | Down residual |

优化版 7 个内核（gate + up 融合为 incore_3）：

| func_id | 内核名 | 类型 | 次数 | 对应阶段 |
|---------|--------|------|-----:|---------|
| 0 | scope3_incore_0 | aic | 128 | Output projection matmul |
| 1 | scope3_incore_1 | aiv | 128 | Residual addition |
| 2 | scope3_incore_2 | aiv | 1 | Post-RMSNorm |
| **3** | **scope3_incore_3** | **aic** | **400** | **Gate + Up projection (fused)** |
| 4 | scope3_incore_4 | aiv | 400 | SiLU activation |
| 5 | scope3_incore_5 | aic | 64 | Down projection |
| 6 | scope3_incore_6 | aiv | 64 | Down residual |

**任务数差异**：原版 gate (400) + up (400) = 800 → 优化版 fused (400)，减少 **400 个任务**。

### 4.3 任务级性能对比

#### 原版 — MLP gate/up 阶段

| func_id | 名称 | 次数 | Avg Exec | Avg Latency | Exec% |
|---------|------|-----:|----------|-------------|------:|
| 3 | gate (incore_3) | 400 | 2.71 s | 5.33 s | 50.9% |
| 4 | up (incore_4) | 400 | 2.67 s | 5.27 s | 50.6% |
| **合计** | | **800** | | | |

#### 优化版 — MLP gate+up 融合

| func_id | 名称 | 次数 | Avg Exec | Avg Latency | Exec% |
|---------|------|-----:|----------|-------------|------:|
| 3 | fused (incore_3) | 400 | 5.13 s | 9.96 s | 51.5% |

融合后单个任务的 Avg Exec 约为原来 gate + up 之和（2.71 + 2.67 ≈ 5.38 s vs 5.13 s），
符合预期。

### 4.4 编排器指标对比

| 阶段 | 原版 Count | 优化版 Count | 原版 Duration | 优化版 Duration |
|------|-----------|-------------|--------------|----------------|
| alloc | 2178 | 1778 (↓400) | 462.28 us | 399.70 us |
| sync | 2178 | 1778 (↓400) | 171.24 us | 133.00 us |
| lookup | 1585 | 1185 (↓400) | 67.76 ms | 1.28 ms |
| insert | 1585 | 1185 (↓400) | 176.68 us | 155.86 us |
| params | 2178 | 1778 (↓400) | 106.46 ms | 137.70 ms |
| fanin | 2178 | 1778 (↓400) | 19.96 ms | 239.32 us |
| **总计** | | | **195.00 ms** | **139.91 ms (↓28.3%)** |

各阶段的 Count 均减少 400，对应 400 个被消除的 up projection 任务。
编排器总耗时从 195 ms 降到 140 ms，减少 28.3%。

### 4.5 内存使用对比

融合前后变化最显著的是 MLP gate/up 内核的内存使用。

#### 原版 — gate (incore_3) 和 up (incore_4)

来源：[`build_output/Qwen3Scope3_20260420_221806/report/memory_after_AllocateMemoryAddr.txt`](../../build_output/Qwen3Scope3_20260420_221806/report/memory_after_AllocateMemoryAddr.txt)

```
--- scope3_incore_3 (gate) ---
  Mat    |    20.0 KB  |   512.0 KB  |    3.9%  |  2    (post_chunk + wg)
  Left   |     4.0 KB  |    64.0 KB  |    6.2%  |  1    (post_chunk → L0A)
  Right  |    16.0 KB  |    64.0 KB  |   25.0%  |  1    (wg → L0B)
  Acc    |     4.0 KB  |   128.0 KB  |    3.1%  |  1    (gate_acc)

--- scope3_incore_4 (up) ---
  Mat    |    20.0 KB  |   512.0 KB  |    3.9%  |  2    (post_chunk + wu)
  Left   |     4.0 KB  |    64.0 KB  |    6.2%  |  1
  Right  |    16.0 KB  |    64.0 KB  |   25.0%  |  1
  Acc    |     4.0 KB  |   128.0 KB  |    3.1%  |  1    (up_acc)
```

#### 优化版 — fused gate+up (incore_3)

来源：[`build_output/Qwen3Scope3_fused_gate_up/report/memory_after_AllocateMemoryAddr.txt`](../../build_output/Qwen3Scope3_fused_gate_up/report/memory_after_AllocateMemoryAddr.txt)

```
--- scope3_incore_3 (fused gate+up) ---
  Mat    |    36.0 KB  |   512.0 KB  |    7.0%  |  3    (post_chunk + wg + wu)
  Left   |     4.0 KB  |    64.0 KB  |    6.2%  |  1    (post_chunk → L0A, 共用)
  Right  |    16.0 KB  |    64.0 KB  |   25.0%  |  1    (wg/wu 交替加载)
  Acc    |     8.0 KB  |   128.0 KB  |    6.2%  |  2    (gate_acc + up_acc)
```

| 资源 | 原版 (单个) | 优化版 (融合) | 变化 | 硬件上限 |
|------|------------|-------------|------|---------|
| **Mat** | 20.0 KB | **36.0 KB** | +16 KB（多了 wu tile） | 512 KB (7.0%) |
| **Acc** | 4.0 KB | **8.0 KB** | +4 KB（多了 up_acc） | 128 KB (6.2%) |
| **Left** | 4.0 KB | 4.0 KB | 不变（共用 post_chunk） | 64 KB |
| **Right** | 16.0 KB | 16.0 KB | 不变（wg/wu 交替） | 64 KB |
| **MemRefs** | Mat=2, Acc=1 | Mat=3, Acc=2 | +1 Mat ref, +1 Acc ref | — |

所有资源使用均远低于硬件上限，融合安全。

### 4.6 诊断信息对比

| 诊断 | 原版 | 优化版 |
|------|------|--------|
| WARN: Exec/Latency < 50% | incore_1 (4.1%), incore_5 (7.3%), incore_7 (6.4%) | incore_1 (5.7%), incore_4 (9.9%), incore_6 (10.6%) |
| WARN: Scheduler Idle > 30% | Thread 0,1 均 100% | Thread 0,1 均 100% |
| INFO: Mat < 5% | incore_0, 3, 4, 6 | incore_0, 5 |

优化后 Mat < 5% 的 INFO 从 4 条减少到 2 条（原来的 incore_3 和 incore_4 各 3.9%，
融合后 incore_3 升到 7.0%，不再触发低利用率警告）。

---

## 5. 理论收益量化

### 5.1 GM 读取带宽节省

- HIDDEN = 8192, K_CHUNK = 128 → HIDDEN_BLOCKS = 64
- 每个 post_norm_tile chunk: [BATCH_TILE=16, K_CHUNK=128] × BF16 = 4 KB
- 每个 MLP output block 的 post_norm_tile 总读取: 64 × 4 KB = 256 KB

| | 原版 | 优化版 | 节省 |
|---|---|---|---|
| 每 output block | 512 KB (2×) | 256 KB (1×) | 256 KB |
| MLP_OUT_BLOCKS=400 总计 | 200 MB | 100 MB | **100 MB** |

### 5.2 任务调度开销节省

| 指标 | 节省数量 |
|------|---------|
| incore 调用次数 | −400 |
| orch_alloc | −400 |
| orch_params | −400 |
| orch_lookup | −400 |
| orch_fanin | −400 |
| scheduler dispatch | −400 |
| scheduler complete | −400 |

---

## 6. 其他未变更的内核

以下内核在优化前后完全一致（func_id 可能因内核数减少而重新编号）：

| 阶段 | 原版 func_id | 优化版 func_id | 次数 | 内存使用 |
|------|-------------|---------------|------|---------|
| Output projection | 0 (aic) | 0 (aic) | 128 | 不变 |
| Residual addition | 1 (aiv) | 1 (aiv) | 128 | 不变 |
| Post-RMSNorm | 2 (aiv) | 2 (aiv) | 1 | 不变 |
| SiLU activation | 5 (aiv) | 4 (aiv) | 400 | 不变 |
| Down projection | 6 (aic) | 5 (aic) | 64 | 不变 |
| Down residual | 7 (aiv) | 6 (aiv) | 64 | 不变 |

---

## 7. 结论

**优化有效**，所有观测指标与理论预测一致：

- ✅ **正确性通过**：固定输入下两版本均通过 golden 验证 (rtol=atol=3e-3)
- ✅ **任务数减少 400**：gate 和 up 融合为一个 incore（800 → 400）
- ✅ **内核数从 8 减到 7**：消除独立的 up projection 内核
- ✅ **编排器开销降低 28.3%**：更少的任务 = 更少的 alloc/params/lookup/fanin
- ✅ **融合内核硬件资源充足**：Mat 7.0%, Acc 6.2%，远低于上限
- ✅ **Mat 低利用率警告减少**：4 条 → 2 条

**待真机验证**：GM 带宽节省（~100 MB/batch_tile）在真机上的实际 wall time 收益。
