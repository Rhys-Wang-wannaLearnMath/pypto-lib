# S3-1：Gate/Up 投影融合优化记录

## 概要

| 项目 | 内容 |
|------|------|
| **优化编号** | S3-1 |
| **目标文件** | `examples/models/qwen-opt/qwen3_32b_decode_scope3_opt.py` |
| **原始文件** | `examples/models/qwen3/qwen3_32b_decode_scope3.py` |
| **优化类型** | 结构变换 — 内核融合 |
| **涉及 Scope** | Scope 3, Stage 3–5（MLP gate/up 投影） |
| **正确性风险** | 极低（数学严格等价） |

---

## 1. 优化内容

### 1.1 变更描述

将 MLP 的 gate 投影和 up 投影从**两个独立的 `with pl.at` 块**合并为**一个**，
使 `post_norm_tile` 的每个 K_CHUNK 块在每个 MLP output block 中只被从 GM 读取一次
（而非两次）。

### 1.2 变更前（第 107–129 行）

```python
for ob in pl.range(MLP_OUT_BLOCKS):
    o0 = ob * MLP_OUT_CHUNK

    # Block A: gate projection — 遍历所有 HIDDEN_BLOCKS 个 post_norm_tile chunks
    with pl.at(level=pl.Level.CORE_GROUP):
        post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
        wg_0 = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
        gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN_BLOCKS):
            k0 = kb * K_CHUNK
            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
            wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
            gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)

    # Block B: up projection — 再次遍历同样的 chunks（重复 GM 读取）
    with pl.at(level=pl.Level.CORE_GROUP):
        post_chunk_0 = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, 0])
        wu_0 = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [0, o0])
        up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN_BLOCKS):
            k0 = kb * K_CHUNK
            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
            wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
            up_acc = pl.matmul_acc(up_acc, post_chunk, wu)
```

### 1.3 变更后

```python
for ob in pl.range(MLP_OUT_BLOCKS):
    o0 = ob * MLP_OUT_CHUNK

    # Gate + Up 融合：post_norm_tile 每个 chunk 只读一次
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

### 1.4 SiLU 块无变更

紧随其后的 SiLU activation 块完全不变：

```python
    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
        mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
        mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
        mlp_tile = pl.assemble(mlp_tile, mlp_chunk_bf16, [0, o0])
```

---

## 2. 数学等价性证明

### 2.1 运算序列对比

| 步骤 | 融合前 | 融合后 | 等价？ |
|------|--------|--------|:------:|
| 1 | `gate_acc = matmul(post_chunk_0, wg_0)` | 相同 | ✓ |
| 2 | `gate_acc += matmul_acc(post_chunk, wg)` × (HIDDEN_BLOCKS-1) | 相同 | ✓ |
| 3 | `up_acc = matmul(post_chunk_0, wu_0)` | 相同 | ✓ |
| 4 | `up_acc += matmul_acc(post_chunk, wu)` × (HIDDEN_BLOCKS-1) | 相同 | ✓ |

唯一区别：步骤 1-2 和 3-4 从"先全部做完 gate 再全部做完 up"变为"交替做 gate 和 up"。
但每步的输入数据 (`post_chunk`, `wg`, `wu`) 和操作 (`matmul`, `matmul_acc`) 完全相同。

### 2.2 数据依赖分析

- `gate_acc` 只被 gate 的 `matmul`/`matmul_acc` 写入 → 不依赖 `up_acc`
- `up_acc` 只被 up 的 `matmul`/`matmul_acc` 写入 → 不依赖 `gate_acc`
- `post_chunk` 是只读的 → gate 和 up 共享读取不产生冲突

**结论**：gate 和 up 之间无数据依赖，融合是安全的。

### 2.3 精度路径

- 融合前后的 dtype cast 序列完全相同（均为 BF16 输入 → FP32 累加）
- 累加顺序不变（K 维度仍从 0 到 HIDDEN_BLOCKS-1 顺序遍历）
- 预期：**零精度差异**（bit-exact 等价）

---

## 3. 硬件约束验证

融合后单个 incore scope 需要同时持有 gate 和 up 的操作数。

| 资源 | 融合前单个 scope | 融合后单个 scope | 硬件上限 | 状态 |
|------|-----------------|-----------------|---------|------|
| **Acc (L0C)** | 4 KB (1 × [16,64] FP32) | 8 KB (2 × [16,64] FP32) | 128 KB | ✓ 6.3% |
| **Mat (L1)** | 20 KB (post + w) | 36 KB (post + wg + wu) | 512 KB | ✓ 7.0% |
| **L0A** | 4 KB | 4 KB（共用 post_chunk） | 64 KB | ✓ |
| **L0B** | 16 KB | 16 KB（wg/wu 交替加载） | 64 KB | ✓ |

---

## 4. 量化收益分析

参数值（Qwen3-32B, Scope 3）：
- HIDDEN = 8192, K_CHUNK = 128 → HIDDEN_BLOCKS = 64
- INTERMEDIATE = 25600, MLP_OUT_CHUNK = 64 → MLP_OUT_BLOCKS = 400
- BATCH_TILE = 16
- 每个 post_norm_tile chunk: [16, 128] BF16 = 4 KB

### 4.1 GM 读取带宽节省

| 指标 | 融合前 | 融合后 | 节省 |
|------|--------|--------|------|
| 每 output block 的 post_norm_tile 读取 | 2 × 64 × 4 KB = **512 KB** | 1 × 64 × 4 KB = **256 KB** | 256 KB |
| 400 个 output block 总读取 | **200 MB** | **100 MB** | **100 MB** |

### 4.2 任务数减少

| 指标 | 融合前 | 融合后 | 减少 |
|------|--------|--------|------|
| 每 output block 的 incore 调用数 | 3（gate + up + SiLU） | 2（gate_up_fused + SiLU） | 1 |
| 400 个 output block 总 incore 调用 | 1200 | 800 | **400** |

### 4.3 编排器开销减少

每减少一个 incore 调用，编排器节省一次完整的 orch_alloc → orch_params → orch_insert
流程。400 个减少的调用 = 400 次编排开销节省。

---

## 5. 验证方法

### 5.1 正确性验证

优化后的程序使用与原始版本**完全相同的 golden_fn、tensor_specs、rtol/atol**：

```bash
python examples/models/qwen-opt/qwen3_32b_decode_scope3_opt.py -p a2a3sim
```

验证通过条件：`[RUN] PASS`，rtol=3e-3, atol=3e-3。

### 5.2 性能数据验证

```bash
# 带 profiling 运行
python examples/models/qwen-opt/qwen3_32b_decode_scope3_opt.py -p a2a3sim --runtime-profiling

# 分析性能数据
python tools/perf_analyzer.py build_output/Qwen3Scope3_fused_gate_up/ -v
```

对比原始版本性能数据，关注：
- **Task Statistics 表**：总任务数应减少约 400
- **Orchestrator Phase Breakdown**：各阶段 Count 应减少
- **Scheduler Thread Statistics**：dispatch/complete 次数应减少

### 5.3 自定义输出目录

通过 `RunConfig.compile` 中的 `output_dir` 参数，本优化版本的编译产物固定输出到
`build_output/Qwen3Scope3_fused_gate_up/`，便于与原始版本的
`build_output/Qwen3Scope3_<timestamp>/` 进行对比。

---

## 6. 关于 output_dir 自定义

`golden/runner.py` 中的 `RunConfig.compile` docstring 明确列出 `output_dir` 为
支持的编译选项之一：

```python
@dataclass
class RunConfig:
    compile: dict[str, Any]  # Kwargs forwarded to pypto.ir.compile
                             # (e.g. backend_type, dump_passes, output_dir, ...)
```

因此可以通过以下方式自定义性能文件的输出目录名：

```python
config=RunConfig(
    compile=dict(
        dump_passes=True,
        output_dir="build_output/my_custom_name",
    ),
    ...
)
```

编译产物（包括 `swimlane_data/perf_swimlane_*.json`、`kernel_config.py`、
`report/memory_after_AllocateMemoryAddr.txt` 等）都会输出到该目录下。
