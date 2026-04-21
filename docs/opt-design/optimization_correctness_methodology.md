# PyPTO 优化正确性保障方法论

在 PyPTO 框架下对 Qwen3 等模型进行源码级性能优化时，如何在保证正确性的前提下
完成一次完整优化迭代。本文档定义了从基线建立到回归保护的全流程方法论，
以及每个阶段的具体验证步骤和判定标准。

---

## 目录

1. [核心原则](#1-核心原则)
2. [现有正确性基础设施](#2-现有正确性基础设施)
3. [Phase 0 — 优化前准备](#3-phase-0--优化前准备)
4. [Phase 1 — 数学等价性验证](#4-phase-1--数学等价性验证)
5. [Phase 2 — 实施与验证](#5-phase-2--实施与验证)
6. [Phase 3 — 精度影响分析](#6-phase-3--精度影响分析)
7. [Phase 4 — 跨平台验证](#7-phase-4--跨平台验证)
8. [Phase 5 — 回归保护](#8-phase-5--回归保护)
9. [实战案例：S3-1 Gate/Up 融合](#9-实战案例s3-1-gateup-融合)
10. [完整检查清单](#10-完整检查清单)

---

## 1. 核心原则

**优化不改变语义**：任何性能优化必须在数学上等价于原始实现，在硬件上产生
容差范围内的相同结果。

### 1.1 优化分类与风险等级

| 优化类型 | 典型操作 | 正确性风险 | 验证要求 |
|----------|----------|-----------|---------|
| **参数调优** | 修改 K_CHUNK、BATCH_TILE、chunk size | 低 | 整除性检查 + golden 对比 |
| **并行化** | `pl.range` → `pl.parallel`/`chunk` | 中 | 迭代独立性证明 + golden 对比 + 多输入 |
| **结构变换** | 内核融合、循环重排、stage 合并 | 高 | 数学等价证明 + 硬件约束验证 + golden 对比 + 边界测试 |

### 1.2 黄金法则

1. **一次只做一个变更**：不要同时修改 tiling 参数 + 并行化 + 融合
2. **固定输入先行**：先用相同输入对比，再用随机输入验证
3. **容差不能放宽**：优化后必须通过原始容差（rtol/atol）
4. **性能必须提升**：如果优化没有带来可测量的性能改善，不值得引入额外复杂度

---

## 2. 现有正确性基础设施

### 2.1 Golden 验证框架

项目在 `golden/` 模块下提供完整的验证链：

```
build_program() → ir.compile() → execute_compiled() → validate_golden()
     ↑                                                       ↑
 PyPTO 程序                                         torch.allclose(actual, expected, rtol, atol)
     ↑                                                       ↑
 golden_fn(tensors)                                    golden_outputs
```

**核心组件**：

| 模块 | 文件 | 功能 |
|------|------|------|
| `TensorSpec` | `golden/tensor_spec.py` | 定义张量名称、形状、dtype、初始化策略、是否为输出 |
| `validate_golden` | `golden/validation.py` | 基于 `torch.allclose` 的多输出验证，含详细错误报告 |
| `run` | `golden/runner.py` | 编译→生成输入→执行→计算 golden→验证 的一站式入口 |
| `RunConfig` | `golden/runner.py` | 配置容差（rtol/atol）、编译选项、运行时选项 |

### 2.2 `run()` 函数的完整流程

```python
# golden/runner.py 中 run() 的执行流程：

def run(program, tensor_specs, config, golden_fn, golden_data, runtime_dir):
    # 1. 编译（或跳过，使用 runtime_dir）
    compiled = ir.compile(program, **config.compile)

    # 2. 生成输入（或从 golden_data 加载）
    if golden_data:
        tensors = load_from_cache(golden_data)     # 固定输入
    else:
        tensors = {spec.name: spec.create_tensor() for spec in tensor_specs}
        save_tensors(compiled.output_dir / "data/in", input_snapshot)  # 自动持久化

    # 3. 设备执行
    execute_compiled(work_dir, ordered_tensors, **config.runtime)

    # 4. 计算 golden（或从缓存加载）
    if golden_data:
        golden_outputs = load_from_cache(golden_data / "out")
    else:
        scratch = clone(input_snapshot)            # 克隆隔离！
        golden_fn(scratch)                         # PyTorch 参考实现
        golden_outputs = extract_outputs(scratch)
        save_tensors(compiled.output_dir / "data/out", golden_outputs)

    # 5. 验证
    validate_golden(device_outputs, golden_outputs, rtol=config.rtol, atol=config.atol)
```

**关键特性**：

- **输入克隆隔离**：`golden_fn` 收到的是输入的 clone，不会被设备执行污染。
  这在 `tests/golden/test_runner.py::TestGoldenFnPath::test_golden_fn_sees_cloned_inputs_not_live_tensors`
  中有专门的测试验证。

- **数据自动持久化**：每次运行自动保存 `data/in/*.pt`（输入快照）和
  `data/out/*.pt`（golden 输出），可用于回归对比。

- **Golden 缓存**：`golden_data` 参数支持读取预存的输入/输出数据，跳过随机生成
  和 golden 计算，实现**可复现的固定输入验证**。

- **多输出验证**：所有标记 `is_output=True` 的 TensorSpec 均被逐一验证。

### 2.3 `validate_golden()` 的错误报告

```python
# golden/validation.py

def validate_golden(outputs, golden, rtol=1e-5, atol=1e-5):
    for name, actual in outputs.items():
        if not torch.allclose(actual, golden[name], rtol=rtol, atol=atol):
            # 报告：
            # - 不匹配元素数量 / 总元素数量
            # - rtol 和 atol 值
            # - 前 20 个不匹配元素的位置、实际值和期望值
            raise AssertionError(...)
```

### 2.4 各 Scope 的现有容差配置

| 文件 | rtol | atol | 原因 |
|------|------|------|------|
| `qwen3_32b_decode_scope1.py` | 1e-3 | 1e-3 | RMSNorm + BF16 投影 |
| `qwen3_32b_decode_scope1_tile.py` | 1e-3 | 1e-3 | Tile DSL 版本 |
| `qwen3_32b_decode_scope2.py` | 1e-3 | 1e-3 | Attention 含 BF16 round-trip |
| `qwen3_32b_decode_scope3.py` | 3e-3 | 3e-3 | MLP 多级 matmul 累积误差更大 |
| `qwen3_32b_decode.py` | 3e-3 | 3e-3 | 全层，三个 scope 累积误差 |

**规则**：优化后的程序**必须通过与原始程序相同的容差**。如果需要放宽容差，
必须有明确的技术原因和审批。

### 2.5 测试矩阵

**CI 自动化** (`.github/workflows/ci.yml`)：

| Job | 平台 | 测试范围 | 触发条件 |
|-----|------|---------|---------|
| `unit-tests` | CPU-only (mock pypto) | `tests/golden/` 单元测试 | push/PR to main |
| `sim` | a2a3sim / a5sim | `examples/beginner/` + `examples/intermediate/` | push/PR to main |
| `a2a3` | 910B 真机 | 同上 | push/PR to main (self-hosted) |

**手动测试** (`examples/models/qwen3/run_all_tests.sh`)：

```bash
bash examples/models/qwen3/run_all_tests.sh [-p PLATFORM] [--profiling]
```

测试顺序：scope 级 → 完整层 → tile 变体 → prefill → 训练。

> **注意**：Qwen3 模型测试 **不在 CI 中**，必须手动运行。这意味着 Qwen3 优化的
> 回归保护完全依赖手动执行。

---

## 3. Phase 0 — 优化前准备

### 3.1 建立 Golden 基线

在做任何修改之前，运行原始版本并保存 golden 数据：

```bash
# 1. 运行目标 scope 的原始版本
python examples/models/qwen3/qwen3_32b_decode_scope3.py -p a2a3sim

# 2. 运行后，build_output/<ProgramName>_<timestamp>/data/ 下自动保存了：
#    - data/in/*.pt   — 随机生成的输入张量
#    - data/out/*.pt  — golden 参考输出
#
# 3. 将其拷贝为 golden 基线（不可变）
BASELINE_DIR="golden_baseline/scope3_$(date +%Y%m%d)"
mkdir -p "$BASELINE_DIR"
cp -r build_output/Qwen3Scope3_*/data/* "$BASELINE_DIR/"
echo "基线保存到: $BASELINE_DIR"
```

**为什么需要基线**：保存的 `data/in/*.pt` 包含此次运行的随机输入。后续所有验证
**必须使用相同输入**才能进行 bit-exact 对比（排除输入差异导致的误差）。

### 3.2 记录基线性能（推荐）

```bash
# 带 profiling 运行（仅在真机上有意义）
python examples/models/qwen3/qwen3_32b_decode_scope3.py -p a2a3 --runtime-profiling

# 分析基线性能
python tools/perf_analyzer.py build_output/Qwen3Scope3_*/ -o baseline_perf.md -v
```

基线性能报告中关注的关键指标：
- **Wall Time**：端到端耗时
- **Exec/Latency ratio**：越高越好（低值说明调度开销大）
- **Scheduler idle%**：< 10% 为佳
- **Orchestrator orch_alloc%**：< 10% 为佳
- **Memory utilization**：各内核各内存空间的利用率

---

## 4. Phase 1 — 数学等价性验证

在动手写代码之前，先在文档中证明优化的数学等价性。

### 4.1 参数调优验证清单

修改 tiling 常量（K_CHUNK、MLP_OUT_CHUNK、BATCH_TILE 等）时：

- [ ] **整除性验证**
  - `HIDDEN % K_CHUNK == 0`
  - `INTERMEDIATE % MLP_OUT_CHUNK == 0`
  - `HIDDEN % Q_OUT_CHUNK == 0`
  - `KV_HIDDEN % KV_OUT_CHUNK == 0`
  - `BATCH % BATCH_TILE == 0`

- [ ] **TILELET 约束**（vector 操作数 ≤ 2048 B）
  - `BATCH_TILE × K_CHUNK × sizeof(FP32) ≤ 2048`
  - `BATCH_TILE × OUT_CHUNK × sizeof(FP32) ≤ 2048`
  - 注：Opaque 模式下编译器自动处理，但明确知道更好

- [ ] **TILE 约束**（cube 操作数 ≤ 16384 B）
  - `K_CHUNK × OUT_CHUNK × sizeof(BF16) ≤ 16384`
  - `SEQ_TILE × HEAD_DIM × sizeof(BF16) ≤ 16384`
  - 注：Tile DSL 版本必须满足

- [ ] **Vec 容量约束**
  - `BATCH_TILE × HIDDEN × sizeof(element) × double_buffer_factor ≤ 192 KB`
  - 例：BATCH_TILE=4, HIDDEN=8192, FP32: 4 × 8192 × 4 = 128 KB, ×2 (double buffer) = 256 KB > 192 KB → 需要注意

- [ ] **计算逻辑不变**
  - tiling 只影响分块粒度
  - matmul_acc 的 K 维度遍历顺序不变
  - dtype cast 时机不变

### 4.2 并行化验证清单

将 `pl.range` 改为 `pl.parallel` 或添加 `chunk` 参数时：

- [ ] **迭代独立性**
  - 每个迭代只写入 output tensor 的不同区域（无 overlap）
  - 无跨迭代的读-写依赖

- [ ] **无跨迭代 reduction**
  - 循环体内不存在跨迭代的累加变量
  - 内层 K 循环的 `matmul_acc` 是 OK 的（在 ob 内部，不跨 ob）

- [ ] **`pl.assemble` 目标无冲突**
  - 每个迭代写入不同的 `[row, col]` 区域
  - 例：`pl.assemble(out, tile, [b0, ob * CHUNK])` — 不同 ob 写入不同列

**示例分析**（Scope 3 output projection）：

```python
for ob in pl.range(Q_OUT_BLOCKS):   # 可否改为 parallel?
    o0 = ob * Q_OUT_CHUNK           # 每个 ob 对应不同的列偏移
    # 内部读取 attn_out 的所有 K_CHUNK 块（只读，不写）    ✓
    # 写入 resid1_tile[0:BATCH_TILE, o0:o0+Q_OUT_CHUNK]    ✓ 不重叠
    # 内层 K 循环的 matmul_acc 只在 ob 内部累加              ✓ 不跨 ob
    # → 可以安全并行化
```

### 4.3 结构变换验证清单

内核融合（如 gate/up 合并）、stage 合并时：

- [ ] **数学运算序列不变**
  - 融合前后的 matmul、add、mul 操作序列相同
  - 操作的输入和输出 tensor 不变

- [ ] **精度路径不变**
  - 中间结果的 dtype cast 顺序不变（FP32 → BF16 的时机不变）
  - 没有引入额外的 BF16 round-trip

- [ ] **硬件资源足够**
  - Acc (L0C): 融合后所有累加器的总大小 ≤ 128 KB
  - Mat (L1): 融合后所有 L1 缓冲区的总大小 ≤ 512 KB
  - L0A: 融合后左矩阵 ≤ 64 KB
  - L0B: 融合后右矩阵 ≤ 64 KB（可交替加载）
  - Vec (UB): 融合后向量操作数 ≤ 192 KB

- [ ] **数据复用正确**
  - 共享的输入 tile 在所有消费者中是只读使用
  - 没有消费者修改共享数据

---

## 5. Phase 2 — 实施与验证

### 5.1 最小变更原则

每次只做 **一个** 优化变更。不要在同一个 commit 中同时：
- 修改 tiling 参数
- 添加并行化
- 融合内核

如果多个优化可以叠加，逐个实施，每个通过验证后再做下一个。

### 5.2 固定输入验证

使用 Phase 0 保存的 golden 基线数据验证优化后的程序。

**方法 A — 使用 `golden_data` 参数（推荐）**：

```python
# 在优化后的程序 main 中：
result = run(
    program=build_optimized_program(),
    tensor_specs=build_tensor_specs(),
    golden_fn=golden_fn,
    config=RunConfig(
        rtol=3e-3,              # 必须与原始版本相同
        atol=3e-3,
        compile=dict(dump_passes=True),
        runtime=dict(platform="a2a3sim", device_id=0),
    ),
    golden_data="/path/to/golden_baseline/scope3_20260420",
)
```

`golden_data` 指向 Phase 0 保存的目录，`run()` 会：
1. 从 `golden_data/in/` 加载固定输入（跳过随机生成）
2. 在设备上执行优化后的程序
3. 从 `golden_data/out/` 加载预存的 golden 输出
4. 对比设备输出和 golden 输出

**方法 B — 手动对比**：

```bash
# 运行优化后版本（使用默认随机输入）
python optimized_scope3.py -p a2a3sim

# 手动对比输出
python3 -c "
import torch
baseline = torch.load('golden_baseline/scope3_20260420/out/out.pt', weights_only=True)
optimized = torch.load('build_output/OptimizedScope3_*/data/out/out.pt', weights_only=True)
print(f'allclose(rtol=3e-3, atol=3e-3): {torch.allclose(baseline, optimized, rtol=3e-3, atol=3e-3)}')
diff = (baseline.float() - optimized.float()).abs()
print(f'max abs diff:  {diff.max().item():.6e}')
print(f'mean abs diff: {diff.mean().item():.6e}')
"
```

> **注意**：方法 B 中两次运行使用不同的随机输入，无法做 bit-exact 对比。
> 只能验证"在各自的 golden 下都通过容差"。方法 A 更严格。

### 5.3 多输入验证

固定输入验证通过后，使用随机输入验证多次，覆盖不同的数值范围：

```bash
# 运行 N 次，每次使用不同的随机输入
for i in $(seq 1 10); do
    echo "=== Run $i/10 ==="
    python optimized_scope3.py -p a2a3sim || { echo "FAIL at run $i"; exit 1; }
done
echo "All 10 runs passed"
```

**为什么需要多次**：
- BF16 精度问题可能只在特定数值范围触发（如非常大/小的值、接近零的除法）
- 随机输入覆盖了 init_value 函数的采样空间
- 10 次是最低要求，关键优化建议 20+ 次

### 5.4 边界条件测试

针对特定优化类型的额外测试：

| 优化类型 | 需要额外测试的边界 | 如何测试 |
|----------|-------------------|---------|
| Tiling 参数修改 | HIDDEN 不是 K_CHUNK 整数倍 | 修改模型参数（如 HIDDEN=8000） |
| Attention 并行化 | seq_len=1（单 token） | `--max-seq` 替换为固定 seq_lens=[1]*16 |
| Attention 并行化 | seq_len=max_seq（满 cache） | 使用 `--max-seq` 参数 |
| Batch 并行化 | batch=1 | 修改 BATCH 常量 |
| MLP 融合 | 极端权重值 | 修改 init_value 函数 |

**示例：测试极端权重值**

```python
# 在 build_tensor_specs() 中替换 init 函数
def init_extreme_w_gate():
    t = torch.randn(hidden_size, intermediate_size) * 100  # 放大 100x
    return t / hidden_size ** 0.5

# 或测试接近零的权重
def init_tiny_w_gate():
    t = torch.randn(hidden_size, intermediate_size) * 1e-6
    return t / hidden_size ** 0.5
```

---

## 6. Phase 3 — 精度影响分析

### 6.1 何时精度可能变化

即使数学等价，以下操作会导致浮点精度微小变化：

1. **累加顺序变化**：`(a+b)+c ≠ a+(b+c)` 在浮点数下成立。如果优化改变了
   matmul_acc 的 K 维度遍历顺序，结果可能有微小差异（通常在 1e-6 量级）。

2. **BF16 round-trip 时机变化**：FP32 → BF16 → FP32 的 cast 位置变化会改变
   中间精度。例如，在 Scope 1 中将 q_proj 从 FP32 改为 BF16 输出，
   会在 RoPE 计算前引入一次额外的 BF16 量化。

3. **并行 reduction 合并顺序**：将顺序 reduction 改为并行（如 tree reduction），
   合并顺序不同导致不同 rounding。

4. **超越函数的硬件实现差异**：`exp`、`sqrt`、`recip` 在真机上可能有 1-2 ULP
   与模拟器的差异。

### 6.2 精度对比工具

除了 `torch.allclose`（pass/fail 二值判定），建议使用以下精度报告函数：

```python
def precision_report(baseline: torch.Tensor, optimized: torch.Tensor, name: str = "output"):
    """对比两个 tensor 的精度差异，输出详细报告。"""
    diff = (baseline.float() - optimized.float()).abs()
    rel_diff = diff / (baseline.float().abs() + 1e-10)

    print(f"[{name}] shape={baseline.shape} dtype={baseline.dtype}")
    print(f"  max abs diff:  {diff.max().item():.6e}")
    print(f"  mean abs diff: {diff.mean().item():.6e}")
    print(f"  max rel diff:  {rel_diff.max().item():.6e}")
    print(f"  mean rel diff: {rel_diff.mean().item():.6e}")

    # 各误差区间的元素占比
    for threshold in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        pct = (diff > threshold).float().mean().item() * 100
        print(f"  elements > {threshold:.0e}: {pct:.2f}%")

    # NaN/Inf 检测
    nan_count = torch.isnan(optimized).sum().item()
    inf_count = torch.isinf(optimized).sum().item()
    if nan_count > 0 or inf_count > 0:
        print(f"  WARNING: NaN={nan_count}, Inf={inf_count}")
```

### 6.3 容差决策矩阵

| 最大绝对误差 | 最大相对误差 | 判定 | 行动 |
|-------------|-------------|------|------|
| < 1e-5 | < 1e-5 | **精确等价** | 无需额外关注 |
| < 1e-3 | < 1e-3 | **BF16 正常范围** | 可接受，记录精度报告 |
| < 3e-3 | < 3e-3 | **可接受** | 当前 Scope 3 / full decode 的容差水平 |
| 3e-3 ~ 1e-2 | 3e-3 ~ 1e-2 | **需审查** | 调查是否改变了精度路径 |
| > 1e-2 | > 1e-2 | **异常** | 可能存在正确性 bug |
| > 1e-1 | > 1e-1 | **明确 bug** | 必须修复 |
| NaN / Inf | — | **严重 bug** | 必须修复 |

### 6.4 精度变化的可接受条件

如果优化导致精度变化（在容差范围内），需要满足以下条件才能接受：

1. 变化的原因已被识别和记录（如"累加顺序变化"、"BF16 round-trip 时机变化"）
2. 10+ 次随机输入测试均在容差内通过
3. 精度报告显示 mean abs diff 没有数量级的增长
4. 不需要放宽 rtol/atol

---

## 7. Phase 4 — 跨平台验证

### 7.1 验证路径：模拟器 → 真机

```bash
# Step 1: 模拟器验证（编译 + 逻辑正确性）
python optimized_scope3.py -p a2a3sim

# Step 2: 真机验证（硬件正确性 + 性能数据）
python optimized_scope3.py -p a2a3 --runtime-profiling

# Step 3: 性能分析（仅真机有意义）
python tools/perf_analyzer.py build_output/OptimizedScope3_*/ -o optimized_perf.md
```

### 7.2 模拟器与真机的差异

| 维度 | 模拟器 (a2a3sim) | 真机 (a2a3) |
|------|------------------|-------------|
| **浮点精度** | CPU 模拟，可能有微小差异 | 真实硬件实现 |
| **超越函数** | CPU libm 实现 | NPU 专用实现（1-2 ULP 差异） |
| **内存对齐** | 可能更宽松 | 严格 512B 对齐 |
| **时间戳** | CPU 模拟时钟，绝对值无意义 | 真实硬件计数器 |
| **调度行为** | 准确的逻辑顺序 | 真实延迟和争用 |

### 7.3 跨平台容差处理

如果优化在 a2a3sim 通过但在 a2a3 真机失败：

1. **检查超越函数精度差异**：`exp`、`sqrt`、`recip` 在真机上可能有 1-2 ULP 差异
2. **检查内存对齐问题**：`pl.slice` 的起始地址是否满足 512B 对齐
3. **对比精度报告**：如果差异在 1e-3 量级，属于正常的硬件实现差异

**不应做的事**：为了让真机通过而放宽容差。如果真机失败但模拟器通过，
说明可能是对齐或精度路径问题，应排查根因。

---

## 8. Phase 5 — 回归保护

### 8.1 持久化 Golden 数据

优化验证通过后，保存新版本的 golden 数据：

```bash
OPTIMIZED_DIR="golden_cache/scope3_optimized_$(date +%Y%m%d)_v1"
mkdir -p "$OPTIMIZED_DIR"
cp -r build_output/OptimizedScope3_*/data/* "$OPTIMIZED_DIR/"
echo "优化版 golden 保存到: $OPTIMIZED_DIR"
```

### 8.2 全量回归测试

单个 scope 优化后，运行完整的 Qwen3 测试套件：

```bash
bash examples/models/qwen3/run_all_tests.sh -p a2a3sim
```

**检查项**（按重要性排序）：

1. [ ] 优化的 scope 本身通过
2. [ ] 完整 decode (`qwen3_32b_decode.py`) 通过
3. [ ] Tile 变体 (`*_tile.py`) 通过
4. [ ] Mixed 变体 (`*_mixed.py`) 通过
5. [ ] Tilelet 变体 (`*_tilelet.py`) 通过
6. [ ] Prefill 变体通过（如果优化也适用于 prefill）
7. [ ] 训练变体通过（如果优化也适用于训练前向）

### 8.3 性能非回归验证

确认优化确实带来了性能提升：

```bash
# 对比基线和优化后的性能报告
python tools/perf_analyzer.py build_output/Baseline_*/ -o baseline.md
python tools/perf_analyzer.py build_output/Optimized_*/ -o optimized.md
```

**关注指标**：

| 指标 | 期望变化 | 说明 |
|------|---------|------|
| Wall Time | ↓ 降低 | 端到端耗时减少 |
| Exec/Latency ratio | ↑ 提升 | 纯计算占比提高 |
| Scheduler idle% | ↓ 降低 | 调度器更忙碌（好） |
| Orch alloc% | — 不变或 ↓ | ring buffer 压力未增加 |
| Memory utilization | — 不变或 ↑ | 内存利用更充分 |
| 新增 WARN 诊断 | 无 | 不应引入新的性能问题 |

---

## 9. 实战案例：S3-1 Gate/Up 融合

以 S3-1（Gate/Up 投影融合）为例，演示完整的优化正确性保障流程。

### 9.1 Phase 0 — 建立基线

```bash
# 运行原始 scope3
python examples/models/qwen3/qwen3_32b_decode_scope3.py -p a2a3sim

# 保存 golden 基线
cp -r build_output/Qwen3Scope3_*/data golden_baseline/scope3/
```

### 9.2 Phase 1 — 数学等价性证明

**变更描述**：将 gate 和 up 投影从两个独立 `with pl.at` 块合并为一个。

**等价性论证**：
- 融合前：`for kb: gate_acc += post_chunk × wg`，然后 `for kb: up_acc += post_chunk × wu`
- 融合后：`for kb: gate_acc += post_chunk × wg; up_acc += post_chunk × wu`
- 两者的 matmul 和 matmul_acc 调用序列完全相同
- `post_chunk` 是只读数据，gate 和 up 不互相影响
- 没有引入额外的 dtype cast
- **数学严格等价** ✓

**硬件约束验证**：

| 资源 | 用量 | 上限 | 状态 |
|------|------|------|------|
| Acc | 8 KB (2 × 4 KB) | 128 KB | ✓ 6.3% |
| Mat | 36 KB | 512 KB | ✓ 7.0% |
| L0A | 4 KB | 64 KB | ✓ |
| L0B | 16 KB (交替) | 64 KB | ✓ |

### 9.3 Phase 2 — 实施与验证

```python
# 修改 qwen3_32b_decode_scope3.py 第 109–135 行
# 将两个 with pl.at 块合并为一个

# 固定输入验证：
result = run(
    program=build_qwen3_scope3_program(),
    tensor_specs=build_tensor_specs(),
    golden_fn=golden,
    config=RunConfig(rtol=3e-3, atol=3e-3, ...),
    golden_data="golden_baseline/scope3",  # 使用保存的基线
)
assert result.passed  # 必须通过

# 多输入验证（10 次随机输入）：
for i in range(10):
    result = run(...)
    assert result.passed, f"Failed at run {i+1}"
```

### 9.4 Phase 3 — 精度分析

```python
baseline_out = torch.load("golden_baseline/scope3/out/out.pt", weights_only=True)
optimized_out = torch.load("build_output/OptimizedScope3_*/data/out/out.pt", weights_only=True)
precision_report(baseline_out, optimized_out, name="scope3_out")
```

**预期结果**：
- max abs diff ≈ 0（数学严格等价，不应有差异）
- 如果有微小差异（< 1e-6），属于浮点执行顺序的正常波动

### 9.5 Phase 5 — 回归

```bash
# 全量测试
bash examples/models/qwen3/run_all_tests.sh -p a2a3sim

# 如果修改了 decode.py 中的 Scope 3，还需验证完整 decode 版本
python examples/models/qwen3/qwen3_32b_decode.py -p a2a3sim
```

---

## 10. 完整检查清单

每次优化迭代使用此清单逐项确认：

```
Phase 0 — 优化前准备
  □ 运行原始版本并保存 golden 基线数据（data/in + data/out）
  □ 记录基线性能指标（真机 profiling，可选）

Phase 1 — 数学等价性验证
  □ 证明优化的数学等价性（文档化）
  □ 检查硬件约束
      □ TILELET ≤ 2 KB
      □ TILE ≤ 16 KB
      □ Acc ≤ 128 KB
      □ Mat ≤ 512 KB
      □ L0A/L0B ≤ 64 KB
      □ Vec ≤ 192 KB
  □ 检查并行化的迭代独立性（如适用）
  □ 检查整除性约束（如适用）

Phase 2 — 实施与验证
  □ 仅做一个优化变更
  □ 固定输入验证通过（同一 rtol/atol）
  □ 随机输入 10+ 次验证通过
  □ 边界条件测试通过（如适用）

Phase 3 — 精度影响分析
  □ 生成精度对比报告（max/mean abs/rel diff）
  □ 确认误差在容差决策矩阵范围内
  □ 如精度有变化，记录原因
  □ 不需要放宽 rtol/atol

Phase 4 — 跨平台验证
  □ 模拟器 (a2a3sim) 验证通过
  □ 真机 (a2a3) 验证通过（如有访问权限）

Phase 5 — 回归保护
  □ 运行 run_all_tests.sh 全量 Qwen3 测试通过
  □ CI 测试通过（beginner + intermediate examples）
  □ 性能确认提升（非退化）
  □ 保存新 golden 数据作为回归基线
```
