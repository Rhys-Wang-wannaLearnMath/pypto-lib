# Qwen3 完整测试流程详解

以 `qwen3_32b_decode.py`（Qwen3-32B 单层 decode）为例，逐步追踪一轮完整测试
从启动到输出结果的全过程，对应到具体代码和数据流。

---

## 总览：一轮测试做了什么

```
用户执行命令
  │
  ▼
┌──────────────────────────────────────────────────────┐
│ 1. 解析命令行参数（平台、设备、是否 profiling）        │
│ 2. 构建 PyPTO 程序（@pl.program 描述计算图）          │
│ 3. 构建张量规格（16 个 TensorSpec：shape/dtype/初始化）│
│ 4. 调用 golden.run()：                               │
│    4a. 编译 → 生成设备代码 + output_dir               │
│    4b. 创建张量 → 保存输入快照到 data/in/             │
│    4c. 设备执行 → kernel 在 NPU/模拟器上运行          │
│    4d. 计算 golden → PyTorch 参考实现                 │
│    4e. 保存 golden 输出到 data/out/                   │
│    4f. 验证 → torch.allclose 逐元素比对               │
│ 5. 输出 PASS / FAIL                                  │
└──────────────────────────────────────────────────────┘
```

---

## 第 1 步：启动测试

用户在命令行执行：

```bash
python examples/models/qwen3/qwen3_32b_decode.py -p a2a3 -d 0
```

入口代码位于 `qwen3_32b_decode.py` 的 `if __name__ == "__main__"` 块
（第 692–725 行），解析三个参数：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `-p` / `--platform` | 目标平台 | `a2a3` |
| `-d` / `--device` | NPU 设备编号 | `0` |
| `--runtime-profiling` | 是否开启性能分析 | `False` |

然后调用：

```python
result = run(
    program=build_qwen3_decode_program(),   # 步骤 2
    tensor_specs=build_tensor_specs(),       # 步骤 3
    golden_fn=golden_qwen3_decode,           # 步骤 4d 使用
    config=RunConfig(rtol=3e-3, atol=3e-3, runtime=dict(...)),
)
```

---

## 第 2 步：构建 PyPTO 程序

`build_qwen3_decode_program()` （第 61–441 行）构造一个 `@pl.program` 类
`Qwen3Decode`，其中包含一个 `@pl.function` 方法 `qwen3_decode`。

### 函数签名：读取哪些张量

`qwen3_decode` 方法声明了 **16 个参数**，对应一个完整 Transformer decode 层
所需的全部数据：

| 参数名 | Shape | DType | 角色 |
|---|---|---|---|
| `hidden_states` | [16, 8192] | BF16 | 输入 hidden states |
| `input_rms_weight` | [1, 8192] | FP32 | 输入 RMSNorm 权重 |
| `wq` | [8192, 8192] | BF16 | Q 投影权重 |
| `wk` | [8192, 1024] | BF16 | K 投影权重 |
| `wv` | [8192, 1024] | BF16 | V 投影权重 |
| `seq_lens` | [16] | INT32 | 每个 batch 的上下文长度 |
| `rope_cos` | [4096, 128] | FP32 | RoPE 余弦表 |
| `rope_sin` | [4096, 128] | FP32 | RoPE 正弦表 |
| `k_cache` | [8388608, 128] | BF16 | K 缓存 (BATCH×KV_HEADS×MAX_SEQ) |
| `v_cache` | [8388608, 128] | BF16 | V 缓存 |
| `wo` | [8192, 8192] | BF16 | 输出投影权重 |
| `post_rms_weight` | [1, 8192] | FP32 | 后 RMSNorm 权重 |
| `w_gate` | [8192, 25600] | BF16 | MLP gate 权重 |
| `w_up` | [8192, 25600] | BF16 | MLP up 权重 |
| `w_down` | [25600, 8192] | BF16 | MLP down 权重 |
| `out` | [16, 8192] | BF16 | **输出** |

### 计算图：Kernel 做了什么

函数体描述了三个 scope 的计算逻辑：

**Scope 1 — 输入 RMSNorm + Q/K/V 投影：**
- 遍历 batch 维度（步长 `BATCH_TILE=16`）
- 对 `hidden_states` 做 RMSNorm：分块求平方和 → 方差 → `1/sqrt(var+eps)` → 归一化
- 归一化结果与 `wq`、`wk`、`wv` 做矩阵乘法，得到 `q_proj`、`k_proj`、`v_proj`

**Scope 2 — RoPE + KV Cache 更新 + 注意力：**
- 对每个 batch 和每个 KV head：
  - K 向量做 RoPE 旋转 → 写入 `k_cache`
  - V 向量写入 `v_cache`
  - Q 向量做 RoPE 旋转
  - Q × K^T → softmax（online softmax，分 `SEQ_TILE=64` 块处理） → × V
  - 累积注意力输出到 `attn_out`

**Scope 3 — 输出投影 + 残差 + MLP：**
- `attn_out × wo` → 输出投影
- 加残差 `hidden_states`
- 后 RMSNorm
- MLP：gate/up 投影 → SiLU 激活 → down 投影
- 加残差 → 最终输出 `out`

> **注意：这一步不执行任何计算。** 它只是用 PyPTO DSL 描述计算图，
> 后续由编译器将其转化为设备代码。

---

## 第 3 步：构建张量规格

`build_tensor_specs()` （第 444–540 行）返回一个 **16 个 `TensorSpec`** 的列表，
与程序函数签名一一对应。

每个 `TensorSpec` 包含：
- `name` — 张量名称（必须匹配函数参数名）
- `shape` — 形状
- `dtype` — PyTorch 数据类型
- `init_value` — 初始化函数（`None` 则零初始化）
- `is_output` — 是否为输出张量（需要验证）

### 初始化策略示例

```python
# hidden_states: [-0.5, 0.5) 均匀随机 → 转为 BF16
def init_hidden_states():
    return torch.rand(batch, hidden_size) - 0.5

# wq: Xavier 缩放的均匀随机
def init_wq():
    return torch.rand(hidden_size, hidden_size) / hidden_size ** 0.5

# seq_lens: [1, MAX_SEQ] 范围内的随机整数
def init_seq_lens():
    return torch.randint(1, max_seq + 1, (batch,), dtype=torch.int32)

# out: 标记 is_output=True，无 init_value → 零初始化
TensorSpec("out", [batch, hidden], torch.bfloat16, is_output=True)
```

---

## 第 4 步：执行 `golden.run()`

核心流程在 `golden/runner.py` 的 `run()` 函数（第 93–172 行）中。
以下逐一说明每个子步骤。

### 4a. 编译

```python
compiled = ir.compile(program, **compile_kwargs)
```

- 读取 `program`（步骤 2 生成的 `Qwen3Decode` 类）
- 根据 `platform` 自动选择后端（`a2a3` → `Ascend910B`，`a5` → `Ascend950`）
- 调用 `pypto.ir.compile` 将 PyPTO DSL → PTO IR → 设备汇编代码
- 生成结果保存在 `compiled.output_dir` 目录中（包含汇编文件、编译产物等）

### 4b. 创建张量 + 保存输入快照

```python
tensors = {spec.name: spec.create_tensor() for spec in tensor_specs}
```

遍历 16 个 `TensorSpec`，每个调用 `create_tensor()` 生成 PyTorch 张量：
- 有 `init_value` 回调的：调用回调函数生成随机数据，转为目标 dtype
- `init_value=None` 的（如 `out`）：创建全零张量

然后保存**输入快照**：

```python
input_snapshot = {spec.name: tensors[spec.name].clone()
                  for spec in tensor_specs
                  if not spec.is_output or spec.init_value is not None}
_save_tensors(compiled.output_dir / "data" / "in", input_snapshot)
```

**输出目录结构：**

```
<compiled.output_dir>/
└── data/
    └── in/
        ├── hidden_states.pt     # [16, 8192] BF16
        ├── input_rms_weight.pt  # [1, 8192] FP32
        ├── wq.pt               # [8192, 8192] BF16
        ├── wk.pt               # [8192, 1024] BF16
        ├── wv.pt               # [8192, 1024] BF16
        ├── seq_lens.pt         # [16] INT32
        ├── rope_cos.pt         # [4096, 128] FP32
        ├── rope_sin.pt         # [4096, 128] FP32
        ├── k_cache.pt          # [8388608, 128] BF16
        ├── v_cache.pt          # [8388608, 128] BF16
        ├── wo.pt               # [8192, 8192] BF16
        ├── post_rms_weight.pt  # [1, 8192] FP32
        ├── w_gate.pt           # [8192, 25600] BF16
        ├── w_up.pt             # [8192, 25600] BF16
        └── w_down.pt           # [25600, 8192] BF16
```

每个 `.pt` 文件是 `torch.save` 保存的 PyTorch 张量，可通过 `torch.load()` 加载复现。

### 4c. 设备执行

```python
ordered = [tensors[spec.name] for spec in tensor_specs]
compiled(*ordered, config=PyPTORunConfig(**config.runtime))
```

- 将 16 个张量按 `TensorSpec` 声明顺序排列为列表
- 调用 `compiled(...)` 在目标设备上执行 kernel
- 执行完成后，`tensors["out"]` 中的数据已被 kernel 写入（设备结果）
- `k_cache` 和 `v_cache` 也会被 kernel 更新（scope 2 写入新的 KV 缓存条目）

### 4d. 计算 Golden 参考

```python
scratch = {}
for spec in tensor_specs:
    if spec.is_output and spec.init_value is None:
        scratch[spec.name] = torch.zeros(spec.shape, dtype=spec.dtype)
    else:
        scratch[spec.name] = input_snapshot[spec.name].clone()
golden_fn(scratch)
```

- 从 `input_snapshot`（步骤 4b 保存的原始输入）**重新克隆**一份干净的张量
- 输出张量（`out`）初始化为全零
- 调用 `golden_qwen3_decode(scratch)` —— 这是纯 PyTorch 实现的参考函数

`golden_qwen3_decode`（第 543–689 行）用 PyTorch 执行完全相同的计算：
1. **Scope 1 golden**：RMSNorm（分块求平方和 → `1/sqrt(var+eps)` → 归一化 → BF16），
   然后 Q/K/V 投影（`normed @ wq/wk/wv`）
2. **Scope 2 golden**：K RoPE → 写 cache → Q RoPE → 分块 attention
   （online softmax，与 kernel 相同的 BF16 cast 行为）
3. **Scope 3 golden**：`attn_out @ wo` + 残差 → RMSNorm → SiLU-gated MLP → 残差

golden 函数直接修改 `scratch["out"]`，使其包含期望的输出。

### 4e. 保存 Golden 输出

```python
golden_outputs = {spec.name: scratch[spec.name]
                  for spec in tensor_specs if spec.is_output}
_save_tensors(compiled.output_dir / "data" / "out", golden_outputs)
```

**输出目录结构：**

```
<compiled.output_dir>/
└── data/
    ├── in/
    │   └── (15 个输入 .pt 文件)
    └── out/
        └── out.pt              # [16, 8192] BF16 — golden 期望输出
```

### 4f. 验证

```python
validate_golden(device_outputs, golden_outputs, rtol=config.rtol, atol=config.atol)
```

对每个输出张量（这里只有 `out`）：
- 调用 `torch.allclose(actual, expected, rtol=3e-3, atol=3e-3)`
- **通过**：返回 `RunResult(passed=True)`
- **失败**：抛出 `AssertionError`，包含：
  - 不匹配元素数 / 总元素数
  - 前 20 个不匹配元素的索引、实际值、期望值

---

## 第 5 步：输出最终结果

```python
if not result.passed:
    if result.error:
        print(f"Result: {result.error}")
    raise SystemExit(1)
```

| 结果 | 含义 | 退出码 |
|---|---|---|
| `PASS` | 设备输出与 golden 在 `rtol=3e-3, atol=3e-3` 内匹配 | 0 |
| `FAIL` | 存在超出容差的元素，打印详细错误信息 | 1 |

---

## 完整的目录输出结构

一轮测试结束后，`compiled.output_dir` 下的文件结构如下：

```
<output_dir>/                   # pypto.ir.compile 生成的目录
├── (编译产物：汇编代码、IR dump 等，具体由 pypto 编译器决定)
└── data/
    ├── in/                     # 输入快照（15 个 .pt 文件）
    │   ├── hidden_states.pt
    │   ├── input_rms_weight.pt
    │   ├── wq.pt
    │   ├── wk.pt
    │   ├── wv.pt
    │   ├── seq_lens.pt
    │   ├── rope_cos.pt
    │   ├── rope_sin.pt
    │   ├── k_cache.pt
    │   ├── v_cache.pt
    │   ├── wo.pt
    │   ├── post_rms_weight.pt
    │   ├── w_gate.pt
    │   ├── w_up.pt
    │   └── w_down.pt
    └── out/                    # Golden 期望输出（1 个 .pt 文件）
        └── out.pt
```

`data/in/` 和 `data/out/` 保证了测试的**完全可复现性**：可以重新加载这些
张量，跳过随机初始化，直接对比设备结果与 golden。

---

## Qwen3 各文件测试差异

不同文件测试的范围和容差不同：

| 文件 | 测试范围 | 容差 (rtol/atol) |
|---|---|---|
| `qwen3_32b_decode.py` | 完整单层 decode（3-scope） | 3e-3 |
| `qwen3_32b_decode_mixed.py` | 完整单层 decode（TILELET 感知） | 2e-2 |
| `qwen3_32b_decode_tile.py` | 完整单层 decode（TILE 版本） | 3e-3 |
| `qwen3_32b_decode_scope1.py` | 仅 Scope 1（RMSNorm + QKV 投影） | 1e-3 |
| `qwen3_32b_decode_scope2.py` | 仅 Scope 2（注意力） | 1e-3 |
| `qwen3_32b_decode_scope3.py` | 仅 Scope 3（输出投影 + MLP） | 3e-3 |
| `qwen3_32b_prefill_tilelet.py` | 完整单层 prefill | 2e-2 |
| `qwen3_32b_prefill_scope1.py` | Prefill Scope 1 | 1e-3 |
| `qwen3_32b_prefill_scope3.py` | Prefill Scope 3 | 3e-3 |

容差较大的文件（`2e-2`）通常是因为更多的 BF16 精度截断导致累积误差更大。

---

## 小结

一轮 Qwen3 完整测试的本质是：**用随机数据驱动单层 Transformer kernel，
在设备上执行后与 PyTorch 参考实现逐元素比对**。它验证的是 kernel 计算逻辑
的正确性，而非端到端推理能力。测试全程自动化，所有输入和期望输出都以 `.pt`
文件持久化，保证可复现和可调试。
