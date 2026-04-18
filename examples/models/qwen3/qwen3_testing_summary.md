# Qwen3 测试总结

本文档总结 pypto-lib 中 Qwen3 "完整测试"的含义、涉及的步骤、当前状态，
以及它是否等同于完整的推理任务。

---

## "完整测试"指的是什么？

在 pypto-lib 中，**Qwen3 完整测试**是指能够：

1. **编译** — 将 PyPTO 语言描述的 Qwen3 Transformer 层编译为设备代码。
2. **执行** — 在目标平台（真实硬件或模拟器）上运行编译后的 kernel。
3. **验证** — 将设备输出与 PyTorch 实现的 golden 参考在指定容差
   （`rtol` / `atol`）内进行逐元素比对。

它**并非**端到端推理（例如 tokenizer 分词、多层前向传播、采样输出文本）。
每个测试验证的是**单个 Transformer 层**（或其中某个子 scope）的计算正确性：

- 通过 `TensorSpec` 生成随机输入张量（hidden states、权重、KV cache 等）
- 在设备上运行编译后的 kernel
- 在 PyTorch 中执行相同计算（`golden_fn`）
- 使用 `torch.allclose` 逐元素比对输出

---

## 测试流水线步骤

`examples/models/qwen3/` 下的所有示例脚本均遵循相同的流水线，
由 `golden/` 测试框架驱动：

```
构建程序 → 构建张量规格 → 编译 → 设备执行 → 计算 golden → 验证
```

### 步骤 1：构建程序 (`build_*_program()`)

用 PyPTO 语言构造 `@pl.program` 类，描述 kernel 的完整计算图。
对于完整的单层测试，包含：RMSNorm、Q/K/V 投影、RoPE、注意力、
输出投影、MLP 和残差连接。

### 步骤 2：构建张量规格 (`build_tensor_specs()`)

使用 `golden.TensorSpec` 定义每个输入/输出张量的 shape、dtype 和
初始化策略。权重通常使用 Xavier 缩放的随机值初始化。

### 步骤 3：编译 (`pypto.ir.compile`)

将 PyPTO 程序编译为目标设备代码。后端根据 `--platform` 参数选择：

| 平台 | 后端 | 说明 |
|---|---|---|
| `a2a3` | Ascend910B | 真实 NPU 设备 |
| `a2a3sim` | Ascend910B | 模拟器 |
| `a5` | Ascend950 | 真实 NPU 设备 |
| `a5sim` | Ascend950 | 模拟器 |

### 步骤 4：设备执行

在选定的平台上运行编译后的 kernel，输入张量保存至 `data/in/` 以便复现。

### 步骤 5：计算 Golden 参考 (`golden_fn`)

在纯 PyTorch 中执行相同计算，模拟 kernel 的精度行为（如 BF16 类型转换、
分块累加）。Golden 输出保存至 `data/out/`。

### 步骤 6：验证 (`validate_golden`)

使用 `torch.allclose` 在可配置容差下比对设备输出和 golden 输出。
不匹配的元素会报告索引和具体数值。

---

## Qwen3 文件清单与测试范围

### Decode（每个 batch 生成一个 token）

| 文件 | 范围 | HIDDEN | 可测试 |
|---|---|---|---|
| `qwen3_32b_decode.py` | 完整层（3-scope） | 8192 | 是 |
| `qwen3_32b_decode_mixed.py` | 完整层（TILELET 感知） | 8192 | 是 |
| `qwen3_32b_decode_tile.py` | 完整层（TILE 版本） | 8192 | 是 |
| `qwen3_32b_decode_scope1.py` | Scope 1：RMSNorm + QKV 投影 | 8192 | 是 |
| `qwen3_32b_decode_scope1_tile.py` | Scope 1：tiled 变体 | 8192 | 是 |
| `qwen3_32b_decode_scope2.py` | Scope 2：注意力 | 8192 | 是 |
| `qwen3_32b_decode_scope3.py` | Scope 3：输出投影 + MLP | 8192 | 是 |

### Prefill（每个 batch 处理多个 token）

| 文件 | 范围 | HIDDEN | 可测试 |
|---|---|---|---|
| `qwen3_32b_prefill.py` | 完整层 | **5120** | 是（HIDDEN 值有误） |
| `qwen3_32b_prefill_tilelet.py` | 完整层（TILELET 感知） | **5120** | 是（HIDDEN 值有误） |
| `qwen3_32b_prefill_scope1.py` | Scope 1：RMSNorm + QKV 投影 | 8192 | 是 |
| `qwen3_32b_prefill_scope3.py` | Scope 3：输出投影 + MLP | 8192 | 是 |

### 训练

| 文件 | 范围 | HIDDEN | 可测试 |
|---|---|---|---|
| `qwen3_32b_training_forward_and_backward.py` | 前向 + 反向 + 优化器 | **5120** | 是（HIDDEN 值有误） |

### 其他

| 文件 | 说明 |
|---|---|
| `qwen3-32b.py` | 早期单体版本（HIDDEN=5120） |
| `qwen3_tilelet.md` | TILELET/TILE 尺寸参考文档（HIDDEN=5120，已过时） |

---

## 如何运行测试

所有测试脚本通过 `python <script> [options]` 执行：

```bash
# 在真实 NPU 上运行（Ascend910B）
python examples/models/qwen3/qwen3_32b_decode.py -p a2a3 -d 0

# 在模拟器上运行
python examples/models/qwen3/qwen3_32b_decode.py -p a2a3sim

# 启用性能分析
python examples/models/qwen3/qwen3_32b_decode.py -p a2a3 --runtime-profiling
```

`PASS` 表示设备 kernel 的输出与 PyTorch golden 参考在容差范围内匹配。
`FAIL` 会以退出码 1 终止并打印不匹配的元素。

---

## 这是完整的推理吗？

**不是。** 当前测试验证的是单个 Transformer 层 kernel 的计算正确性，
而非端到端推理流水线。完整的推理系统还需要：

- **Tokenizer** — 将文本转换为 token ID
- **Embedding 层** — 将 token ID 映射为 hidden states
- **多层编排** — 串联 64 个 Transformer 层（Qwen3-32B）
- **LM Head** — 将最终 hidden states 投影到词表 logits
- **采样策略** — greedy、top-k、top-p 等
- **KV Cache 管理** — 自回归步骤间的真实序列跟踪
- **真实模型权重** — 加载预训练权重而非随机初始化

pypto-lib 目前测试的是**单层计算 kernel**，这是推理系统中性能最关键的
构建单元。端到端推理不在 pypto-lib 的范围内。

---

## 已知问题（截至 PR #67 及相关修复）

### 已解决

- **HIDDEN 维度** — Decode 文件已从硬编码 5120 修正为
  `NUM_HEADS * HEAD_DIM = 8192`（PR #67、PR #99）
- **attn_out 零初始化** — 使用 `pl.full` + `pl.assemble` 实现（PR #67）
- **RMSNorm** — 修正为使用 `rsqrt` 并保持投影在 FP32 精度（PR #69、PR #77）

### 待解决

1. **Prefill 和训练文件 HIDDEN=5120** — `qwen3_32b_prefill.py`、
   `qwen3_32b_prefill_tilelet.py`、`qwen3_32b_training_forward_and_backward.py`
   和 `qwen3_tilelet.md` 仍使用 `HIDDEN=5120`，与 Qwen3-32B 模型规格
   （`64 * 128 = 8192`）不一致。

2. **Batch 维度初始化问题** — `qwen3_32b_decode_mixed.py` 中仅对
   `q_proj` / `attn_out` 的第一个 `BATCH_TILE` 行进行了零初始化
   （偏移 `[0, q0]`）。当 `BATCH_CFG > BATCH_TILE` 时，其余行未被置零。
   （Gemini 在 PR #67 review 中指出，尚未修复。）

3. **CI 未覆盖 models 目录** — CI 仅运行 `examples/beginner/` 和
   `examples/intermediate/`。所有 `examples/models/qwen3/` 下的测试
   需要手动执行。
