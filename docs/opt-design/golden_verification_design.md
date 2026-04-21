# Golden 验证系统设计与实现

记录 pypto-lib 中 golden 验证系统的完整设计：golden 数据如何产生、
验证如何执行、以及如何用于优化前后的公平对比。

---

## 1. 整体架构

```
用户代码 (scope3.py)
    │
    │  提供三样东西：
    │  ├── build_program()    → PyPTO 程序（待编译到设备上执行）
    │  ├── build_tensor_specs() → 张量规格（名称、形状、dtype、初始化方式）
    │  └── golden_fn()        → PyTorch 参考实现（CPU 上执行，产生期望输出）
    │
    ▼
golden.run()                  ← 一站式入口，串联以下 5 个阶段
    │
    ├── ① compile        编译 PyPTO 程序
    ├── ② generate inputs 生成或加载输入张量
    ├── ③ runtime         在设备（真机/模拟器）上执行
    ├── ④ compute golden  用 PyTorch 计算期望输出（或从缓存加载）
    └── ⑤ validate        torch.allclose 逐输出对比
```

**核心源文件**：

| 文件 | 职责 |
|------|------|
| `golden/tensor_spec.py` | `TensorSpec` — 定义每个张量的规格和初始化方式 |
| `golden/runner.py` | `run()` — 编译、执行、验证的一站式入口 |
| `golden/validation.py` | `validate_golden()` — 基于 `torch.allclose` 的对比逻辑 |

---

## 2. Golden 数据的产生

### 2.1 输入张量的产生

由 `TensorSpec.create_tensor()` 生成（`golden/tensor_spec.py:51-71`）：

```python
@dataclass
class TensorSpec:
    name: str                    # 张量名称，对应函数参数名
    shape: list[int]             # 形状
    dtype: torch.dtype           # 数据类型
    init_value: ... = None       # 初始化策略
    is_output: bool = False      # 是否为输出张量
```

`init_value` 支持多种初始化方式：

| init_value 类型 | 行为 | 示例 |
|----------------|------|------|
| `None` | 全零 | `TensorSpec("out", [16,8192], torch.bfloat16, is_output=True)` |
| `int` / `float` | 常量填充 | `init_value=7` |
| `torch.Tensor` | 直接使用 | `init_value=my_tensor` |
| `torch.randn` 等工厂函数 | 调用 `fn(shape, dtype=dtype)` | `init_value=torch.randn` |
| 自定义 callable | 调用 `fn()` 无参 | `init_value=lambda: torch.rand(16,8192)-0.5` |

**Qwen3 Scope 3 的实际用法**（自定义 callable，产生特定分布的随机值）：

```python
def init_w_gate():
    return (torch.rand(hidden_size, intermediate_size) - 0.5) / hidden_size ** 0.5

TensorSpec("w_gate", [8192, 25600], torch.bfloat16, init_value=init_w_gate)
```

### 2.2 Golden 输出的产生

由用户提供的 `golden_fn` 在 **CPU 上用 PyTorch** 计算（`golden/runner.py:240-248`）：

```python
# runner.py 中的实际代码
scratch = {}
for spec in tensor_specs:
    if spec.is_output and spec.init_value is None:
        scratch[spec.name] = torch.zeros(spec.shape, dtype=spec.dtype)
    else:
        scratch[spec.name] = input_snapshot[spec.name].clone()  # ← 克隆！
golden_fn(scratch)
golden_outputs = {spec.name: scratch[spec.name] for spec in tensor_specs if spec.is_output}
```

**关键设计：输入克隆隔离**

`golden_fn` 收到的是输入的 **clone**（`input_snapshot[name].clone()`），
不是设备执行使用的同一份张量。这保证了：
- 设备执行修改张量内容不会影响 golden 计算
- golden 计算修改张量内容不会影响设备输入

### 2.3 Scope 3 的 golden_fn 示例

```python
def golden(tensors):
    """纯 PyTorch 参考实现，CPU 执行"""
    attn_out = tensors["attn_out"]
    wo = tensors["wo"]
    # ... 读取所有输入

    # 1. Output projection + residual
    o_proj = torch.matmul(attn_out.float(), wo.float())
    resid1 = o_proj + hidden_states.float()

    # 2. RMSNorm
    variance = resid1.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    normed_bf16 = (resid1 * inv_rms * post_rms_weight).bfloat16()

    # 3. SwiGLU MLP
    gate = torch.matmul(normed_bf16.float(), w_gate.float())
    up = torch.matmul(normed_bf16.float(), w_up.float())
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    down = torch.matmul(mlp_bf16.float(), w_down.float())

    # 4. Final residual → 写入输出
    tensors["out"][:] = (down + resid1).bfloat16()
```

---

## 3. 数据持久化

### 3.1 自动保存

每次 `run()` 执行（无 `golden_data` 参数时），自动保存（`runner.py:220,249`）：

```python
# 保存输入快照
_save_tensors(work_dir / "data" / "in", input_snapshot)

# 保存 golden 输出
_save_tensors(work_dir / "data" / "out", golden_outputs)
```

产生的目录结构：

```
build_output/Qwen3Scope3_20260419_204418/
└── data/
    ├── in/                          ← 输入快照（随机生成时的值）
    │   ├── attn_out.pt
    │   ├── hidden_states.pt
    │   ├── wo.pt
    │   ├── post_rms_weight.pt
    │   ├── w_gate.pt
    │   ├── w_up.pt
    │   └── w_down.pt
    └── out/                         ← golden_fn 的 PyTorch 输出
        └── out.pt
```

### 3.2 从缓存加载（golden_data 模式）

当传入 `golden_data` 路径时（`runner.py:197-212,236-239`）：

```python
# 输入：从缓存加载，不随机生成
tensors = _load_tensors(data_dir, "in", input_names)

# Golden：从缓存加载，不调用 golden_fn
golden_outputs = _load_tensors(data_dir, "out", output_names)
```

**两个"不触发"**：
- `TensorSpec.create_tensor()` 不被调用 → 不产生新随机数
- `golden_fn()` 不被调用 → 不重新计算 PyTorch 参考输出

---

## 4. 验证逻辑

### 4.1 validate_golden 实现

`golden/validation.py:15-46`：

```python
def validate_golden(outputs, golden, rtol=1e-5, atol=1e-5):
    for name, actual_tensor in outputs.items():
        actual = actual_tensor.cpu()
        expected = golden[name].cpu()

        if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
            # 报告：不匹配数量、前 20 个不匹配元素的位置和值
            raise AssertionError(...)
```

### 4.2 torch.allclose 的判定标准

```
|actual - expected| ≤ atol + rtol × |expected|
```

对每个元素逐一检查，全部通过才算 PASS。

### 4.3 各 Scope 的容差设置

| 程序 | rtol | atol | 原因 |
|------|------|------|------|
| scope1 | 1e-3 | 1e-3 | BF16 投影的基本精度 |
| scope2 | 1e-3 | 1e-3 | Attention 含 BF16 round-trip |
| scope3 | 3e-3 | 3e-3 | MLP 多级 matmul 累积误差更大 |
| 完整 decode | 3e-3 | 3e-3 | 三个 scope 级联的累积误差 |

---

## 5. 两种验证模式

### 模式 A：随机输入（默认）

```bash
python scope3.py -p a2a3sim
```

```
随机生成输入 → 设备执行 → 输出
                              ↕ allclose
克隆输入 → golden_fn → golden 输出
```

每次运行输入不同。验证的是"程序对任意输入都正确"。

### 模式 B：固定输入（--golden-data）

```bash
python scope3.py -p a2a3sim --golden-data build_output/Qwen3Scope3_20260419_204418/data
```

```
从文件加载固定输入 → 设备执行 → 输出
                                  ↕ allclose
从文件加载固定 golden ─────→ golden 输出
```

每次运行输入相同。验证的是"程序对这组特定输入的输出与 PyTorch 一致"。

### 优化前后公平对比：两者都用模式 B，指向同一目录

```
                    同一个 golden_data 目录
                   ┌────────┴────────┐
                   ▼                 ▼
            原版 PyPTO 程序     优化版 PyPTO 程序
                   │                 │
            设备执行输出_A       设备执行输出_B
                   │                 │
                   ▼                 ▼
            vs golden (同一份)  vs golden (同一份)
```

- 输入：**同一文件**（`in/*.pt`）
- Golden：**同一文件**（`out/*.pt`）
- 唯一变量：PyPTO 程序本身

---

## 6. 错误报告

验证失败时，`validate_golden` 输出：

```
Output 'out' does not match golden.
Mismatched elements: 42/131072
rtol=0.003, atol=0.003
First 20 mismatches:
    [1024] actual=0.015625, expected=0.012695
    [1025] actual=-0.003906, expected=-0.007812
    ...
```

包含：
- **不匹配元素数量** / 总元素数量
- 使用的 **rtol 和 atol**
- 前 **20 个不匹配元素**的索引、实际值、期望值

---

## 7. run() 的完整执行流程图

```
run(program, tensor_specs, golden_fn, golden_data, config)
 │
 ├─ golden_data 有值?
 │   ├─ 是 → 从 golden_data/in/ 加载输入（不随机生成）
 │   └─ 否 → 调用 TensorSpec.create_tensor() 生成输入
 │            └─ 自动保存到 build_output/<name>/data/in/
 │
 ├─ ir.compile(program) → 编译
 │
 ├─ execute_compiled(work_dir, tensors) → 设备执行（修改 tensors in-place）
 │
 ├─ golden_data 有值?
 │   ├─ 是 → 从 golden_data/out/ 加载 golden（不调用 golden_fn）
 │   └─ 否 → 克隆输入 → golden_fn(克隆) → 提取输出
 │            └─ 自动保存到 build_output/<name>/data/out/
 │
 └─ validate_golden(device_outputs, golden_outputs, rtol, atol)
     ├─ PASS → RunResult(passed=True)
     └─ FAIL → RunResult(passed=False, error="Output 'out' does not match...")
```
