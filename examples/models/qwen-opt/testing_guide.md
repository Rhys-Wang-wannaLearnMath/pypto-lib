# Qwen3 优化版对比测试指南

本文档说明如何对 `examples/models/qwen-opt/` 下的优化版 Scope 文件进行
**正确性验证** 和 **性能对比测试**，确保优化后的代码与原版行为一致。

---

## 1. 文件映射

| 优化版 (qwen-opt/) | 原版 (qwen3/) | 容差 | 默认平台 |
|---|---|---|---|
| `qwen3_32b_decode_scope1_opt.py` | `qwen3_32b_decode_scope1.py` | 1e-3 | `a2a3`（原版默认 `a5`） |
| `qwen3_32b_decode_scope1_tile_opt.py` | `qwen3_32b_decode_scope1_tile.py` | 1e-3 | `a2a3` |
| `qwen3_32b_decode_scope2_opt.py` | `qwen3_32b_decode_scope2.py` | 1e-3 | `a2a3` |
| `qwen3_32b_decode_scope3_opt.py` | `qwen3_32b_decode_scope3.py` | 3e-3 | `a2a3` |

> **注意**：原版 `qwen3_32b_decode_scope1.py` 默认平台为 `a5`，但对应优化版
> 默认为 `a2a3`。手动对比时请显式指定 `-p` 以确保一致。

---

## 2. 对比测试原理

对比测试的核心思路：**用完全相同的输入数据和 golden 输出，分别跑原版和优化版**，
两者都 PASS 即证明优化没有改变计算结果。

流程：

```
原版运行 → 生成 golden 数据（data/in/ + data/out/）
                ↓
        保存 golden 目录路径
                ↓
优化版运行 → --golden-data 指向同一份数据 → PASS/FAIL
```

`--golden-data` 参数会跳过随机输入生成，直接复用指定目录的输入张量和期望输出。

---

## 3. 对比测试步骤

> **所有命令从 pypto-lib 仓库根目录执行。**

### 3.1 Scope 3 对比（gate/up 融合优化）

```bash
# ── 第一步：运行原版，生成 golden 数据 ──
python examples/models/qwen3/qwen3_32b_decode_scope3.py -p a2a3sim

# 找到产物目录（输出日志中会打印，或查看最新的 build_output 目录）
# 例如：build_output/Qwen3Scope3_20260420_221806/data
GOLDEN=$(ls -td build_output/Qwen3Scope3_*/data | head -1)
echo "Golden data: $GOLDEN"

# ── 第二步：用相同数据运行优化版 ──
python examples/models/qwen-opt/qwen3_32b_decode_scope3_opt.py -p a2a3sim --golden-data $GOLDEN

# 两者都输出 PASS 即验证通过
```

### 3.2 Scope 1 对比

```bash
# 原版（注意显式指定 -p a2a3sim，因为原版默认平台是 a5）
python examples/models/qwen3/qwen3_32b_decode_scope1.py -p a2a3sim

GOLDEN=$(ls -td build_output/Qwen3Scope1_*/data | head -1)
echo "Golden data: $GOLDEN"

# 优化版
python examples/models/qwen-opt/qwen3_32b_decode_scope1_opt.py -p a2a3sim --golden-data $GOLDEN
```

### 3.3 Scope 1 Tile 对比

```bash
python examples/models/qwen3/qwen3_32b_decode_scope1_tile.py -p a2a3sim

GOLDEN=$(ls -td build_output/Qwen3Scope1_*/data | head -1)
echo "Golden data: $GOLDEN"

python examples/models/qwen-opt/qwen3_32b_decode_scope1_tile_opt.py -p a2a3sim --golden-data $GOLDEN
```

### 3.4 Scope 2 对比

```bash
python examples/models/qwen3/qwen3_32b_decode_scope2.py -p a2a3sim

GOLDEN=$(ls -td build_output/Qwen3Scope2_*/data | head -1)
echo "Golden data: $GOLDEN"

python examples/models/qwen-opt/qwen3_32b_decode_scope2_opt.py -p a2a3sim --golden-data $GOLDEN
```

---

## 4. 性能对比测试

在 **模拟器** 上收集编排器任务数和耗时（注意模拟器的 wall time 无真实性能意义，
仅关注任务数和编排器耗时的相对变化）：

```bash
# 原版 + profiling
python examples/models/qwen3/qwen3_32b_decode_scope3.py -p a2a3sim --runtime-profiling

# 优化版 + profiling（复用同一份 golden 数据）
GOLDEN=$(ls -td build_output/Qwen3Scope3_*/data | head -1)
python examples/models/qwen-opt/qwen3_32b_decode_scope3_opt.py -p a2a3sim --runtime-profiling --golden-data $GOLDEN
```

在 **真实 NPU 设备** 上收集真实性能数据：

```bash
# 原版
python examples/models/qwen3/qwen3_32b_decode_scope3.py -p a2a3 -d 0 --runtime-profiling

# 优化版
GOLDEN=$(ls -td build_output/Qwen3Scope3_*/data | head -1)
python examples/models/qwen-opt/qwen3_32b_decode_scope3_opt.py -p a2a3 -d 0 --runtime-profiling --golden-data $GOLDEN
```

性能对比关注指标：
- **编译产生的内核数**
- **运行时任务数**（任务数减少 = 调度开销降低）
- **编排器总耗时**
- **各 scope 执行时间**（真实设备上有意义）

---

## 5. 批量对比测试

一键运行所有 scope 的原版 → 优化版对比：

```bash
#!/usr/bin/env bash
# 从 pypto-lib 根目录运行
set -euo pipefail

PLATFORM="${1:-a2a3sim}"   # 默认模拟器，传 a2a3 则用真实设备
DEVICE="${2:-0}"

declare -A PAIRS=(
    ["examples/models/qwen3/qwen3_32b_decode_scope1.py"]="examples/models/qwen-opt/qwen3_32b_decode_scope1_opt.py"
    ["examples/models/qwen3/qwen3_32b_decode_scope1_tile.py"]="examples/models/qwen-opt/qwen3_32b_decode_scope1_tile_opt.py"
    ["examples/models/qwen3/qwen3_32b_decode_scope2.py"]="examples/models/qwen-opt/qwen3_32b_decode_scope2_opt.py"
    ["examples/models/qwen3/qwen3_32b_decode_scope3.py"]="examples/models/qwen-opt/qwen3_32b_decode_scope3_opt.py"
)

PASS_COUNT=0
FAIL_COUNT=0

for orig in "${!PAIRS[@]}"; do
    opt="${PAIRS[$orig]}"
    scope_name=$(basename "$orig" .py)
    echo ""
    echo "══════════════════════════════════════════════════"
    echo "  Testing: $scope_name"
    echo "══════════════════════════════════════════════════"

    # Run original
    echo "── Original: $orig ──"
    if ! python "$orig" -p "$PLATFORM" -d "$DEVICE"; then
        echo "FAIL: original $orig"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # Find golden data (most recent build_output matching the scope)
    # Extract build name prefix from the output dir pattern
    GOLDEN=$(ls -td build_output/*/data 2>/dev/null | head -1)
    if [[ -z "$GOLDEN" || ! -d "$GOLDEN" ]]; then
        echo "FAIL: could not find golden data for $scope_name"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    echo "Golden data: $GOLDEN"

    # Run optimized with same golden data
    echo "── Optimized: $opt ──"
    if python "$opt" -p "$PLATFORM" -d "$DEVICE" --golden-data "$GOLDEN"; then
        echo "PASS: $scope_name"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "FAIL: optimized $opt"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

echo ""
echo "══════════════════════════════════════════════════"
echo "  Results: $PASS_COUNT PASS, $FAIL_COUNT FAIL"
echo "══════════════════════════════════════════════════"
[[ $FAIL_COUNT -eq 0 ]] && exit 0 || exit 1
```

使用方法：

```bash
# 将上面的脚本保存后运行，或直接复制到终端

# 模拟器上测试
bash examples/models/qwen-opt/run_comparison.sh a2a3sim

# 真实设备上测试
bash examples/models/qwen-opt/run_comparison.sh a2a3 0
```

---

## 6. 理解测试结果

### PASS

```
两版均 PASS，容差内一致 → 优化正确，未改变计算语义
```

### FAIL

如果优化版 FAIL 而原版 PASS，说明优化引入了数值偏差。排查方向：
- **Mismatched elements 很少**（< 0.1%）→ 可能是浮点累加顺序变化导致的精度差异，
  考虑放宽容差确认
- **Mismatched elements 很多** → 优化逻辑有 bug，检查融合/并行化是否改变了数据流

### golden 数据目录结构

```
build_output/<ScopeName>_<timestamp>/data/
├── in/                 # 输入张量（.pt 文件）
│   ├── hidden_states.pt
│   ├── wq.pt
│   └── ...
└── out/                # Golden 期望输出
    └── out.pt
```

可用 `torch.load()` 加载做离线分析：

```python
import torch
inp = torch.load("data/in/hidden_states.pt")
golden = torch.load("data/out/out.pt")
```

---

## 7. 优化记录索引

| 编号 | 优化内容 | 记录文件 | 结果文件 |
|------|----------|----------|----------|
| S1-1 | K 维度分块累加优化 | [`opt-records/S1-1_k_chunk_reduction.md`](opt-records/S1-1_k_chunk_reduction.md) | — |
| S1-2 | QKV 并行投影 | [`opt-records/S1-2_qkv_parallel.md`](opt-records/S1-2_qkv_parallel.md) | — |
| S2-1/2/4 | 注意力融合 + 并行 | [`opt-records/S2-1_2_4_attention_fusion_parallel.md`](opt-records/S2-1_2_4_attention_fusion_parallel.md) | — |
| S3-1 | Gate/Up 投影融合 | [`opt-records/S3-1_gate_up_fusion.md`](opt-records/S3-1_gate_up_fusion.md) | [`opt-results/S3-1_gate_up_fusion_result.md`](opt-results/S3-1_gate_up_fusion_result.md) |
