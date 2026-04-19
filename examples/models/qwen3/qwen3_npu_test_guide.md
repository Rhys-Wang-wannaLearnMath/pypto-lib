# Qwen3 真实 NPU 设备测试准备指南

本文档总结在 **真实 Ascend NPU 设备** 上测试 pypto-lib Qwen3 示例所需的
环境准备、配置和运行方法。

---

## 1. 系统要求

| 项目 | 要求 |
|---|---|
| **操作系统** | Linux（Ascend NPU 服务器，通常 aarch64） |
| **Python** | 3.10+ |
| **C++ 编译器** | g++-15（或 g++）+ ninja-build |
| **NPU 硬件** | Ascend 910B 或 Ascend 950 |
| **CANN 工具包** | Ascend CANN 8.5.0+（`npu-smi info` 可用） |
| **架构** | aarch64（NPU 服务器）或 x86_64 |

---

## 2. 安装步骤

### 2.1 通用依赖

```bash
# 1. 克隆 pypto-lib
git clone https://github.com/hw-native-sys/pypto-lib.git
cd pypto-lib

# 2. 安装系统依赖
sudo apt-get update
sudo apt-get install -y ninja-build
sudo apt-get install -y g++-15 || sudo apt-get install -y g++

# 3. 安装 pypto 编译器 + simpler 运行时
git clone --recurse-submodules --depth=1 https://github.com/hw-native-sys/pypto.git /tmp/pypto
pip install /tmp/pypto
pip install /tmp/pypto/runtime

# 4. 安装 Python 依赖
pip install torch --index-url https://download.pytorch.org/whl/cpu   # golden 参考在 CPU 上计算
pip install nanobind
```

### 2.2 安装 ptoas（PTO 汇编器，v0.25）

根据 NPU 服务器架构选择对应版本：

```bash
# aarch64（真实 NPU 服务器，最常见）
curl -L https://github.com/hw-native-sys/PTOAS/releases/download/v0.25/ptoas-bin-aarch64.tar.gz \
  -o /tmp/ptoas.tar.gz

# x86_64（如服务器为 x86 架构）
curl -L https://github.com/hw-native-sys/PTOAS/releases/download/v0.25/ptoas-bin-x86_64.tar.gz \
  -o /tmp/ptoas.tar.gz

# 解压并设置权限
mkdir -p $HOME/ptoas-bin && tar -xzf /tmp/ptoas.tar.gz -C $HOME/ptoas-bin
chmod +x $HOME/ptoas-bin/ptoas $HOME/ptoas-bin/bin/ptoas
```

> **注意**：ptoas 架构必须与服务器架构一致，否则会运行失败。

### 2.3 安装 pto-isa（固定到 CI 使用的提交）

```bash
git clone https://github.com/hw-native-sys/pto-isa.git $HOME/pto-isa
cd $HOME/pto-isa && git checkout ed0b4643 && cd -
```

> CI 当前锁定 pto-isa 提交 `ed0b4643`，ptoas 版本 `v0.25`。
> 使用其他版本可能导致不兼容。

### 2.4 真实设备额外步骤

```bash
# 确认 NPU 可访问
npu-smi info

# 设置 CANN 环境（根据实际安装路径调整）
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0
export PATH=$ASCEND_HOME_PATH/bin:$PATH

# 如果需要 NPU 版 PyTorch（可选，golden 参考仍在 CPU 上计算）
pip install torch   # 安装带 NPU 支持的版本
```

---

## 3. 环境变量

每次打开新终端时需要设置（建议写入 `~/.bashrc`）：

```bash
export PTOAS_ROOT=$HOME/ptoas-bin
export PTO_ISA_ROOT=$HOME/pto-isa
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0
```

---

## 4. 验证安装

```bash
# 基础检查
python -c "import pypto; print('pypto OK')"
$PTOAS_ROOT/ptoas --version
ls $PTO_ISA_ROOT/README*   # 确认 pto-isa 已克隆

# NPU 专属检查
npu-smi info               # 确认 NPU 可见
```

---

## 5. 可用真实设备平台

| 平台参数 | 后端 | 所需硬件 |
|---|---|---|
| `a2a3` | Ascend910B 真实设备 | 910B NPU + CANN |
| `a5` | Ascend950 真实设备 | 950 NPU + CANN |

CLI 参数说明：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `-p` / `--platform` | 目标平台（`a2a3` 或 `a5`） | `a2a3`（大部分文件） |
| `-d` / `--device` | NPU 设备编号 | `0` |
| `--runtime-profiling` | 开启运行时性能分析 | `False` |
| `--max-seq` | 使用最大序列长度测试（仅 decode/scope2 支持） | `False` |

> **注意**：`qwen3_32b_decode_scope1.py` 的默认平台为 `a5`（而非 `a2a3`），
> 手动测试时请注意显式指定 `-p` 参数。

---

## 6. 运行测试

> **重要**：所有命令必须从 **pypto-lib 仓库根目录** 执行。

### 6.1 基本命令

```bash
# 查看可用 NPU
npu-smi info

# 在设备 0 上运行
python examples/models/qwen3/<文件名> -p a2a3 -d 0

# 带性能分析
python examples/models/qwen3/<文件名> -p a2a3 -d 0 --runtime-profiling

# Ascend950 设备
python examples/models/qwen3/<文件名> -p a5 -d 0
```

### 6.2 推荐测试顺序

在模拟器全部 PASS 后，在真实设备上按相同顺序运行：

```bash
# Decode scope 级别（范围小，编译快）
python examples/models/qwen3/qwen3_32b_decode_scope1.py -p a2a3 -d 0
python examples/models/qwen3/qwen3_32b_decode_scope2.py -p a2a3 -d 0
python examples/models/qwen3/qwen3_32b_decode_scope3.py -p a2a3 -d 0
python examples/models/qwen3/qwen3_32b_decode_scope1_tile.py -p a2a3 -d 0

# 完整 Decode 层
python examples/models/qwen3/qwen3_32b_decode.py -p a2a3 -d 0
python examples/models/qwen3/qwen3_32b_decode_tile.py -p a2a3 -d 0
python examples/models/qwen3/qwen3_32b_decode_mixed.py -p a2a3 -d 0

# Prefill
python examples/models/qwen3/qwen3_32b_prefill_scope1.py -p a2a3 -d 0
python examples/models/qwen3/qwen3_32b_prefill_scope3.py -p a2a3 -d 0
python examples/models/qwen3/qwen3_32b_prefill_tilelet.py -p a2a3 -d 0
python examples/models/qwen3/qwen3_32b_prefill.py -p a2a3 -d 0

# 训练
python examples/models/qwen3/qwen3_32b_training_forward_and_backward.py -p a2a3 -d 0
```

### 6.3 批量运行

```bash
PLATFORM=a2a3
DEVICE=0
for f in \
  examples/models/qwen3/qwen3_32b_decode_scope1.py \
  examples/models/qwen3/qwen3_32b_decode_scope1_tile.py \
  examples/models/qwen3/qwen3_32b_decode_scope2.py \
  examples/models/qwen3/qwen3_32b_decode_scope3.py \
  examples/models/qwen3/qwen3_32b_decode.py \
  examples/models/qwen3/qwen3_32b_decode_tile.py \
  examples/models/qwen3/qwen3_32b_decode_mixed.py \
  examples/models/qwen3/qwen3_32b_prefill_scope1.py \
  examples/models/qwen3/qwen3_32b_prefill_scope3.py \
  examples/models/qwen3/qwen3_32b_prefill_tilelet.py \
  examples/models/qwen3/qwen3_32b_prefill.py \
  examples/models/qwen3/qwen3_32b_training_forward_and_backward.py; do
  echo "========== $f =========="
  python "$f" -p $PLATFORM -d $DEVICE && echo "PASS" || echo "FAIL: $f"
done
```

---

## 7. 性能分析

使用 `--runtime-profiling` 采集 kernel 执行性能数据：

```bash
python examples/models/qwen3/qwen3_32b_decode.py -p a2a3 -d 0 --runtime-profiling
python examples/models/qwen3/qwen3_32b_decode_tile.py -p a2a3 -d 0 --runtime-profiling
python examples/models/qwen3/qwen3_32b_decode_mixed.py -p a2a3 -d 0 --runtime-profiling
```

性能分析数据包括：
- 各 scope（incore function）的执行时间
- 数据搬运（TLOAD/TSTORE）耗时
- 计算单元（CUBE/VEC）利用率

比较不同版本（3-scope / TILE / TILELET）的性能差异，评估 tiling 策略的效果。

---

## 8. 常见问题排查

| 问题 | 原因 | 解决方法 |
|---|---|---|
| `RuntimeError: device not found` | NPU 不可见 | 检查 `npu-smi info`、CANN 安装、设备号 |
| `ImportError: libascendcl.so` | CANN 路径未配置 | `export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0` |
| ptoas 架构不匹配 | x86_64 二进制在 aarch64 上运行 | 下载对应架构的 ptoas（见 2.2） |
| `ModuleNotFoundError: pypto` | pypto 未安装 | `pip install /tmp/pypto && pip install /tmp/pypto/runtime` |
| `ptoas: command not found` | 环境变量未设置 | `export PTOAS_ROOT=$HOME/ptoas-bin` |
| `FileNotFoundError: pto-isa` | pto-isa 未克隆或路径错误 | `export PTO_ISA_ROOT=$HOME/pto-isa` |
| `ModuleNotFoundError: golden` | 工作目录不对 | 确保从 pypto-lib 根目录运行 |
| FAIL 且不匹配数很少 | BF16 精度累积误差 | 属于已知现象，可尝试放宽容差确认 |
