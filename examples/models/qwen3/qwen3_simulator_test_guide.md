# Qwen3 模拟器测试指南

本文档指导如何在 **simulator（模拟器）** 上对现有 Qwen3 示例进行完整测试。
模拟器不需要真实 NPU 硬件，可在普通 x86_64 Linux 机器上运行。

---

## 1. 环境准备

### 1.1 系统要求

- **操作系统**：Linux（Ubuntu 20.04+ 推荐）
- **Python**：3.10+
- **C++ 编译器**：**g++-15**（必须，不能用 g++-11 等低版本替代）+ ninja-build
- **`/tmp` 目录权限**：必须为 `1777`（`drwxrwxrwt`），否则 apt/pip 等工具无法创建临时文件
- **不需要** NPU 硬件或 Ascend CANN 工具包

### 1.2 安装步骤（使用 uv）

项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 虚拟环境和依赖。

```bash
# 1. 克隆 pypto-lib（如尚未克隆）
git clone https://github.com/hw-native-sys/pypto-lib.git
cd pypto-lib

# 2. 确保 /tmp 权限正确（权限不对会导致 apt/pip 报 "Couldn't create temporary file" 错误）
sudo chmod 1777 /tmp

# 3. 安装系统依赖
sudo apt-get update
sudo apt-get install -y ninja-build

# 4. 安装 g++-15（必须是 15，pto-isa 头文件使用了 C++20 <format> 等特性，g++-11/12 不支持）
#    Ubuntu 22.04 默认源没有 g++-15，需要添加 PPA：
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update
sudo apt-get install -y g++-15
#    如果安装后 g++-15 命令不可用，可能需要手动创建符号链接：
#    sudo ln -sf /usr/bin/x86_64-linux-gnu-g++-15 /usr/bin/g++-15
g++-15 --version   # 确认输出 15.x

# 5. 创建 uv 虚拟环境并激活
uv venv .venv --python 3.10
source .venv/bin/activate

# 6. 安装 pypto 编译器 + simpler 运行时
git clone --recurse-submodules --depth=1 https://github.com/hw-native-sys/pypto.git /tmp/pypto
#    ⚠️ pypto 包含 ~490 个 C++ 源文件，构建需要 scikit-build-core + CMake + nanobind。
#    在 4 核机器上首次编译可能需要 20-40 分钟，uv pip install 不显示编译进度，
#    看起来会像"卡住"在 "Building pypto ... Preparing packages..."，实际是在编译 C++。
#    可通过 `ps aux | grep cc1plus` 确认编译是否在进行。
#    如果想看到编译进度，可以先手动构建：
#      cd /tmp/pypto && cmake -B build -G Ninja && cmake --build build -j$(nproc)
#    然后再 uv pip install。
uv pip install /tmp/pypto
uv pip install /tmp/pypto/runtime

# 7. 安装 ptoas（汇编器，x86_64 版本，v0.25）
curl -L https://github.com/hw-native-sys/PTOAS/releases/download/v0.25/ptoas-bin-x86_64.tar.gz \
  -o /tmp/ptoas.tar.gz
mkdir -p $HOME/ptoas-bin && tar -xzf /tmp/ptoas.tar.gz -C $HOME/ptoas-bin
chmod +x $HOME/ptoas-bin/ptoas $HOME/ptoas-bin/bin/ptoas

# 8. 克隆 pto-isa（固定到 CI 使用的提交）
git clone https://github.com/hw-native-sys/pto-isa.git $HOME/pto-isa
cd $HOME/pto-isa && git checkout d96c8784 && cd -

# 9. 安装 Python 依赖
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install nanobind
```

### 1.3 设置环境变量

将以下内容写入 **`~/.profile`**：

```bash
cat >> ~/.profile << 'EOF'

# pypto-lib environment
export PTOAS_ROOT=$HOME/ptoas-bin
export PTO_ISA_ROOT=$HOME/pto-isa
EOF
```

写入后执行 `source ~/.profile` 使当前终端立即生效。

> **为什么写 `~/.profile` 而不是 `~/.bashrc`？** Ubuntu 默认的 `~/.bashrc` 开头有
> `[ -z "$PS1" ] && return`，非交互式 shell（如脚本、CI、IDE 内置终端的某些调用方式）
> 会在此处直接退出，导致写在 `return` 之后的 `export` 不生效。
> 而 `~/.profile` 在登录时执行，且 Ubuntu 的 `~/.profile` 会自动 `source ~/.bashrc`，
> 因此只需写入 `~/.profile` 即可覆盖所有场景。

同时记得激活虚拟环境：

```bash
source /path/to/pypto-lib/.venv/bin/activate
```

### 1.4 验证安装

```bash
python -c "import pypto; print('pypto OK')"
$PTOAS_ROOT/ptoas --version
```

---

## 2. 可用的模拟器平台

| 平台参数 | 模拟目标 | 说明 |
|---|---|---|
| `a2a3sim` | Ascend910B | 模拟 910B 后端，使用 ptoas + pto-isa 在 CPU 上执行 |
| `a5sim` | Ascend950 | 模拟 950 后端 |

通过 `-p` 参数指定，例如 `-p a2a3sim`。

---

## 3. Qwen3 测试文件清单

### 3.1 Decode（推荐首先测试）

| 文件 | 范围 | HIDDEN | 容差 |
|---|---|---|---|
| `qwen3_32b_decode.py` | 完整单层（3-scope） | 8192 | 3e-3 |
| `qwen3_32b_decode_tile.py` | 完整单层（TILE 版本） | 8192 | 3e-3 |
| `qwen3_32b_decode_mixed.py` | 完整单层（TILELET 感知） | 8192 | 2e-2 |
| `qwen3_32b_decode_scope1.py` | Scope 1：RMSNorm + QKV 投影 | 8192 | 1e-3 |
| `qwen3_32b_decode_scope1_tile.py` | Scope 1：tiled 变体 | 8192 | 1e-3 |
| `qwen3_32b_decode_scope2.py` | Scope 2：注意力 | 8192 | 1e-3 |
| `qwen3_32b_decode_scope3.py` | Scope 3：输出投影 + MLP | 8192 | 3e-3 |

### 3.2 Prefill

| 文件 | 范围 | HIDDEN | 容差 |
|---|---|---|---|
| `qwen3_32b_prefill.py` | 完整单层 | 5120 | — |
| `qwen3_32b_prefill_tilelet.py` | 完整单层（TILELET 感知） | 5120 | 2e-2 |
| `qwen3_32b_prefill_scope1.py` | Scope 1：RMSNorm + QKV 投影 | 8192 | 1e-3 |
| `qwen3_32b_prefill_scope3.py` | Scope 3：输出投影 + MLP | 8192 | 3e-3 |

> **注意**：`qwen3_32b_prefill.py` 和 `qwen3_32b_prefill_tilelet.py` 的
> `HIDDEN=5120` 与 Qwen3-32B 实际规格（8192）不一致，属于已知遗留问题。

### 3.3 训练

| 文件 | 范围 | HIDDEN | 容差 |
|---|---|---|---|
| `qwen3_32b_training_forward_and_backward.py` | 前向 + 反向 + 优化器 | 5120 | — |

---

## 4. 运行测试

### 4.1 基本命令格式

```bash
python examples/models/qwen3/<文件名> -p <平台>
```

模拟器上不需要 `-d`（设备编号）参数，默认为 0 即可。

### 4.2 逐个运行（推荐开始时使用）

从 scope 级别的小测试开始，逐步扩大范围：

```bash
# ── 第一步：测试各个子 scope（快速，便于定位问题）──

# Decode Scope 1：RMSNorm + QKV 投影
python examples/models/qwen3/qwen3_32b_decode_scope1.py -p a2a3sim

# Decode Scope 2：注意力
python examples/models/qwen3/qwen3_32b_decode_scope2.py -p a2a3sim

# Decode Scope 3：输出投影 + MLP
python examples/models/qwen3/qwen3_32b_decode_scope3.py -p a2a3sim

# ── 第二步：测试完整单层 ──

# 完整 Decode 层（3-scope）
python examples/models/qwen3/qwen3_32b_decode.py -p a2a3sim

# 完整 Decode 层（TILE 版本）
python examples/models/qwen3/qwen3_32b_decode_tile.py -p a2a3sim

# 完整 Decode 层（TILELET 感知）
python examples/models/qwen3/qwen3_32b_decode_mixed.py -p a2a3sim

# ── 第三步：测试 Prefill ──

# Prefill Scope 1
python examples/models/qwen3/qwen3_32b_prefill_scope1.py -p a2a3sim

# Prefill Scope 3
python examples/models/qwen3/qwen3_32b_prefill_scope3.py -p a2a3sim

# 完整 Prefill 层
python examples/models/qwen3/qwen3_32b_prefill_tilelet.py -p a2a3sim
```

### 4.3 批量运行所有 Decode 测试

```bash
for f in \
  examples/models/qwen3/qwen3_32b_decode_scope1.py \
  examples/models/qwen3/qwen3_32b_decode_scope1_tile.py \
  examples/models/qwen3/qwen3_32b_decode_scope2.py \
  examples/models/qwen3/qwen3_32b_decode_scope3.py \
  examples/models/qwen3/qwen3_32b_decode.py \
  examples/models/qwen3/qwen3_32b_decode_tile.py \
  examples/models/qwen3/qwen3_32b_decode_mixed.py; do
  echo "========== $f =========="
  python "$f" -p a2a3sim && echo "PASS" || echo "FAIL: $f"
done
```

### 4.4 使用 a5sim 平台

将 `-p a2a3sim` 替换为 `-p a5sim` 即可测试 Ascend950 后端：

```bash
python examples/models/qwen3/qwen3_32b_decode.py -p a5sim
```

---

## 5. 理解测试输出

### 5.1 PASS 的情况

测试静默完成，退出码为 0。表示设备 kernel 输出与 PyTorch golden 参考在
容差范围内完全匹配。

### 5.2 FAIL 的情况

退出码为 1，并打印类似以下的错误信息：

```
Result: Output 'out' does not match golden.
Mismatched elements: 1234/131072
rtol=0.003, atol=0.003
First 20 mismatches:
    [42] actual=0.123, expected=0.456
    [99] actual=-0.789, expected=-0.321
    ...
```

关键信息：
- **Mismatched elements** — 不匹配元素数 / 总元素数
- **rtol/atol** — 使用的相对/绝对容差
- **First N mismatches** — 前 N 个不匹配元素的索引和实际值 vs 期望值

### 5.3 测试产物目录

每次运行后，编译和测试产物保存在 `compiled.output_dir` 中：

```
<output_dir>/
├── (编译产物：汇编代码、IR 等)
└── data/
    ├── in/                 # 所有输入张量快照 (.pt 文件)
    │   ├── hidden_states.pt
    │   ├── wq.pt
    │   ├── ...
    │   └── w_down.pt
    └── out/                # Golden 期望输出
        └── out.pt
```

这些 `.pt` 文件可用 `torch.load()` 加载，用于离线调试：

```python
import torch
actual = torch.load("data/in/hidden_states.pt")
golden = torch.load("data/out/out.pt")
```

---

## 6. 常见问题排查

| 问题 | 原因 | 解决方法 |
|---|---|---|
| `ModuleNotFoundError: pypto` | pypto 未安装 | 重新执行 `pip install /tmp/pypto` |
| `ptoas: command not found` 或 `ptoas binary not found` | `PTOAS_ROOT` 环境变量未设置 | `export PTOAS_ROOT=$HOME/ptoas-bin` |
| `FileNotFoundError: pto-isa` | pto-isa 未克隆或路径错误 | `export PTO_ISA_ROOT=$HOME/pto-isa` |
| `FileNotFoundError: 'g++-15'` | g++-15 未安装 | 通过 PPA 安装，见 1.2 节步骤 4 |
| `fatal error: format: No such file or directory` (`#include <format>`) | 使用了 g++-11 等低版本编译器（不支持 C++20 `<format>`） | 必须安装真正的 g++-15，不能用 g++-11 符号链接替代 |
| `g++-15` 安装后 `command not found` | Ubuntu 上 g++-15 包只安装了 `x86_64-linux-gnu-g++-15`，缺少 `g++-15` 链接 | `sudo ln -sf /usr/bin/x86_64-linux-gnu-g++-15 /usr/bin/g++-15` |
| `Couldn't create temporary file /tmp/apt.conf.xxx` | `/tmp` 目录权限不正确（不是 `1777`） | `sudo chmod 1777 /tmp` |
| `uv pip install /tmp/pypto` 卡在 `Building pypto ... Preparing packages` | pypto 含 ~490 个 C++ 文件，编译耗时长（4 核约 20-40 分钟），不是真的卡住 | 用 `ps aux \| grep cc1plus` 确认编译在进行；或先 `cmake --build` 手动编译以查看进度 |
| `ModuleNotFoundError: golden` | 工作目录不对 | 确保从 pypto-lib 根目录运行，或检查 `sys.path` |
| `apt-get update` 下载极慢 | 网络问题，PPA 源在海外 | 配置 apt 代理：写入 `/etc/apt/apt.conf.d/99proxy`，内容为 `Acquire::http::Proxy "http://127.0.0.1:7890";` 和 `Acquire::https::Proxy "http://127.0.0.1:7890";`（端口根据实际代理调整） |
| 编译超时或很慢 | 大模型编译本身耗时较长 | scope 级测试编译更快，先测试 scope 文件 |
| FAIL 且不匹配数很少 | BF16 精度累积误差 | 属于已知现象，可尝试放宽容差确认 |

---

## 7. 模拟器 vs 真实设备的区别

| 对比项 | 模拟器 (`a2a3sim` / `a5sim`) | 真实设备 (`a2a3` / `a5`) |
|---|---|---|
| **硬件要求** | 无需 NPU，普通 x86 机器 | 需要 Ascend NPU + CANN |
| **验证目标** | 计算逻辑正确性（功能仿真） | 计算正确性 + 真实硬件行为 |
| **执行速度** | 较慢（CPU 模拟） | 快（NPU 原生执行） |
| **`--runtime-profiling`** | 无意义（无硬件计数器） | 采集真实性能数据 |
| **测试结果** | PASS/FAIL 与真实设备一致 | PASS/FAIL 与模拟器一致 |
| **适用场景** | 开发调试、CI 自动化 | 性能调优、最终验证 |

---

## 8. 推荐测试顺序

对于首次运行 Qwen3 测试，建议按以下顺序：

```
1. qwen3_32b_decode_scope1.py      ← 最小范围，编译最快
2. qwen3_32b_decode_scope2.py      ← 验证注意力逻辑
3. qwen3_32b_decode_scope3.py      ← 验证 MLP 逻辑
4. qwen3_32b_decode.py             ← 完整单层 decode
5. qwen3_32b_prefill_scope1.py     ← Prefill 子模块
6. qwen3_32b_prefill_scope3.py     ← Prefill 子模块
7. qwen3_32b_decode_tile.py        ← TILE 版本完整层
8. qwen3_32b_decode_mixed.py       ← TILELET 感知版本
9. qwen3_32b_prefill_tilelet.py    ← 完整 Prefill 层
```

如果某个 scope 测试失败，无需继续测试包含该 scope 的完整层测试，
应先定位和修复该 scope 的问题。
