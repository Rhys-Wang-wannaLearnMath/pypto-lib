# Qwen3 真实 NPU 设备测试准备指南

本文档总结在 **真实 Ascend NPU 设备** 上测试 pypto-lib Qwen3 示例所需的
环境准备、配置和运行方法。

---

## 1. 系统要求

| 项目 | 要求 |
|---|---|
| **操作系统** | Linux（Ascend NPU 服务器，通常 aarch64） |
| **Python** | 3.10+ |
| **C++ 编译器** | **g++-15**（必须，不能用 g++-11 等低版本替代）+ ninja-build |
| **NPU 硬件** | Ascend 910B 或 Ascend 950 |
| **CANN 工具包** | Ascend CANN 8.5.0+（`npu-smi info` 可用） |
| **架构** | aarch64（NPU 服务器）或 x86_64 |
| **`/tmp` 目录权限** | 必须为 `1777`（`drwxrwxrwt`），否则 apt/pip 等工具无法创建临时文件 |

---

## 2. 安装步骤

### 2.1 安装步骤（使用 uv）

项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 虚拟环境和依赖。

```bash
# 1. 克隆 pypto-lib（如尚未克隆）
git clone https://github.com/hw-native-sys/pypto-lib.git
cd pypto-lib

# 2. 安装 uv（Python 包管理器，用于快速创建虚拟环境和安装依赖）
curl -LsSf https://astral.sh/uv/install.sh | sh
# 安装后需要重新加载 shell 配置或重新打开终端
source ~/.bashrc  # 或 source ~/.zshrc

# 3. 确保 /tmp 权限正确（权限不对会导致 apt/pip 报 "Couldn't create temporary file" 错误）
sudo chmod 1777 /tmp

# 4. 安装系统依赖
sudo apt-get update
sudo apt-get install -y ninja-build

# 5. 安装 g++-15（必须是 15，pto-isa 头文件使用了 C++20 <format> 等特性，g++-11/12 不支持）
#    Ubuntu 22.04 默认源没有 g++-15，需要添加 PPA：
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update
sudo apt-get install -y g++-15
#    如果安装后 g++-15 命令不可用，可能需要手动创建符号链接：
#    sudo ln -sf /usr/bin/x86_64-linux-gnu-g++-15 /usr/bin/g++-15       # x86_64
#    sudo ln -sf /usr/bin/aarch64-linux-gnu-g++-15 /usr/bin/g++-15      # aarch64
g++-15 --version   # 确认输出 15.x

# 6. 创建 uv 虚拟环境并激活
uv venv .venv --python 3.10
source .venv/bin/activate

# 7. 安装 pypto 编译器 + simpler 运行时
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

# 8. 安装 Python 依赖
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # golden 参考在 CPU 上计算
uv pip install nanobind
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

#### 2.4.1 确认 NPU 可访问并确定设备编号

```bash
npu-smi info
```

从输出中找到你的 NPU 编号。示例输出：

```
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 2     910B4               | OK            | 88.9        39                0    / 0             |
| 0                         | 0000:C2:00.0  | 0           0    / 0          2858 / 32768         |
+===========================+===============+====================================================+
```

上面第一列 `NPU` 显示为 **2**，表示该卡的设备编号为 `2`（而非 `0`）。
后续运行测试时需要用 `-d 2` 指定，例如：

```bash
python examples/models/qwen3/qwen3_32b_decode_scope1.py -p a2a3 -d 2
```

> **注意**：不同服务器的 NPU 编号不同，务必根据 `npu-smi info` 的实际输出确定。

#### 2.4.2 查找 CANN 安装路径

CANN 工具包的安装路径因版本和安装方式而异，常见位置如下：

```bash
# 方法 1：列出 /usr/local/Ascend/ 下的目录，找到实际版本
ls /usr/local/Ascend/

# 常见目录结构：
#   /usr/local/Ascend/ascend-toolkit/latest/   ← 最常见，推荐使用 latest 软链接
#   /usr/local/Ascend/cann-8.5.0/              ← 指定版本号
#   /usr/local/Ascend/cann-25.5.1/             ← 较新版本

# 方法 2：查看 npu-smi 报告的版本号，结合目录推断
npu-smi info | head -3
# 例如输出 "npu-smi 25.5.1  Version: 25.5.1"，则查找对应目录

# 方法 3：如果安装时使用了 ascend-toolkit，通常自带 set_env.sh
find /usr/local/Ascend -name "set_env.sh" 2>/dev/null
# 找到后可以直接 source 它，例如：
#   source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

确定路径后设置环境变量：

```bash
# 示例：如果实际路径为 /usr/local/Ascend/ascend-toolkit/latest
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PATH=$ASCEND_HOME_PATH/bin:$PATH

# 或者如果有 set_env.sh，直接 source 即可（它会设置所有 CANN 相关变量）：
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

#### 2.4.3 NPU 版 PyTorch（可选）

Golden 参考在 CPU 上计算，通常不需要 NPU 版 PyTorch。
如有需要：

```bash
uv pip install torch   # 安装带 NPU 支持的版本
```

---

## 3. 环境变量（持久化）

每次打开新终端都需要重新 `export` 很不方便。推荐使用仓库自带的
**一键配置脚本** 将环境变量写入 `~/.bashrc` 使其自动生效。

#### 3.1 使用一键脚本（推荐）

```bash
# 从 pypto-lib 仓库根目录运行
bash setup_npu_env.sh
```

脚本会自动完成：
1. **检测 CANN 安装路径**（按优先级搜索 `ascend-toolkit/latest` → `cann-*` → `set_env.sh`）
2. **检查** `$HOME/ptoas-bin` 和 `$HOME/pto-isa` 是否存在且有效
3. **检测 NPU 设备编号**（通过 `npu-smi info`），提示 `-d` 参数值
4. **插入环境变量到 `~/.bashrc` 最前面**（在 `[ -z "$PS1" ] && return` 之前，
   确保交互式和非交互式 shell 都能生效）
5. **验证**写入结果

可选参数：

| 参数 | 说明 |
|---|---|
| `--cann /path/to/cann` | 手动指定 CANN 路径（跳过自动检测） |
| `--dry-run` | 仅预览，不写入 `~/.bashrc` |
| `--uninstall` | 从 `~/.bashrc` 中移除脚本写入的环境变量块 |

脚本执行完成后，应用到当前终端：

```bash
source ~/.bashrc
```

> **脚本是幂等的**：重复运行会先移除旧的环境变量块，再插入新的，不会产生重复。
> 每次写入前会自动备份 `~/.bashrc` 为 `~/.bashrc.bak.<timestamp>`。

#### 3.2 手动配置（备选）

如果不想使用脚本，也可以手动操作。

> **关键**：Ubuntu 默认的 `~/.bashrc` 开头通常有一行 `[ -z "$PS1" ] && return`，
> 非交互式 shell 会在此处直接退出，导致写在 `return` **之后** 的 `export` 不生效。
> 因此必须将 `export` 写在这行 **之前**。

```bash
# 根据 2.4.2 节确定 CANN 路径，以下为示例，请替换为实际路径

cat > /tmp/_pypto_env.sh << 'ENVBLOCK'
# ── pypto-lib environment (inserted by setup_npu_env.sh) ──
export PTOAS_ROOT=$HOME/ptoas-bin
export PTO_ISA_ROOT=$HOME/pto-isa
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest   # ← 替换为实际路径
export PATH=$ASCEND_HOME_PATH/bin:$PTOAS_ROOT:$PATH
# ── end pypto-lib environment ──
ENVBLOCK

# 插入到 ~/.bashrc 最前面
cat /tmp/_pypto_env.sh ~/.bashrc > /tmp/_bashrc_new && mv /tmp/_bashrc_new ~/.bashrc
rm /tmp/_pypto_env.sh
```

写入后验证：

```bash
source ~/.bashrc
echo "PTOAS_ROOT=$PTOAS_ROOT"
echo "PTO_ISA_ROOT=$PTO_ISA_ROOT"
echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
```

#### 3.3 激活虚拟环境

环境变量配置好后，记得激活虚拟环境：

```bash
source /path/to/pypto-lib/.venv/bin/activate
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

## 6. Qwen3 测试文件清单

### 6.1 Decode（推荐首先测试）

| 文件 | 范围 | HIDDEN | 容差 |
|---|---|---|---|
| `qwen3_32b_decode.py` | 完整单层（3-scope） | 8192 | 3e-3 |
| `qwen3_32b_decode_tile.py` | 完整单层（TILE 版本） | 8192 | 3e-3 |
| `qwen3_32b_decode_mixed.py` | 完整单层（TILELET 感知） | 8192 | 2e-2 |
| `qwen3_32b_decode_scope1.py` | Scope 1：RMSNorm + QKV 投影 | 8192 | 1e-3 |
| `qwen3_32b_decode_scope1_tile.py` | Scope 1：tiled 变体 | 8192 | 1e-3 |
| `qwen3_32b_decode_scope2.py` | Scope 2：注意力 | 8192 | 1e-3 |
| `qwen3_32b_decode_scope3.py` | Scope 3：输出投影 + MLP | 8192 | 3e-3 |

### 6.2 Prefill

| 文件 | 范围 | HIDDEN | 容差 |
|---|---|---|---|
| `qwen3_32b_prefill.py` | 完整单层 | 5120 | — |
| `qwen3_32b_prefill_tilelet.py` | 完整单层（TILELET 感知） | 5120 | 2e-2 |
| `qwen3_32b_prefill_scope1.py` | Scope 1：RMSNorm + QKV 投影 | 8192 | 1e-3 |
| `qwen3_32b_prefill_scope3.py` | Scope 3：输出投影 + MLP | 8192 | 3e-3 |

> **注意**：`qwen3_32b_prefill.py` 和 `qwen3_32b_prefill_tilelet.py` 的
> `HIDDEN=5120` 与 Qwen3-32B 实际规格（8192）不一致，属于已知遗留问题。

### 6.3 训练

| 文件 | 范围 | HIDDEN | 容差 |
|---|---|---|---|
| `qwen3_32b_training_forward_and_backward.py` | 前向 + 反向 + 优化器 | 5120 | — |

### 6.4 设备负担排序（由轻到重）

以下按对设备（NPU）的 **计算量和内存开销** 从低到高排序。
核心区分因素：Decode 每 batch 仅处理 1 个 token（共 BATCH=16 tokens），
而 Prefill 处理最多 BATCH×MAX_SEQ = 16×4096 = 65536 tokens，
且 Prefill 注意力复杂度为 O(seq²)，远重于 Decode 的 O(seq)。

| 排名 | 文件 | 负担 | 关键开销 |
|:----:|------|:----:|----------|
| 1 | `qwen3_32b_decode_scope1.py` | ★☆☆☆☆ | 16 tokens × 3 matmul（QKV 投影）+ RMSNorm |
| 2 | `qwen3_32b_decode_scope1_tile.py` | ★☆☆☆☆ | 同 scope1，tile DSL 写法不同，计算量相同 |
| 3 | `qwen3_32b_decode_scope3.py` | ★★☆☆☆ | 16 tokens × 4 matmul（O 投影 + gate/up/down MLP）+ SiLU + 2× RMSNorm |
| 4 | `qwen3_32b_decode_scope2.py` | ★★☆☆☆ | 16 tokens × 注意力 O(seq)，需遍历 KV cache（最多 4096 步） |
| 5 | `qwen3-32b.py` | ★★★☆☆ | 完整 decode 单层（scope1+2+3），16 tokens，HIDDEN=8192 |
| 6 | `qwen3_32b_decode.py` | ★★★☆☆ | 同上，3-scope 版本 |
| 7 | `qwen3_32b_decode_tile.py` | ★★★☆☆ | 同上，tile DSL 版本，计算量相同 |
| 8 | `qwen3_32b_decode_mixed.py` | ★★★☆☆ | 同上，TILELET 感知版本，计算量相同 |
| 9 | `qwen3_32b_prefill_scope1.py` | ★★★★☆ | 最多 65536 tokens × 3 matmul + RMSNorm，计算量约为 decode scope1 的 **4096 倍** |
| 10 | `qwen3_32b_prefill_scope3.py` | ★★★★☆ | 最多 65536 tokens × 4 matmul + MLP，计算量约为 decode scope3 的 4096 倍 |
| 11 | `qwen3_32b_prefill.py` | ★★★★★ | 完整 prefill 单层，含 O(seq²) 注意力，HIDDEN=5120 |
| 12 | `qwen3_32b_prefill_tilelet.py` | ★★★★★ | 同上，TILELET 感知版本，计算量相同 |
| 13 | `qwen3_32b_training_forward_and_backward.py` | ★★★★★ | 前向 + 完整反向 + 优化器 ≈ 3× 前向计算量，内存峰值最高 |

> **建议**：首次在真实设备上测试时，按此顺序从上到下执行。
> scope 级测试失败时无需继续测试包含该 scope 的完整层，应先定位和修复问题。

---

## 7. 运行测试

> **重要**：所有命令必须从 **pypto-lib 仓库根目录** 执行。

### 7.1 基本命令

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

### 7.2 推荐测试顺序

在模拟器全部 PASS 后，在真实设备上按相同顺序运行：

```bash
# ── 第一步：测试各个子 scope（快速，便于定位问题）──

# Decode Scope 1：RMSNorm + QKV 投影
python examples/models/qwen3/qwen3_32b_decode_scope1.py -p a2a3 -d 0

# Decode Scope 2：注意力
python examples/models/qwen3/qwen3_32b_decode_scope2.py -p a2a3 -d 0

# Decode Scope 3：输出投影 + MLP
python examples/models/qwen3/qwen3_32b_decode_scope3.py -p a2a3 -d 0

# Decode Scope 1 Tile 变体
python examples/models/qwen3/qwen3_32b_decode_scope1_tile.py -p a2a3 -d 0

# ── 第二步：测试完整单层 ──

# 完整 Decode 层（3-scope）
python examples/models/qwen3/qwen3_32b_decode.py -p a2a3 -d 0

# 完整 Decode 层（TILE 版本）
python examples/models/qwen3/qwen3_32b_decode_tile.py -p a2a3 -d 0

# 完整 Decode 层（TILELET 感知）
python examples/models/qwen3/qwen3_32b_decode_mixed.py -p a2a3 -d 0

# ── 第三步：测试 Prefill ──

# Prefill Scope 1
python examples/models/qwen3/qwen3_32b_prefill_scope1.py -p a2a3 -d 0

# Prefill Scope 3
python examples/models/qwen3/qwen3_32b_prefill_scope3.py -p a2a3 -d 0

# 完整 Prefill 层（TILELET 感知）
python examples/models/qwen3/qwen3_32b_prefill_tilelet.py -p a2a3 -d 0

# 完整 Prefill 层
python examples/models/qwen3/qwen3_32b_prefill.py -p a2a3 -d 0

# ── 第四步：测试训练 ──

python examples/models/qwen3/qwen3_32b_training_forward_and_backward.py -p a2a3 -d 0
```

### 7.3 批量运行

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

## 8. 理解测试输出

### 8.1 PASS 的情况

测试静默完成，退出码为 0。表示设备 kernel 输出与 PyTorch golden 参考在
容差范围内完全匹配。

### 8.2 FAIL 的情况

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

### 8.3 测试产物目录

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

## 9. 性能分析

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

## 10. 常见问题排查

| 问题 | 原因 | 解决方法 |
|---|---|---|
| `RuntimeError: device not found` | NPU 不可见 | 检查 `npu-smi info`、CANN 安装、设备号 |
| `ImportError: libascendcl.so` | CANN 路径未配置 | `export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0` |
| ptoas 架构不匹配 | x86_64 二进制在 aarch64 上运行 | 下载对应架构的 ptoas（见 2.2） |
| `ModuleNotFoundError: pypto` | pypto 未安装 | 重新执行 `uv pip install /tmp/pypto && uv pip install /tmp/pypto/runtime` |
| `ptoas: command not found` | 环境变量未设置 | `export PTOAS_ROOT=$HOME/ptoas-bin` |
| `FileNotFoundError: pto-isa` | pto-isa 未克隆或路径错误 | `export PTO_ISA_ROOT=$HOME/pto-isa` |
| `FileNotFoundError: 'g++-15'` | g++-15 未安装 | 通过 PPA 安装，见 2.1 节步骤 5 |
| `fatal error: format: No such file or directory` (`#include <format>`) | 使用了 g++-11 等低版本编译器（不支持 C++20 `<format>`） | 必须安装真正的 g++-15，不能用 g++-11 符号链接替代 |
| `g++-15` 安装后 `command not found` | 包只安装了带架构前缀的二进制，缺少 `g++-15` 链接 | `sudo ln -sf /usr/bin/aarch64-linux-gnu-g++-15 /usr/bin/g++-15`（x86 则替换为 `x86_64-linux-gnu-g++-15`） |
| `Couldn't create temporary file /tmp/apt.conf.xxx` | `/tmp` 目录权限不正确（不是 `1777`） | `sudo chmod 1777 /tmp` |
| `uv pip install /tmp/pypto` 卡在 `Building pypto ... Preparing packages` | pypto 含 ~490 个 C++ 文件，编译耗时长（4 核约 20-40 分钟），不是真的卡住 | 用 `ps aux \| grep cc1plus` 确认编译在进行；或先 `cmake --build` 手动编译以查看进度 |
| `ModuleNotFoundError: golden` | 工作目录不对 | 确保从 pypto-lib 根目录运行 |
| `apt-get update` 下载极慢 | 网络问题，PPA 源在海外 | 配置 apt 代理：写入 `/etc/apt/apt.conf.d/99proxy`，内容为 `Acquire::http::Proxy "http://127.0.0.1:7890";` 和 `Acquire::https::Proxy "http://127.0.0.1:7890";`（端口根据实际代理调整） |
| FAIL 且不匹配数很少 | BF16 精度累积误差 | 属于已知现象，可尝试放宽容差确认 |
