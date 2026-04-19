# 泳道图（Swimlane Diagram）使用指南

本文档介绍 pypto-lib 项目中泳道图的生成、查看和使用方法。

---

## 1. 什么是泳道图

在本项目中，**泳道图（swimlane diagram）** 是一种运行时性能剖析可视化工具，
用于展示 NPU 内核在不同硬件单元（AIC cube、AIV vector、AICPU、调度器等）上的
**时间线执行情况**。每个硬件单元对应一条"泳道"，事件按时间轴从左到右排列，
可以直观地看到各单元之间的并行度、调度延迟和瓶颈所在。

泳道图使用 **Chrome Trace Event Format**（JSON 格式），可通过 Google
[Perfetto UI](https://ui.perfetto.dev/) 在浏览器中交互查看。

### 1.1 泳道图中的数据来源

底层由 C++ `PerformanceCollector` 类负责采集。每个 AICore / AICPU 线程在执行
过程中写入 `PerfRecord`，包含以下关键字段：

| 字段 | 含义 |
|------|------|
| `start_time` | 任务开始时间戳（`get_sys_cnt`） |
| `end_time` | 任务结束时间戳 |
| `duration` | 执行时长（end - start） |
| `dispatch_time` | AICPU 将任务派发给 AICore 的时间戳 |
| `finish_time` | AICPU 观察到任务完成的时间戳 |
| `task_id` | 任务标识 |
| `core_id` | 执行该任务的核心编号 |
| `fanout_tasks` | 后继任务列表 |

此外，AICPU 编排器还记录各阶段的累计耗时摘要（`AicpuOrchSummary`）：
`sync_cycle`、`alloc_cycle`、`args_cycle`、`lookup_cycle`、`heap_cycle`、
`insert_cycle`、`fanin_cycle`、`scope_end_cycle` 等。

---

## 2. 生成泳道图

### 2.1 两种平台均可生成

| 平台 | 能否生成泳道数据 | 产物 | 备注 |
|------|-----------------|------|------|
| **真实设备**（`a2a3` / `a5`） | 能 | `perf_swimlane_*.json` + `merged_swimlane_*.json` | 时间戳来自真实硬件计数器，绝对耗时有实际意义 |
| **模拟器**（`a2a3sim` / `a5sim`） | 能 | 仅 `perf_swimlane_*.json` | 时间戳来自 CPU 模拟的 `get_sys_cnt`，绝对耗时无实际意义，但执行顺序和调度流程准确 |

> **注意**：模拟器上 **不会** 运行 `swimlane_converter.py`（即不产生
> `merged_swimlane_*.json`），因为 `_generate_swimlane()` 仅在非 sim 平台执行。
> 但 `perf_swimlane_*.json` 本身已经是 Chrome Trace Event Format，
> **可以直接在 Perfetto 中打开查看**。

### 2.2 开启 runtime profiling

在运行任何 example 时，添加 `--runtime-profiling` 参数即可（模拟器和真机通用）：

```bash
# 模拟器上生成泳道图
python examples/beginner/matmul.py -p a2a3sim --runtime-profiling

# 真实设备上生成泳道图
python examples/beginner/matmul.py -p a2a3 --runtime-profiling

# Qwen3 示例
python examples/models/qwen3/qwen3_32b_decode.py -p a2a3 --runtime-profiling
```

如果通过 Python API 调用，在 `RunConfig` 中设置 `runtime_profiling=True`：

```python
from pypto.runtime.runner import RunConfig

config = RunConfig(
    platform="a2a3sim",       # 或 "a2a3"、"a5sim"、"a5"
    device_id=0,
    runtime_profiling=True,   # 开启泳道图采集
)
```

> **注意**：开启 `runtime_profiling` 时，系统会自动启用 `save_kernels=True`，
> 因为泳道文件需要引用 `kernel_config.py` 中的内核元数据。

### 2.3 产物输出

#### 真实设备（`a2a3` / `a5`）

```
<work_dir>/
└── swimlane_data/
    ├── perf_swimlane_<timestamp>.json    # 原始性能数据（由 C++ DeviceRunner 产生）
    └── merged_swimlane_<timestamp>.json  # 合并后的泳道 JSON（推荐查看此文件）
```

- **`perf_swimlane_*.json`** — C++ 运行时导出的原始 Chrome Trace JSON，最初写入
  当前工作目录的 `outputs/` 下，随后被 Python 层移入 `swimlane_data/`
- **`merged_swimlane_*.json`** — 由 `swimlane_converter.py` 将原始数据、CANN
  设备日志（`*.log`）、内核配置（`kernel_config.py`）合并后生成，包含更丰富的
  调度器深度分析信息

#### 模拟器（`a2a3sim` / `a5sim`）

```
<work_dir>/
└── swimlane_data/
    └── perf_swimlane_<timestamp>.json    # 原始性能数据（可直接在 Perfetto 中查看）
```

模拟器上**没有** `merged_swimlane_*.json`，因为代码中跳过了转换步骤：

```python
# pypto/runtime/runner.py, _collect_swimlane_data()
if not platform.endswith("sim"):
    _generate_swimlane(...)   # 仅真实设备执行
```

### 2.4 C++ 层的生成机制

无论模拟器还是真机，C++ `DeviceRunner::run()` 在 `enable_profiling=True` 时
都会执行相同的采集逻辑：

```cpp
// device_runner.cpp (sim 和 onboard 共用逻辑)
if (runtime.enable_profiling) {
    perf_collector_.collect_all();       // 收集所有核心的 PerfRecord
    export_swimlane_json();              // 导出到 outputs/perf_swimlane_<ts>.json
}
```

`PerformanceCollector` 的完整生命周期：
1. `initialize()` — 在设备上分配 PerfBuffer（每核一个）+ PhaseBuffer（每 AICPU 线程一个）
2. 执行期间 — AICore 将计时写入 PerfBuffer 的 WIP 暂存槽，AICPU 提交到 records 数组
3. `collect_all()` — Host 通过 memcpy（sim）或 rtMemcpy（真机）将数据拷回
4. `export_swimlane_json()` — 序列化为 Chrome Trace Event Format JSON
5. `finalize()` — 释放设备内存

---

## 3. 模拟器泳道图 vs 真机泳道图

### 3.1 模拟器泳道图的含义

模拟器上的时间戳来自 CPU 模拟的 `get_sys_cnt`（而非 NPU 硬件计数器），因此：

- **有意义的信息**：
  - 各内核任务的 **执行顺序** 和 **调度流程**
  - 多核/多线程之间的 **逻辑并行关系**
  - AICPU 编排器各阶段（sync、alloc、param_copy、lookup、fanin 等）的 **相对耗时比例**
  - 任务间的 **依赖关系**（通过 fanout_tasks 可视化）
  - 调度逻辑是否正确（如死锁、饥饿等问题）

- **无实际意义的信息**：
  - **绝对耗时**（CPU 模拟速度与 NPU 完全不同，不能用来评估真实性能）
  - **硬件利用率**（无真实硬件计数器数据）

### 3.2 适用场景对比

| 场景 | 模拟器泳道图 | 真机泳道图 |
|------|-------------|-----------|
| 调试调度逻辑 | 适用 | 适用 |
| 验证任务依赖关系 | 适用 | 适用 |
| 分析真实性能瓶颈 | 不适用 | **适用** |
| 评估硬件利用率 | 不适用 | **适用** |
| 比较优化前后绝对耗时 | 不适用 | **适用** |
| CI 中验证调度正确性 | **适用** | 需要硬件 |

---

## 4. 查看泳道图

### 4.1 是否需要额外配置？

**不需要任何额外安装或配置。** Perfetto UI 是 Google 提供的纯前端 Web 应用，
无需安装、无需登录、无需后端服务。只要有浏览器即可使用。

### 4.2 使用 Perfetto UI（推荐）

1. 在浏览器中打开 [https://ui.perfetto.dev/](https://ui.perfetto.dev/)
2. 点击 **Open trace file**
3. 选择要查看的文件：
   - 真机：优先选择 `merged_swimlane_<timestamp>.json`（信息更完整）
   - 模拟器：选择 `perf_swimlane_<timestamp>.json`
4. 即可看到交互式泳道时间线视图

### 4.3 Perfetto 基本操作

| 操作 | 快捷键 / 鼠标 |
|------|---------------|
| 缩放时间轴 | `Ctrl + 滚轮` 或 `W` / `S` |
| 平移时间轴 | 按住拖拽 或 `A` / `D` |
| 选择事件 | 单击某个色块 |
| 查看事件详情 | 选中后看底部面板 |
| 搜索事件 | `/` 键打开搜索框 |
| 标记时间区间 | 拖拽选取一段区间，底部显示统计信息 |

### 4.4 使用 chrome://tracing（备选）

在 Chrome 浏览器地址栏输入 `chrome://tracing`，点击 **Load** 加载 JSON 文件。
功能较 Perfetto 简单，但无需联网。

---

## 5. 泳道图的完整工作流程

### 5.1 真实设备流程

```
运行程序（runtime_profiling=True, platform="a2a3"）
    |
    +-- 执行前：_snapshot_profiling_state()
    |       +-- 快照 outputs/ 下现有的 perf_swimlane_*.json
    |       +-- 快照 CANN 设备日志目录下现有的 *.log
    |
    +-- C++ DeviceRunner::run()（enable_profiling=True）
    |       +-- AICore/AICPU 执行，写入 PerfRecord
    |       +-- perf_collector_.collect_all()
    |       +-- export_swimlane_json() -> outputs/perf_swimlane_<ts>.json
    |
    +-- 执行后：_collect_swimlane_data()
            +-- 发现新的 perf_swimlane_*.json -> 移入 swimlane_data/
            +-- _generate_swimlane()
                    +-- 等待新的 CANN 设备日志（最长 15 秒）
                    +-- 调用 swimlane_converter.py
                    |   输入：perf_swimlane_*.json + kernel_config.py + device log
                    |   输出：merged_swimlane_<timestamp>.json
                    +-- 打印 "Swimlane JSON written to: ..."
```

### 5.2 模拟器流程

```
运行程序（runtime_profiling=True, platform="a2a3sim"）
    |
    +-- 执行前：_snapshot_profiling_state()
    |       +-- 快照 outputs/ 下现有的 perf_swimlane_*.json
    |       +-- 跳过设备日志快照（sim 无 CANN 日志）
    |
    +-- C++ DeviceRunner::run()（enable_profiling=True）
    |       +-- AICore/AICPU 模拟执行，写入 PerfRecord（CPU 模拟时间戳）
    |       +-- perf_collector_.collect_all()
    |       +-- export_swimlane_json() -> outputs/perf_swimlane_<ts>.json
    |
    +-- 执行后：_collect_swimlane_data()
            +-- 发现新的 perf_swimlane_*.json -> 移入 swimlane_data/
            +-- 跳过 _generate_swimlane()（platform.endswith("sim") 为 True）
            +-- 最终产物：swimlane_data/perf_swimlane_<ts>.json
```

### 5.3 核心代码位置

| 模块 | 函数 | 作用 |
|------|------|------|
| `pypto.runtime.runner` | `_snapshot_profiling_state()` | 执行前快照 perf 文件和设备日志 |
| `pypto.runtime.runner` | `_collect_swimlane_data()` | 执行后收集数据、移动文件 |
| `pypto.runtime.runner` | `_generate_swimlane()` | 仅真机：调用 `swimlane_converter.py` |
| C++ `device_runner.cpp` | `DeviceRunner::run()` | 执行内核，采集 profiling 数据 |
| C++ `performance_collector.h/.cpp` | `PerformanceCollector` | 底层 PerfRecord 采集和 JSON 导出 |

---

## 6. 在 scene_test 中使用

`simpler_setup.scene_test` 模块也内置了泳道图采集功能，适用于通过 scene test
框架运行的测试用例。启用方式为添加 `--enable-profiling` 参数。

每个 test case 执行后会自动：

1. 检测新的 `perf_swimlane_*.json`（通过 `_snapshot_perf_files()` 前后对比）
2. 将文件重命名为包含 case 标签的格式（如 `perf_swimlane_<ts>_<case>.json`）
3. 调用 `swimlane_converter.py` 生成 `merged_swimlane_<ts>_<case>.json`

> **注意**：当 `--rounds > 1` 时，profiling 会被自动禁用（仅在第一轮采集）。

---

## 7. 常见问题

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| 没有生成 `perf_swimlane_*.json` | 未传入 `--runtime-profiling` 参数 | 确认命令行包含 `--runtime-profiling` 或 API 中 `runtime_profiling=True` |
| 模拟器上没有 `merged_swimlane_*.json` | 正常行为，模拟器跳过转换步骤 | 直接在 Perfetto 中打开 `perf_swimlane_*.json` 即可 |
| `swimlane_converter.py` not found | Simpler 子模块中 `tools/` 目录缺失 | 确保 Simpler 完整安装，检查 `simpler_root / tools / swimlane_converter.py` |
| `No perf_swimlane_*.json found` | 执行时 profiling 未正常采集 | 确认 `enable_profiling=True` 已传递给 `execute_on_device()` |
| 日志警告 `no new device log found` | CANN 设备日志写入超时（>15s） | 检查 `ASCEND_WORK_PATH` 环境变量和日志目录权限 |
| Perfetto 打开 JSON 后空白 | JSON 文件内容为空或格式不正确 | 用 `python -m json.tool <文件名>.json` 验证格式 |
| 模拟器泳道图时间看起来不合理 | 模拟器时间戳来自 CPU 模拟，绝对值无意义 | 关注执行顺序和相对关系，不要用模拟器数据评估真实性能 |

---

## 8. 参考资料

- [Perfetto UI](https://ui.perfetto.dev/) — 在线 Trace 查看器（纯前端，无需安装）
- [Chrome Trace Event Format 规范](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/) — JSON 格式定义
- `docs/pto2_rt.md` 第 17 节 — 提到使用 Perfetto swimlane 进行性能分析的策略
