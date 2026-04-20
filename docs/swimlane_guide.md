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
> `perf_swimlane_*.json` 是项目自定义格式（`tasks` 数组），**不能**直接在 Perfetto
> 中打开。如需在模拟器上查看泳道图，需手动运行 `swimlane_converter.py` 转换
> （详见第 4.2 节）。

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
    └── perf_swimlane_<timestamp>.json    # 原始性能数据（自定义格式，需转换后才能用 Perfetto 查看）
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
4. `export_swimlane_json()` — 序列化为项目自定义格式 JSON（`tasks` 数组 + 调度器阶段数据）
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

**不需要任何额外安装或配置。** [Perfetto UI](https://ui.perfetto.dev/) 是
Google 提供的纯前端 Web 应用，无需安装、无需登录、无需后端服务。
推荐使用 Chrome / Edge 浏览器（Firefox 也可以但部分交互不如 Chromium 流畅）。

### 4.2 两种 JSON 文件的区别

在打开文件之前，需要了解产物的区别：

| 文件 | 格式 | 来源 | 能否直接用 Perfetto 打开 |
|------|------|------|:------------------------:|
| `merged_swimlane_*.json` | **Chrome Trace Event Format**（`traceEvents` 数组） | `swimlane_converter.py` 转换后生成 | **能**，直接拖入即可 |
| `perf_swimlane_*.json` | 项目自定义格式（`tasks` 数组 + 调度器阶段数据） | C++ `PerformanceCollector` 直接导出 | **不能**直接打开，需先转换 |

> **模拟器用户注意**：模拟器上只产生 `perf_swimlane_*.json`（自定义格式），
> 无法直接拖入 Perfetto 查看。如需可视化，需手动运行 `swimlane_converter.py`
> 转换为 `merged_swimlane_*.json`，或编写脚本将其转为 Chrome Trace 格式。
>
> 真机上 `swimlane_converter.py` 会自动执行，直接查看 `merged_swimlane_*.json` 即可。

### 4.3 打开泳道图

**方法一：拖拽（最快）**

直接将 `merged_swimlane_*.json` 文件拖入 Perfetto UI 页面即可。

**方法二：菜单打开**

1. 浏览器打开 [https://ui.perfetto.dev/](https://ui.perfetto.dev/)
2. 点击左上角 **Open trace file**（或按快捷键 `Ctrl+O`）
3. 选择 `merged_swimlane_*.json` 文件
4. 等待加载完成，即可看到泳道时间线视图

**方法三：从远程服务器查看（无图形界面时）**

如果 JSON 文件在远程 NPU 服务器上，本地没有图形界面：

```bash
# 方法 A：用 scp 拷贝到本地
scp user@npu-server:/path/to/merged_swimlane_*.json ./

# 方法 B：用 VS Code Remote 直接下载文件

# 方法 C：用 Python 启动临时 HTTP 服务器，浏览器直接下载
cd /path/to/swimlane_data
python -m http.server 8080
# 然后在本地浏览器打开 http://npu-server:8080/ 下载文件
```

### 4.4 理解泳道图布局

加载 `merged_swimlane_*.json` 后，Perfetto 会显示多条水平泳道，
每条代表一个硬件执行单元。典型布局如下：

```
┌─────────────────────────────────────────────────────────────────────┐
│  时间轴 →                                                           │
│                                                                     │
│  ┌─ Orchestrator ──────────────────────────────────────────────┐   │
│  │  [sync][alloc][params][lookup][insert][fanin] ...           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─ Scheduler Thread 0 ───────────────────────────────────────┐   │
│  │  [scan][dispatch][complete][scan][dispatch]...              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─ AIC Core 0 (cube) ────────────────────────────────────────┐   │
│  │  [task 0]  [task 4]  [task 8]  ...                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─ AIV Core 0 (vector) ──────────────────────────────────────┐   │
│  │  [task 1] [task 5] [task 9] ...                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─ AIC Core 1 (cube) ────────────────────────────────────────┐   │
│  │  [task 2]  [task 6]  [task 10] ...                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ...                                                               │
└─────────────────────────────────────────────────────────────────────┘
```

各泳道含义：

| 泳道 | 说明 |
|------|------|
| **Orchestrator** | AICPU 编排器线程，负责提交任务。各色块对应编排阶段：`sync`（同步 tensormap）、`alloc`（ring 分配）、`params`（参数拷贝）、`lookup`（依赖查找）、`heap`（堆分配）、`insert`（tensormap 插入）、`fanin`（前驱等待）、`scope_end`（scope 结束处理） |
| **Scheduler Thread N** | AICPU 调度器线程，负责扫描就绪队列并分派任务到 AICore。各色块对应：`scan`（扫描队列）、`dispatch`（分派任务）、`complete`（完成回收）、`idle`（空闲等待） |
| **AIC Core N** | AICore cube 核心（矩阵计算单元），执行 matmul 等 CUBE 操作 |
| **AIV Core N** | AICore vector 核心（向量计算单元），执行 RMSNorm、SiLU 等向量操作 |

### 4.5 Perfetto 导航操作

#### 基础快捷键

| 操作 | 快捷键 / 鼠标 | 说明 |
|------|---------------|------|
| **缩放** | `W` 放大 / `S` 缩小 | 以鼠标位置为中心缩放 |
| **缩放**（备选） | `Ctrl + 滚轮` | 同上 |
| **平移** | `A` 向左 / `D` 向右 | 沿时间轴平移 |
| **平移**（备选） | 按住中键拖拽 | 自由平移 |
| **选择事件** | 单击色块 | 底部面板显示该事件详情 |
| **框选时间区间** | 在时间轴区域拖拽 | 底部面板显示区间统计 |
| **缩放到选区** | 框选后按 `F` | 将视图缩放到选中的时间区间 |
| **全局视图** | `Ctrl+Shift+F` | 缩放到显示完整 trace |
| **搜索** | `/` | 打开搜索框，按名称搜索事件 |
| **标记** | `M` | 在当前位置添加标记 |
| **高亮同名事件** | 右击色块 → Highlight | 高亮所有同名事件 |

#### 展开/折叠泳道

- 点击泳道左侧的 **▶ / ▼** 箭头可展开或折叠子泳道
- 拖拽泳道边缘可调整泳道高度

### 4.6 查看事件详情

单击某个事件色块后，底部面板（**Current Selection**）会显示：

| 字段 | 含义 |
|------|------|
| **Name** | 事件名称（如 task_id、阶段名） |
| **Category** | 事件类别（如 `aic`、`aiv`、`scheduler`） |
| **Start time** | 事件开始时间（相对于 trace 起点） |
| **Duration** | 事件持续时间 |
| **Arguments** | 额外参数，如 `core_id`、`func_id`、`task_id`、`fanout` 等 |

### 4.7 使用 SQL 查询（进阶）

Perfetto 内置 SQL 引擎，可对 trace 数据做精确查询。点击左侧导航栏的
**Query (SQL)** 标签页，输入 SQL 语句：

```sql
-- 查看所有事件按持续时间降序排列（找出最慢的任务）
SELECT name, dur, ts, track_id
FROM slice
ORDER BY dur DESC
LIMIT 20;

-- 统计每个泳道（track）上的事件数和总耗时
SELECT t.name AS track_name, COUNT(*) AS event_count,
       SUM(s.dur) AS total_dur_ns
FROM slice s
JOIN track t ON s.track_id = t.id
GROUP BY t.name
ORDER BY total_dur_ns DESC;

-- 搜索特定 task_id 的事件
SELECT * FROM slice WHERE name LIKE '%task_42%';
```

### 4.8 实用分析技巧

#### 4.8.1 识别性能瓶颈

1. **找最长的色块**：缩放到全局视图（`Ctrl+Shift+F`），最宽的色块就是耗时最长的任务
2. **找空闲间隙**：泳道中色块之间的空白区域表示该核心空闲，空闲过多说明调度效率低
3. **对比 AIC vs AIV**：如果 AIC（cube）泳道很满而 AIV（vector）大量空闲，说明计算瓶颈在向量操作的前驱依赖上

#### 4.8.2 检查调度开销

1. 查看 **Scheduler** 泳道中 `scan` 和 `idle` 阶段的占比
2. 如果 `idle` 占比过高，说明任务提交速度跟不上核心执行速度
3. 查看 **Orchestrator** 泳道中各阶段耗时比例，找出编排瓶颈（如 `fanin` 过长表示前驱等待时间多）

#### 4.8.3 分析任务依赖链

1. 在 `perf_swimlane_*.json` 中每个 task 记录了 `fanout`（后继任务列表）
2. 在 Perfetto 中单击某个 task 事件，查看 Arguments 中的 `fanout` 字段
3. 搜索 fanout 中的 task_id，追踪依赖链，找到关键路径

#### 4.8.4 对比不同版本

将两次运行的 `merged_swimlane_*.json` 分别在两个浏览器标签页中打开，
对比同一任务的持续时间变化。也可以用 SQL 查询导出 CSV 做进一步分析：

```sql
-- 导出所有事件的名称和持续时间
SELECT name, dur / 1000.0 AS dur_us FROM slice ORDER BY ts;
```

点击查询结果下方的 **Download as CSV** 按钮保存。

### 4.9 使用 chrome://tracing（备选）

在 Chrome 浏览器地址栏输入 `chrome://tracing`，点击 **Load** 加载
`merged_swimlane_*.json` 文件。

与 Perfetto UI 对比：

| 特性 | Perfetto UI | chrome://tracing |
|------|:-----------:|:----------------:|
| 需联网 | 首次需要 | 不需要 |
| SQL 查询 | 支持 | 不支持 |
| 大文件性能 | 更好 | 一般 |
| 搜索/过滤 | 丰富 | 基础 |
| 快捷键 | WASD + 丰富 | WASD |

> **推荐**：优先使用 Perfetto UI。仅在无法联网时使用 chrome://tracing。

---

## 5. 手动转换 perf_swimlane 为 merged_swimlane

### 5.1 为什么需要转换

C++ `PerformanceCollector` 导出的 `perf_swimlane_*.json` 是项目 **自定义格式**，
其结构如下：

```json
{
  "version": 2,
  "tasks": [
    {
      "task_id": 12345,
      "func_id": 0,
      "core_id": 0,
      "core_type": "aic",
      "ring_id": 0,
      "start_time_us": 0.000,
      "end_time_us": 15.234,
      "duration_us": 15.234,
      "dispatch_time_us": 0.000,
      "finish_time_us": 16.100,
      "fanout": [12346, 12347],
      "fanout_count": 2
    },
    ...
  ],
  "aicpu_scheduler_phases": [
    [
      {"start_time_us": 0.5, "end_time_us": 1.2, "phase": "scan", "loop_iter": 0, "tasks_processed": 3},
      {"start_time_us": 1.2, "end_time_us": 2.0, "phase": "dispatch", "loop_iter": 0, "tasks_processed": 1},
      ...
    ]
  ],
  "aicpu_orchestrator": {
    "start_time_us": 0.000,
    "end_time_us": 120.456,
    "submit_count": 64,
    "phase_us": {
      "sync": 2.1, "alloc": 5.3, "params": 8.7,
      "lookup": 3.2, "heap": 1.1, "insert": 4.5,
      "fanin": 15.8, "scope_end": 0.9
    }
  },
  "aicpu_orchestrator_phases": [...],
  "core_to_thread": [0, 0, 1, 1]
}
```

Perfetto 无法识别这种格式。它需要的是 **Chrome Trace Event Format**（`traceEvents` 数组）：

```json
{
  "traceEvents": [
    {"ph": "X", "name": "task_0", "cat": "aic", "pid": 1, "tid": 0, "ts": 0.0, "dur": 15.234, "args": {...}},
    {"ph": "X", "name": "scan",   "cat": "scheduler", "pid": 2, "tid": 0, "ts": 0.5, "dur": 0.7, "args": {...}},
    ...
  ]
}
```

`swimlane_converter.py` 的作用就是将自定义格式转为 Chrome Trace 格式，
并可选地合并 CANN 设备日志和 `kernel_config.py` 中的内核元数据。

### 5.2 找到 swimlane_converter.py

`swimlane_converter.py` 位于 **pypto 源码仓库** 的 `runtime/tools/` 目录下，
**不会**被 `pip install` 安装到 `site-packages` 中。你需要确保已克隆 pypto 源码：

```bash
# 如果之前已克隆到 /tmp/pypto，则脚本位于：
ls /tmp/pypto/runtime/tools/swimlane_converter.py

# 如果已删除，重新克隆：
git clone --recurse-submodules --depth=1 https://github.com/hw-native-sys/pypto.git /tmp/pypto
```

自动流程中，`_generate_swimlane()` 通过 `get_simpler_root()` 解析该路径：
- 编辑模式安装（`pip install -e`）：`<pypto-repo>/runtime/tools/swimlane_converter.py`
- wheel 安装：搜索 `simpler_setup._assets` 目录（但 tools/ 不在其中）
- 回退：从当前工作目录向上查找包含 `runtime/` 的 git 仓库

### 5.3 模拟器上手动转换

模拟器的自动流程跳过了转换步骤，但可以手动执行：

```bash
# 基本用法：最少只需传入 perf_swimlane_*.json
python /tmp/pypto/runtime/tools/swimlane_converter.py \
    <work_dir>/swimlane_data/perf_swimlane_20250420_153000.json \
    -o <work_dir>/swimlane_data/merged_swimlane_20250420_153000.json
```

参数说明：

| 参数 | 说明 | 是否必须 |
|------|------|:--------:|
| 第一个位置参数 | `perf_swimlane_*.json` 文件路径 | 是（不传则自动选取 `outputs/` 下最新的） |
| `-o` | 输出文件路径 | 否（默认由输入文件名推导） |
| `-k` | `kernel_config.py` 路径，提供内核名称映射 | 否（真机推荐，模拟器可选） |
| `-d` | 设备编号（用于查找 CANN 设备日志） | 否（仅真机有意义） |
| `--device-log` | 直接指定 CANN 设备日志文件路径 | 否（仅真机有意义） |

模拟器上的典型用法（无设备日志、无 kernel_config）：

```bash
# 最简单的转换——只需传入 perf_swimlane 文件
python /tmp/pypto/runtime/tools/swimlane_converter.py \
    build_output/qwen3_32b_decode_scope1_*/swimlane_data/perf_swimlane_*.json

# 如果有 kernel_config.py，可以加上 -k 获得内核名称映射
python /tmp/pypto/runtime/tools/swimlane_converter.py \
    build_output/qwen3_32b_decode_scope1_*/swimlane_data/perf_swimlane_*.json \
    -k build_output/qwen3_32b_decode_scope1_*/kernel_config.py
```

转换完成后，生成的 `merged_swimlane_*.json` 可直接拖入 Perfetto 查看。

### 5.4 真机 vs 模拟器转换的区别

| 对比项 | 真机转换 | 模拟器手动转换 |
|--------|----------|------------------|
| **触发方式** | 自动（`_generate_swimlane()` 在运行后自动调用） | 手动（需自行运行 `swimlane_converter.py`） |
| **CANN 设备日志** | 自动收集并传入（`--device-log` 或 `-d`） | 无（模拟器无 CANN 日志） |
| **kernel_config.py** | 自动传入（`-k`），提供内核名称映射 | 可手动传入，也可省略 |
| **输出中的 Scheduler 深度分析** | 包含（从 CANN 日志提取 scheduler 线程活动） | 不包含（无 CANN 日志） |
| **时间戳含义** | 确实硬件时钟周期，绝对耗时可信 | CPU 模拟时钟，绝对耗时无意义 |
| **产物内容完整度** | 最完整（tasks + scheduler phases + 设备日志分析） | 基本完整（tasks + scheduler phases，缺少设备日志部分） |

### 5.5 转换后数据的含义

转换后的 `merged_swimlane_*.json` 中每个 Chrome Trace 事件包含：

```json
{
  "ph": "X",           // 事件类型："X" = Complete event（有 ts + dur）
  "name": "task_42",   // 事件名称（task_id 或阶段名）
  "cat": "aic",        // 类别：aic / aiv / scheduler / orchestrator
  "pid": 1,            // 进程 ID（用于分组泳道）
  "tid": 0,            // 线程 ID（对应 core_id 或 thread_id）
  "ts": 123.456,       // 开始时间（微秒）
  "dur": 15.234,       // 持续时间（微秒）
  "args": {            // 额外参数（在 Perfetto 底部面板查看）
    "task_id": 42,
    "func_id": 3,
    "core_type": "aic",
    "fanout": [43, 44]
  }
}
```

各类别事件的含义：

| 类别 (`cat`) | 对应泳道 | 含义 |
|---------------|----------|------|
| `aic` | AIC Core N | 一个 cube 内核任务的执行区间（matmul 等） |
| `aiv` | AIV Core N | 一个 vector 内核任务的执行区间（RMSNorm、SiLU 等） |
| `scheduler` | Scheduler Thread N | 调度器各阶段：`scan`、`dispatch`、`complete`、`idle` |
| `orchestrator` | Orchestrator | 编排器各阶段：`orch_sync`、`orch_alloc`、`orch_params`、`orch_lookup`、`orch_heap`、`orch_insert`、`orch_fanin`、`orch_scope_end` |

#### 模拟器转换后数据的解读要点

- **`ts`、`dur` 的绝对值无意义**：CPU 模拟速度与 NPU 完全不同，不能用于性能评估
- **相对顺序是准确的**：任务的先后执行顺序、并行关系、依赖链都与真机一致
- **`fanout` 依赖关系是准确的**：可用于验证任务图的正确性
- **调度器行为是准确的**：`scan` / `dispatch` / `complete` 的顺序和次数反映真实调度逻辑
- **缺少 CANN 日志深度分析**：真机版本的 merged_swimlane 可能还包含从 CANN 设备日志提取的调度器线程级别的详细信息，模拟器版本不包含这部分

### 5.6 批量转换示例

如果你用 `run_all_tests.sh --profiling` 运行了多个测试，
可以批量转换所有 `perf_swimlane_*.json`：

```bash
CONVERTER=/tmp/pypto/runtime/tools/swimlane_converter.py

for perf in build_output/*/swimlane_data/perf_swimlane_*.json; do
    dir=$(dirname "$perf")
    base=$(basename "$perf" | sed 's/^perf_/merged_/')
    echo "转换: $perf"
    python "$CONVERTER" "$perf" -o "$dir/$base" || echo "  转换失败: $perf"
done

echo "完成。可用 Perfetto 打开 build_output/*/swimlane_data/merged_swimlane_*.json"
```

---

## 6. 泳道图的完整工作流程

### 6.1 真实设备流程

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

### 6.2 模拟器流程

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

### 6.3 核心代码位置

| 模块 | 函数 | 作用 |
|------|------|------|
| `pypto.runtime.runner` | `_snapshot_profiling_state()` | 执行前快照 perf 文件和设备日志 |
| `pypto.runtime.runner` | `_collect_swimlane_data()` | 执行后收集数据、移动文件 |
| `pypto.runtime.runner` | `_generate_swimlane()` | 仅真机：调用 `swimlane_converter.py` |
| C++ `device_runner.cpp` | `DeviceRunner::run()` | 执行内核，采集 profiling 数据 |
| C++ `performance_collector.h/.cpp` | `PerformanceCollector` | 底层 PerfRecord 采集和 JSON 导出 |

---

## 7. 在 scene_test 中使用

`simpler_setup.scene_test` 模块也内置了泳道图采集功能，适用于通过 scene test
框架运行的测试用例。启用方式为添加 `--enable-profiling` 参数。

每个 test case 执行后会自动：

1. 检测新的 `perf_swimlane_*.json`（通过 `_snapshot_perf_files()` 前后对比）
2. 将文件重命名为包含 case 标签的格式（如 `perf_swimlane_<ts>_<case>.json`）
3. 调用 `swimlane_converter.py` 生成 `merged_swimlane_<ts>_<case>.json`

> **注意**：当 `--rounds > 1` 时，profiling 会被自动禁用（仅在第一轮采集）。

---

## 8. 常见问题

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| 没有生成 `perf_swimlane_*.json` | 未传入 `--runtime-profiling` 参数 | 确认命令行包含 `--runtime-profiling` 或 API 中 `runtime_profiling=True` |
| 模拟器上没有 `merged_swimlane_*.json` | 正常行为，模拟器跳过自动转换步骤 | 手动运行 `swimlane_converter.py` 转换（见第 5 节） |
| `swimlane_converter.py` not found | Simpler 子模块中 `tools/` 目录缺失 | 确保 Simpler 完整安装，检查 `simpler_root / tools / swimlane_converter.py` |
| `No perf_swimlane_*.json found` | 执行时 profiling 未正常采集 | 确认 `enable_profiling=True` 已传递给 `execute_on_device()` |
| 日志警告 `no new device log found` | CANN 设备日志写入超时（>15s） | 检查 `ASCEND_WORK_PATH` 环境变量和日志目录权限 |
| Perfetto 打开 JSON 后空白 | JSON 文件内容为空或格式不正确 | 用 `python -m json.tool <文件名>.json` 验证格式 |
| 模拟器泳道图时间看起来不合理 | 模拟器时间戳来自 CPU 模拟，绝对值无意义 | 关注执行顺序和相对关系，不要用模拟器数据评估真实性能 |

---

## 9. 参考资料

- [Perfetto UI](https://ui.perfetto.dev/) — 在线 Trace 查看器（纯前端，无需安装）
- [Chrome Trace Event Format 规范](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/) — JSON 格式定义
- `docs/pto2_rt.md` 第 17 节 — 提到使用 Perfetto swimlane 进行性能分析的策略
