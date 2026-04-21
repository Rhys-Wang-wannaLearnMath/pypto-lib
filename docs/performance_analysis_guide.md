# 性能数据分析指南

本文档全面介绍 PyPTO-Lib 项目中 `build_output/` 产出的性能数据结构、各项指标含义、
分析方法论和优化指引。适用于在真实设备（910B）上运行后产生的性能数据。

---

## 目录

1. [数据源全景](#1-数据源全景)
2. [任务级性能指标](#2-任务级性能指标)
3. [调度器指标](#3-调度器指标-scheduler-phases)
4. [编排器指标](#4-编排器指标-orchestrator-phases)
5. [内存使用指标](#5-内存使用指标-memory-report)
6. [核心映射与硬件拓扑](#6-核心映射与硬件拓扑)
7. [端到端分析方法论](#7-端到端分析方法论)
8. [常见性能问题模式与优化方向](#8-常见性能问题模式与优化方向)
9. [工具使用指南](#9-工具使用指南)

---

## 1. 数据源全景

### 1.1 build_output 目录结构

每次编译和运行会在 `build_output/` 下生成一个以 `<ProgramName>_<timestamp>` 命名的
目录。典型结构如下：

```
build_output/
└── Qwen3Scope1_20260419_204240/
    ├── kernel_config.py                  # 内核配置：func_id → 内核名称/源码/核心类型
    ├── data/                             # 输入/输出张量数据（.pt 格式）
    │   ├── in/                           # 输入张量
    │   └── out/                          # 输出张量（golden 比对用）
    ├── kernels/                          # 编译后的内核源码和二进制
    │   ├── aic/                          # AIC（cube）内核 — .cpp 和 .so
    │   └── aiv/                          # AIV（vector）内核 — .cpp 和 .so
    ├── orchestration/                    # 编排函数源码和二进制 — .cpp 和 .so
    ├── passes_dump/                      # 编译器各 pass 后的 IR dump
    ├── ptoas/                            # PTO-ISA 汇编中间产物
    ├── report/                           # 编译报告
    │   └── memory_after_AllocateMemoryAddr.txt  # 内存使用报告
    └── swimlane_data/                    # 运行时性能数据
        ├── perf_swimlane_*.json          # 原始性能数据（项目自定义格式）
        └── merged_swimlane_*.json        # 转换后的 Chrome Trace 格式（供 Perfetto 查看）
```

### 1.2 各产物说明

| 产物 | 生成时机 | 用途 |
|------|---------|------|
| **`kernel_config.py`** | 编译阶段 | 定义 func_id 到内核名称、源码路径、核心类型（aic/aiv）的映射；定义 runtime 配置（线程数、block_dim） |
| **`perf_swimlane_*.json`** | 运行阶段（C++ DeviceRunner 导出） | 原始性能数据：任务执行时间、调度器各阶段耗时、编排器各阶段耗时、核心映射 |
| **`merged_swimlane_*.json`** | 运行后（swimlane_converter.py 转换） | Chrome Trace Event 格式，可直接在 Perfetto 中可视化查看 |
| **`memory_after_AllocateMemoryAddr.txt`** | 编译阶段（AllocateMemoryAddr pass 后） | 各内核函数在不同内存空间的使用量、上限和利用率 |
| **`passes_dump/`** | 编译阶段 | 编译器各 pass 后的 IR 快照，用于调试编译器行为 |
| **`kernels/`** | 编译阶段 | 内核 C++ 源码和编译后的 .so 二进制 |
| **`orchestration/`** | 编译阶段 | 编排函数 C++ 源码和编译后的 .so 二进制 |

### 1.3 数据格式详解：perf_swimlane_*.json

该文件是整个性能分析的核心数据源，由 C++ `PerformanceCollector` 在设备运行后导出。
格式为项目自定义 JSON（非 Chrome Trace 格式），顶层结构如下：

```json
{
  "version": 2,
  "tasks": [...],
  "aicpu_scheduler_phases": [...],
  "aicpu_orchestrator_phases": [...],
  "core_to_thread": [...]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `version` | int | 格式版本号（当前为 2） |
| `tasks` | list[dict] | 所有内核任务的执行记录 |
| `aicpu_scheduler_phases` | list[list[dict]] | 各调度器线程的阶段时间线（外层索引=线程号） |
| `aicpu_orchestrator_phases` | list[list[dict]] | 各编排器线程的阶段时间线（外层索引=线程号） |
| `core_to_thread` | list[int] | 核心 ID 到调度器线程号的映射表 |

---

## 2. 任务级性能指标

### 2.1 任务记录字段

每个 task 对象包含以下字段：

| 字段 | 类型 | 含义 |
|------|------|------|
| `task_id` | int (u64) | PTO2 任务 ID，编码为 `(ring_id << 32) \| local_id` |
| `func_id` | int | 内核函数 ID，对应 `kernel_config.py` 中的 `KERNELS[].func_id` |
| `core_id` | int | 执行该任务的硬件核心编号（0-23 为 AIC，24+ 为 AIV） |
| `core_type` | str | 核心类型：`"aic"`（cube 矩阵计算）或 `"aiv"`（vector 向量计算） |
| `ring_id` | int | 任务所在的 ring buffer ID |
| `start_time_us` | float | AICore 内核开始执行的时间戳（微秒） |
| `end_time_us` | float | AICore 内核执行完成的时间戳（微秒） |
| `duration_us` | float | 内核执行时间 = `end_time_us - start_time_us` |
| `dispatch_time_us` | float | AICPU 调度器将任务派发给 AICore 的时间戳（微秒） |
| `finish_time_us` | float | AICPU 调度器观察到任务完成的时间戳（微秒） |
| `fanout` | list[int] | 后继任务 ID 列表（依赖关系：当前任务完成后才能执行这些任务） |
| `fanout_count` | int | 后继任务数量 |

### 2.2 核心时间指标

以下指标从 task 字段计算得出，是性能分析的基础：

```
时间线：
  dispatch_time    start_time              end_time    finish_time
       |               |                      |             |
       |── Head OH ──▶|◀──── Exec Time ────▶|── Tail OH ──|
       |◀────────────── Latency ───────────────────────────▶|
```

| 指标 | 计算方式 | 含义 |
|------|---------|------|
| **Exec Time（执行时间）** | `end_time_us - start_time_us` = `duration_us` | AICore 上的纯内核执行时间。这是任务的实际计算耗时 |
| **Head Overhead（启动开销）** | `start_time_us - dispatch_time_us` | 从 AICPU 派发任务到 AICore 开始执行的延迟。包括硬件握手、参数传递等 |
| **Tail Overhead（完成检测延迟）** | `finish_time_us - end_time_us` | 从 AICore 执行完成到 AICPU 调度器检测到完成的延迟。取决于调度器轮询频率 |
| **Latency（端到端延迟）** | `finish_time_us - dispatch_time_us` | AICPU 视角的完整任务耗时 = Head OH + Exec + Tail OH |
| **Exec/Latency Ratio（执行效率比）** | `Exec Time / Latency × 100%` | 表示纯计算在总延迟中的占比。越高越好；低值意味着调度开销大 |

### 2.3 指标解读

- **Exec Time 高、Latency 也高** → 内核计算本身就慢，需要优化算法或 tiling 策略
- **Exec Time 低、Latency 高** → 调度开销过大（Head OH 或 Tail OH 过高）
- **Head OH 高** → 任务派发到执行启动的延迟大，可能的原因：
  - AICore 正在执行其他任务（排队等待）
  - 硬件握手开销
- **Tail OH 高** → 调度器发现任务完成的延迟大，可能的原因：
  - 调度器轮询不够频繁（idle 时间长）
  - 调度器在处理其他核心的任务（complete 阶段耗时长）

### 2.4 task_id 编码

PTO2 的 task_id 是 64 位无符号整数，编码规则为：

```
task_id = (ring_id << 32) | local_id
```

- **ring_id**（高 32 位的低 8 位）：ring buffer 编号
- **local_id**（低 32 位）：ring 内的本地任务序号

显示时通常格式化为 `r{ring}t{local}`（如 `r2t1`），当 ring=0 时简写为 `t{local}`。

### 2.5 fanout 依赖关系

`fanout` 数组定义了任务间的 **生产者-消费者依赖关系**：

- 当前任务完成后，其 fanout 中列出的后继任务的 **fanin 计数减一**
- 当后继任务的所有前驱都完成（fanin 计数归零），该任务变为 **就绪状态**，可被调度器派发
- 空 `fanout` 表示该任务是叶节点（无后继依赖）

**关键路径**：从第一个任务到最后一个任务的最长依赖链决定了程序的理论最短执行时间。
优化关键路径上的任务对整体性能影响最大。

---

## 3. 调度器指标（Scheduler Phases）

### 3.1 数据结构

`aicpu_scheduler_phases` 是一个二维数组：外层索引为调度器线程号，内层为该线程的
阶段记录列表。典型配置有 4 个 AICPU 线程（线程 0-2 为调度器，线程 3 为编排器）。

每条记录包含：

| 字段 | 含义 |
|------|------|
| `phase` | 阶段名称：`scan` / `dispatch` / `complete` / `idle` |
| `start_time_us` | 阶段开始时间（微秒） |
| `end_time_us` | 阶段结束时间（微秒） |
| `loop_iter` | 调度器主循环迭代次数 |
| `tasks_processed` | 本次阶段处理的任务数量 |

### 3.2 各阶段含义

| 阶段 | 含义 | 性能意义 |
|------|------|---------|
| **scan** | 扫描就绪队列，查找可派发的任务 | scan 时间长 → 就绪队列大或扫描算法效率低 |
| **dispatch** | 将就绪任务派发到 AICore 执行 | dispatch 时间是调度核心开销 |
| **complete** | 检查已完成的任务，更新后继任务的 fanin 计数 | complete 时间长 → 完成的任务多或依赖链复杂 |
| **idle** | 无就绪任务可调度，等待 | idle 占比高 → 任务提交速度跟不上执行速度，或任务间依赖过深 |

### 3.3 调度效率评估

- **idle 占比** = idle 总时间 / 调度器活跃总时间 × 100%
  - < 10%：调度饱和（好）
  - 10%-30%：有一定空闲（可接受）
  - \> 30%：调度器大量空闲（需优化任务图结构或提交策略）
- **dispatch 占比** = dispatch 总时间 / 非 idle 总时间
  - 表示调度器用于实际派发任务的时间比例
- **各线程负载均衡**：比较各调度器线程的总处理任务数和活跃时间，不均衡说明核心分配需要调整

---

## 4. 编排器指标（Orchestrator Phases）

### 4.1 数据结构

`aicpu_orchestrator_phases` 同样是二维数组（多线程编排器），每条记录包含：

| 字段 | 含义 |
|------|------|
| `phase` | 阶段名称（见下表） |
| `start_time_us` | 阶段开始时间（微秒） |
| `end_time_us` | 阶段结束时间（微秒） |
| `submit_idx` | 任务提交序号（编排器提交任务的顺序索引） |
| `task_id` | 对应的 PTO2 任务 ID |

### 4.2 各阶段含义

编排器负责构建任务图并提交任务到调度器。每提交一个任务经过以下阶段：

| 阶段 | 含义 | 性能意义 |
|------|------|---------|
| **orch_alloc** | 在 ring buffer 中分配任务槽位 | alloc 时间长 → ring buffer 满，等待槽位释放 |
| **orch_sync** | 同步 TensorMap 状态 | sync 时间长 → TensorMap 竞争或缓存失效 |
| **orch_lookup** | 在 TensorMap 中查找输入张量的依赖关系 | lookup 时间长 → TensorMap 条目多或哈希冲突 |
| **orch_insert** | 将输出张量插入 TensorMap | insert 时间长 → TensorMap 扩容或冲突 |
| **orch_params** | 拷贝任务参数（输入/输出张量描述符）到任务槽 | params 时间长 → 参数多或张量描述符复杂 |
| **orch_fanin** | 等待前驱任务完成（设置 fanin 计数） | fanin 时间长 → 前驱任务未完成，编排器阻塞等待 |
| **orch_scope_end** | scope 结束处理（释放资源） | 通常很短 |

### 4.3 编排器瓶颈识别

- **orch_alloc 占比高** → ring buffer 容量不足，编排器频繁等待空闲槽位。
  可能的优化：增大 ring buffer 大小、减少任务粒度
- **orch_fanin 占比高** → 编排器提交任务的速度快于执行速度，在等待前驱完成。
  说明任务执行是瓶颈而非编排
- **orch_params 占比高** → 参数拷贝开销大。可能的优化：减少任务参数数量、使用更紧凑的参数编码
- **orch_lookup / orch_insert 占比高** → TensorMap 操作成为瓶颈。可能的优化：减少 TensorMap 条目数

---

## 5. 内存使用指标（Memory Report）

### 5.1 报告来源

`report/memory_after_AllocateMemoryAddr.txt` 由编译器 `AllocateMemoryAddr` pass
在为每个内核函数分配内存地址后生成。这是 **编译时静态分析** 的结果，反映内核代码
对各硬件内存空间的使用情况。

### 5.2 内存空间含义（910B 架构）

Ascend 910B 处理器的 AICore 包含多个专用内存空间：

| 内存空间 | 硬件名称 | 容量（典型） | 用途 |
|---------|---------|-------------|------|
| **Vec** | Unified Buffer (UB) | 192 KB | AIV（vector 核心）的工作缓冲区，存放向量运算的输入/输出 tile |
| **Mat** | L1 Buffer | 512 KB | AIC（cube 核心）的矩阵缓冲区，存放 matmul 的输入/输出 tile |
| **Left** | L0A | 64 KB | Cube 单元的左矩阵输入寄存器文件 |
| **Right** | L0B | 64 KB | Cube 单元的右矩阵输入寄存器文件 |
| **Acc** | L0C / Accumulator | 128 KB | Cube 单元的累加器，存放 matmul 中间结果 |

### 5.3 报告字段解读

```
--- qwen3_scope1_incore_1 ---

  Space  |  Used       |  Limit      |  Usage   |  MemRefs
  -------+-------------+-------------+----------+---------
  Mat    |    80.0 KB  |   512.0 KB  |   15.6%  |  2
  Left   |    16.0 KB  |    64.0 KB  |   25.0%  |  1
  Right  |    64.0 KB  |    64.0 KB  |  100.0%  |  1
  Acc    |     4.0 KB  |   128.0 KB  |    3.1%  |  1
```

| 列 | 含义 |
|----|------|
| **Space** | 内存空间名称 |
| **Used** | 编译器分配的实际使用量 |
| **Limit** | 该空间的硬件上限 |
| **Usage** | 使用率 = Used / Limit × 100% |
| **MemRefs** | 该空间中的内存引用（tile）数量 |

### 5.4 使用率解读

| 使用率范围 | 评价 | 说明 |
|-----------|------|------|
| **< 10%** | 极低 | 内核使用的 tile 很小，可能可以增大 tile 尺寸提高计算效率 |
| **10% - 50%** | 正常偏低 | 有提升空间，视具体算法而定 |
| **50% - 90%** | 良好 | 充分利用了内存空间 |
| **90% - 100%** | 接近上限 | 需要注意是否有扩展余地；如果需要更大的 tile 则可能成为瓶颈 |
| **100%** | 满载 | 已达硬件上限，无法分配更大的 tile |

### 5.5 典型分析场景

- **AIV 内核 Vec 使用率极低**：内核只操作小 tile，可考虑增大 tile 尺寸以提高向量单元利用率
- **AIC 内核 Right 使用率 100%**：右矩阵缓冲区已满，tiling 受限于 L0B 容量。
  如果 matmul 性能不理想，需要考虑调整 tiling 策略以平衡 L0A/L0B 使用
- **AIC 内核 Acc 使用率极低**：累加器空间大量浪费，可增大输出 tile 尺寸
- **AIC 内核 Mat 使用率低**：L1 缓冲区未充分利用，可通过 double buffering 或更大的 tile 提高利用率

---

## 6. 核心映射与硬件拓扑

### 6.1 core_to_thread 映射

`core_to_thread` 是一个数组，索引为 core_id，值为调度器线程号。例如：

```json
[0, 1, 0, 1, 0, 1, ...]
```

表示 core 0 由调度器线程 0 负责，core 1 由线程 1 负责，以此类推。

### 6.2 910B 硬件拓扑

典型配置（block_dim=24）：

- **24 个 AIC 核心**（core_id 0-23）：cube 矩阵计算单元
- **24 个 AIV 核心**（core_id 24-47）：vector 向量计算单元
- AIC core_id N 与 AIV core_id N+24 通常共享 L1 缓冲区
- **4 个 AICPU 线程**：
  - 线程 0-2：调度器（Scheduler），负责派发任务到 AICore
  - 线程 3：编排器（Orchestrator），负责构建任务图

### 6.3 核心类型与内核映射

通过 `kernel_config.py` 的 `KERNELS` 列表可知每个 func_id 运行在哪种核心上：

- `"core_type": "aic"` → 在 AIC（cube）核心上执行，典型操作：matmul
- `"core_type": "aiv"` → 在 AIV（vector）核心上执行，典型操作：RMSNorm、SiLU、elementwise

---

## 7. 端到端分析方法论

### 7.1 性能瓶颈定位流程

```
1. 总览
   └── 查看总执行时间（最早 dispatch → 最晚 finish）
   └── 查看任务数量和 func_id 分布

2. 任务级分析
   └── 按 func_id 分组统计 Avg Exec / Avg Latency
   └── 找出 Exec 最高的 func_id → 计算瓶颈
   └── 找出 Exec/Latency 最低的 func_id → 调度瓶颈

3. 调度器分析
   └── 查看各线程 idle 占比
   └── 如果 idle 高 → 任务提交不足或依赖过深
   └── 如果 dispatch 高 → 调度本身开销大

4. 编排器分析
   └── 查看各阶段时间占比
   └── alloc 高 → ring buffer 不足
   └── fanin 高 → 执行是瓶颈
   └── params 高 → 参数传递开销大

5. 内存分析
   └── 查看各内核的内存使用率
   └── 使用率极低 → 可增大 tile 提高效率
   └── 使用率满载 → 可能限制了 tile 大小

6. 综合诊断
   └── 结合以上分析给出优化建议
```

### 7.2 跨 Scope 对比

当一个模型包含多个 scope（如 Qwen3 的 Scope1/2/3）时，跨 scope 对比有助于发现：

- **哪个 scope 是时间瓶颈**：比较各 scope 的总执行时间
- **各 scope 的计算特征差异**：比较 AIC/AIV 任务比例和执行时间
- **调度效率差异**：某些 scope 可能因为任务图结构不同而有不同的调度效率
- **内存使用差异**：不同 scope 的内核可能有不同的 tiling 策略

### 7.3 关键比率

| 比率 | 公式 | 健康范围 | 含义 |
|------|------|---------|------|
| **Exec/Latency** | ΣExec / ΣLatency | > 80% | 越高越好，低值说明调度开销大 |
| **Scheduler Idle%** | Σidle / Σ(所有阶段) | < 10% | 低值说明调度器始终繁忙 |
| **Orch Alloc%** | Σorch_alloc / Σ(所有 orch 阶段) | < 10% | 高值说明 ring buffer 压力大 |
| **Orch Fanin%** | Σorch_fanin / Σ(所有 orch 阶段) | — | 高值说明编排器在等待前驱完成 |
| **Memory Utilization** | Used / Limit | 30%-90% | 过低浪费资源，过高限制灵活性 |

---

## 8. 常见性能问题模式与优化方向

### 8.1 调度器空闲过多

**症状**：scheduler idle 占比 > 30%

**可能原因与优化**：
- 任务图依赖过深（串行链过长）→ 优化算法减少依赖深度
- 编排器提交速度不足 → 检查 orch_alloc/orch_fanin 阶段
- 任务粒度过大（少量大任务）→ 拆分为更多小任务提高并行度

### 8.2 Exec/Latency 比过低

**症状**：某 func_id 的 Exec/Latency < 50%

**可能原因与优化**：
- Head OH 高 → AICore 上有排队等待，检查核心利用率和负载均衡
- Tail OH 高 → 调度器检测完成延迟大，考虑优化调度器轮询策略

### 8.3 编排器 alloc 阻塞

**症状**：orch_alloc 阶段出现数毫秒级别的长尾

**可能原因与优化**：
- Ring buffer 容量不足 → 增大 ring buffer
- 任务完成回收不及时 → 检查调度器 complete 阶段效率

### 8.4 单内核执行时间过长

**症状**：某 func_id 的 Avg Exec 远高于其他

**可能原因与优化**：
- Tiling 不优化 → 检查内存使用率，调整 tile 大小
- 算法效率低 → 检查内核源码（`kernels/aic/` 或 `kernels/aiv/`）
- 数据搬运瓶颈 → 通过 Perfetto 泳道图查看内核内部时间线

### 8.5 负载不均衡

**症状**：不同核心的任务数或执行时间差异大

**可能原因与优化**：
- 任务分配策略不合理 → 检查编排函数中的任务分配逻辑
- 某些核心执行更耗时的任务 → 考虑任务拆分使各核心负载均衡

---

## 9. 工具使用指南

### 9.1 一键分析脚本

项目提供 `tools/perf_analyzer.py` 用于自动分析 `build_output/` 下的性能数据：

```bash
# 分析所有 scope
python tools/perf_analyzer.py build_output/

# 分析单个 scope
python tools/perf_analyzer.py build_output/Qwen3Scope1_20260419_204240/

# 指定报告输出路径
python tools/perf_analyzer.py build_output/ -o my_report.md

# 详细模式
python tools/perf_analyzer.py build_output/ -v
```

脚本会：
1. 自动扫描目录下的 `perf_swimlane_*.json`、`kernel_config.py`、`memory_after_AllocateMemoryAddr.txt`
2. 计算所有维度的性能指标
3. 在终端输出彩色摘要
4. 生成结构化 Markdown 报告（默认输出到 `build_output/perf_report.md`）

### 9.2 与 Perfetto 配合

`perf_analyzer.py` 提供全局定量分析，Perfetto 提供交互式时间线查看。
推荐工作流：

1. 先运行 `perf_analyzer.py` 获取宏观指标和瓶颈诊断
2. 根据诊断结果，在 Perfetto 中打开对应的 `merged_swimlane_*.json` 深入查看特定任务或时间段
3. 使用 Perfetto SQL 查询精确分析特定指标

### 9.3 与 swimlane_converter.py 配合

- `swimlane_converter.py`（位于 `examples/`）负责将 `perf_swimlane_*.json` 转换为
  Perfetto 可打开的 `merged_swimlane_*.json`，同时输出基础的任务统计
- `perf_analyzer.py` 是更全面的分析工具，覆盖调度器、编排器、内存使用等多个维度，
  并支持跨 scope 对比和自动诊断

详细的泳道图使用方法请参阅 `docs/swimlane_guide.md`。
