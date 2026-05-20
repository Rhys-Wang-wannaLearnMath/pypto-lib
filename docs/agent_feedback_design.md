# Agent-Friendly Feedback 设计方案

> **目的**:为 `pypto-lib` 增加一个**面向 LLM agent** 的运行反馈层,在不破坏现有人类视角(swimlane / Perfetto / colored 表格)的前提下,从同一份 raw 数据派生出 agent 易于消费、推理、行动的结构化信号。

---

## 0. TL;DR

| 角色 | 现有 view | 计划补齐的 view |
|------|-----------|----------------|
| **人类** | `swimlane_data/merged_swimlane_*.json`(Perfetto)+ `tools/perf_analyzer.py` 终端表格 + `report/*` | 已成熟,**不动** |
| **Agent** | (无) | `build_output/<run>/agent_summary.json`(~1.5 KB,Tier 0)+ 按需 `agent_summary.detail.json`(Tier 1)+ 工具检索 raw(Tier 2) |

落地动作:
1. 新增 `tools/agent_feedback.py` —— 复用 `perf_analyzer.py` 的 `ScopeAnalysis` 数据模型,再做 **去重 / 因果归因 / lever 建议 / delta-vs-baseline** 的派生
2. 可选地在 `golden/runner.py:run()` 末尾自动调用,使每次 `runtime_profiling=True` 的运行都产出 agent 摘要
3. 同时引入 `report/perf_hints_aggregated.json`(去重后的 perf_hint)和 `report/passes_diff.json`(相邻 pass 之间的 op-count delta),作为 Tier 1 派生件

---

## 1. 背景:人类与 Agent 的反馈带宽差异

泳道图把 13523 个 traceEvents 压成一张图,人类用视觉聚类瞬间识别"哪两条道之间存在 gap";同样的 4.3 MB JSON 喂给 LLM agent 会:
- 超出上下文窗口
- 全是 `(start_us, dur, pid, tid)` 坐标,**缺 semantic 标签**
- 重复结构(2672 条 `dependency` 事件)成为 token 噪声 → bias agent 推理

| 维度 | 人类 | LLM Agent |
|------|------|-----------|
| 输入带宽 | 视觉 ~10⁷ bit/s,瞬时识别二维模式 | 文本顺序读 token,~10⁴ token/req |
| 聚合 | 大脑实时做空间聚类 | 必须**输入端就 pre-aggregated** |
| 检索 | 鼠标缩放 + 跳转 + 反复看 | 一次注入,不易再回头(除非有工具) |
| 噪声 | 自动忽略重复 | 重复 = token 浪费 + 注意力被偏移 |
| 因果推理 | 凭直觉假设 + 验证 | 严格依赖输入显式列出的因果 |
| 错误模式 | 走眼 / 疲劳 / 偏见 | 幻觉 / 被无关细节带偏 / 数字截断 |

**核心洞察**:对人类有效的"密集可视化",对 agent 是**低密度高噪声**的负担。Agent 需要的不是"更详细的图",而是 **"数字 + 因果 + 可执行 lever + 预期收益"** 的紧凑结构化文本。

---

## 2. 现有产物盘点(以 `Qwen3Scope2_20260507_112341` 为例)

实际目录布局 ~7 MB:

```
build_output/Qwen3Scope2_20260507_112341/
├── kernel_config.py                      4 KB    func_id ↔ kernel_name 映射 + RUNTIME_CONFIG
├── kernels/                            1.1 MB    5 个 stage 的 .cpp + .o(代码生成产物)
├── orchestration/                      676 KB    .cpp + .so
├── passes_dump/  (31 个 .py)           856 KB    每 pass 后的 IR snapshot,合计 5926 行
├── ptoas/                              180 KB    5 个 stage 的 .cpp + .pto
├── report/
│   ├── perf_hints.log                            编译期 PH00x 提示(每点重复多次)
│   └── memory_after_AllocateMemoryAddr.txt       SRAM 占用率表(已结构化)
└── swimlane_data/                      4.3 MB
    ├── l2_perf_records_*.json                    528 个 raw record
    └── merged_swimlane_*.json                    13523 个 traceEvents(Perfetto 用)
```

外加 stdout 上的 **Task Statistics by Function** 表 + **Scheduler Overhead Deep Dive** 段落。

### 2.1 Agent-friendliness 评分

| 产物 | Agent | 原因 |
|------|-------|------|
| stdout `Task Statistics by Function` | ✅✅ | 5 行 8 列,聚合彻底;只缺 baseline 对比列 |
| `memory_after_AllocateMemoryAddr.txt` | ✅ | 表格化;但缺 `is_bottleneck` 标志 |
| `kernel_config.py` | ✅ | 短小,直接喂 |
| stdout `Scheduler Overhead Deep Dive` | ⚠️ | 数字 OK,但夹了 `"Key insight: ..."` 散文 → 锚定 agent 推理方向 |
| `perf_hints.log` | ❌ | 同 `(file,line,hint_id)` 重复 6×;无 frequency 字段 |
| `passes_dump/*.py` × 31 | ❌ | 5926 行;全喂爆,选 1 个又看不全 |
| `merged_swimlane.json` | ❌❌ | 4.3 MB 纯坐标,无 high-level semantic |
| `l2_perf_records.json` | ❌ | raw,已被 stdout 派生过,二次喂入是冗余 |
| `kernels/aiv/*.cpp` | ❌ | 不是 agent 可改的 lever |
| `ptoas/*.pto` | ❌ | 过低层 |

**结论**:7 MB raw 中,**真正适合直接送进 agent 上下文的内容压缩后约 1.5 KB**。其余应通过工具按需检索。

### 2.2 已有工具:`tools/perf_analyzer.py`

该脚本(912 行)已经做了大部分**解析**工作:
- 解析 `kernel_config.py` → `func_id_to_name`、`func_id_to_core_type`、`RUNTIME_CONFIG`(`block_dim`、`aicpu_thread_num` 等)
- 解析 `memory_after_AllocateMemoryAddr.txt` → `list[MemoryEntry]`
- 解析 `swimlane_data/perf_swimlane_*.json` → `TaskStats`、`SchedulerStats`、`OrchestratorStats`
- 输出:terminal colored 表格 + structured Markdown 报告

但它**没有**:
- agent 友好的 JSON 摘要
- perf_hints 去重聚合
- bottleneck 因果 + lever + counterfactual 字段
- baseline / delta 对比
- pass dump 的 op-count diff

设计原则之一:**`agent_feedback.py` 站在 `perf_analyzer.py` 已经构造的 `ScopeAnalysis` 上做派生,不重复解析逻辑。**

---

## 3. 设计原则(给 Agent 的 9 条)

| # | 原则 | 例 |
|---|------|----|
| 1 | **Self-contained** | 不需要打开 Perfetto / IDE 就能理解 |
| 2 | **Causal** | 不只"慢",写明"因为 X 所以慢" |
| 3 | **Actionable lever** | 给出可改的 file:line + 改什么 |
| 4 | **Counterfactual** | "改 X → 预计 Tail OH 从 5us 降到 ~2us" |
| 5 | **De-duplicated** | 同一信号合并 + frequency 字段 |
| 6 | **Schemaful (JSON)** | 不要散文 → 不要预设结论 |
| 7 | **Source-linked** | 指向 `file:line:col`,不是 PC/instruction |
| 8 | **Delta-oriented** | 与 baseline 比的 Δ 比绝对值更有指导意义 |
| 9 | **Tiered / Lazy** | 概要 → on-demand 详情 → 工具检索 raw |

> **关键自律**:agent_summary 中**不写自然语言结论**(如 "Key insight: scheduler is the bottleneck")。仅给数字 + ratio + lever 候选,**让 agent 自己组合假设**。这避免预处理器的偏见污染 agent 的推理。

---

## 4. 三层 Feedback 架构

```
┌─────────────────────────────────────────────────────────────────┐
│ Tier 0  agent_summary.json           ~1.5 KB   总是塞进 agent  │
├─────────────────────────────────────────────────────────────────┤
│ Tier 1  agent_summary.detail.json    ~10 KB    agent 主动 fetch │
│         report/perf_hints_aggregated.json                       │
│         report/passes_diff.json                                 │
├─────────────────────────────────────────────────────────────────┤
│ Tier 2  raw artifacts(MB 级)        通过工具 (read_file/grep) │
│         swimlane_data/*  passes_dump/*  kernels/*               │
└─────────────────────────────────────────────────────────────────┘
```

**Tier 0** 永远在 agent 上下文里;**Tier 1** 是 topic-specific 的展开;**Tier 2** 留给具体工具(MCP / file_read / grep)。

---

## 5. `agent_summary.json` 形式化 Schema(Tier 0)

### 5.1 Schema(JSON Schema 风格)

```jsonc
{
  "schema_version": "1.0.0",
  "run_id": "string",                 // 例如 "Qwen3Scope2_20260507_112341"
  "scope": "string",                  // 例如 "Qwen3Scope2"
  "timestamp": "string (ISO-8601)",
  "verdict": {
    "passed": "boolean",
    "error": "string|null",
    "rtol": "number",
    "atol": "number"
  },
  "config": {
    "platform": "string",             // a2a3 | a2a3sim | a5 | a5sim
    "device_id": "integer",
    "block_dim_requested": "integer", // 24 by default
    "block_dim_used": "integer",      // 真正使用的(被 clamp 后)
    "aicpu_thread_num": "integer",
    "scheduler_thread_num": "integer" // = aicpu_thread_num - 1
  },
  "perf": {
    "wall_time_us": "number",
    "task_count": "integer",
    "avg_exec_us": "number",
    "avg_latency_us": "number",
    "exec_to_latency_pct": "number",
    "head_oh_pct": "number",
    "tail_oh_pct": "number",
    "tail_oh_p50_us": "number",
    "tail_oh_p90_us": "number",
    "tail_oh_p99_us": "number"
  },
  "tasks_by_function": [
    {
      "func_id": "integer",
      "name": "string",               // 例 "qk_matmul"
      "core_type": "string",          // "aic" | "aiv"
      "count": "integer",
      "avg_exec_us": "number",
      "avg_latency_us": "number",
      "exec_pct": "number",
      "avg_head_oh_us": "number",
      "avg_tail_oh_us": "number"
    }
  ],
  "scheduler": {
    "thread_count": "integer",
    "phases": {
      "complete_pct": "number",
      "dispatch_pct": "number",
      "scan_pct": "number",
      "idle_pct": "number"
    },
    "tasks_per_loop": ["number"],     // 每 sched 线程的均值
    "fanout_edges_total": "integer",
    "fanin_edges_total": "integer"
  },
  "memory": {
    "max_usage_pct": "number",
    "threshold_pct": "number",        // 例 70
    "all_below_threshold": "boolean",
    "per_kernel": [
      {"kernel": "string", "space": "string",
       "used_kb": "number", "limit_kb": "number", "usage_pct": "number"}
    ]
  },
  "perf_hints": [
    {
      "id": "string",                 // "PH001"
      "kind": "string",               // "TileInnermostDimGranularity"
      "tile_op": "string",            // "load" | "store"
      "innermost_B": "integer",
      "recommended_B": "integer",
      "site": {"file": "string", "line": "integer", "col": "integer"},
      "frequency": "integer",         // 同 site 出现次数(去重前)
      "structural": "boolean|null"    // 已知由 model shape 决定 → true;算法约束 → true;否则 null(agent 可决定)
    }
  ],
  "bottlenecks": [
    {
      "id": "string",                 // BK1, BK2, ...
      "kind": "string",               // 见 §5.2
      "severity": "string",           // "high" | "medium" | "low"
      "metric": { /* 自由结构,数字优先 */ },
      "root_cause": "string",         // 一句话因果(短),纯事实,无主观判断
      "lever": [                       // 候选改动(可为空)
        {
          "option": "string",
          "param_or_site": "string",
          "from": "any",
          "to": "any",
          "predicted": { /* 数字预期 */ },
          "risk": "string",
          "complexity": "string"      // "low" | "medium" | "high"
        }
      ],
      "source": "string|null"         // file:line[-line]
    }
  ],
  "anomalies": [
    {
      "kind": "string",
      "metric": { /* 数字 */ },
      "interpretation": "string",
      "actionable": "boolean"
    }
  ],
  "delta_from_baseline": {            // null 当无 baseline 时
    "baseline_run_id": "string",
    "wall_time_us_delta": "number",
    "wall_time_pct_delta": "number",
    "task_count_delta": "integer",
    "perf_hint_delta": {"new": [...], "removed": [...], "unchanged": "integer"},
    "no_meaningful_change": "boolean"
  } | null,
  "raw_artifacts": {
    "swimlane_merged": "string (relative path)",
    "swimlane_perf_records": "string",
    "passes_dump_dir": "string",
    "kernels_dir": "string",
    "memory_report": "string",
    "perf_hints_log": "string",
    "perf_hints_aggregated": "string"
  }
}
```

### 5.2 Bottleneck `kind` 词表

| kind | 触发条件 | 默认 lever 模板 |
|------|---------|----------------|
| `core_underutilization` | `block_dim_used < block_dim_requested` | 调整 `aicpu_thread_num` 使 `block_dim ÷ (aicpu_thread_num-1) = AICore` |
| `dispatch_overhead` | `tail_oh_pct + head_oh_pct > 25` | 合并相邻 stage / 减少 task 总数 |
| `cache_line_underutilization` | 出现 `innermost_B < 32` 且 `frequency ≥ 16` | 在 algo 允许下 horizontally pack tile |
| `memory_pressure` | `max_usage_pct > threshold_pct` | 调整 tile 形状或减少 buffer |
| `scheduler_starvation` | `idle_pct > 50` | 增加任务并行度或减小 sched 数 |
| `incomplete_perf_collection` | `collected < expected × 0.95` | (anomaly,不是 bottleneck) |

### 5.3 Bottleneck Severity 规则

```
high   : 影响 ≥ 10% 的 wall_time 或可用算力
medium : 5% ~ 10%
low    : < 5%
```

### 5.4 实例:本次 `Qwen3Scope2_20260507_112341` 的 Tier 0

```jsonc
{
  "schema_version": "1.0.0",
  "run_id": "Qwen3Scope2_20260507_112341",
  "scope": "Qwen3Scope2",
  "timestamp": "2026-05-07T11:23:41+08:00",
  "verdict": {"passed": true, "error": null, "rtol": 1e-3, "atol": 1e-3},
  "config": {
    "platform": "a2a3", "device_id": 0,
    "block_dim_requested": 24, "block_dim_used": 18,
    "aicpu_thread_num": 4, "scheduler_thread_num": 3
  },
  "perf": {
    "wall_time_us": 695.6, "task_count": 528,
    "avg_exec_us": 16.64, "avg_latency_us": 23.76, "exec_to_latency_pct": 70.04,
    "head_oh_pct": 8.7, "tail_oh_pct": 21.2,
    "tail_oh_p50_us": 3.7, "tail_oh_p90_us": 11.0, "tail_oh_p99_us": 19.2
  },
  "tasks_by_function": [
    {"func_id": 0, "name": "rope_kv_cache",   "core_type": "aiv", "count":  16,
     "avg_exec_us":  9.85, "avg_latency_us": 16.01, "exec_pct": 61.5,
     "avg_head_oh_us": 0.53, "avg_tail_oh_us": 5.63},
    {"func_id": 1, "name": "qk_matmul",       "core_type": "aic", "count": 128,
     "avg_exec_us": 22.73, "avg_latency_us": 30.83, "exec_pct": 73.7,
     "avg_head_oh_us": 3.59, "avg_tail_oh_us": 4.51},
    {"func_id": 2, "name": "softmax",         "core_type": "aiv", "count": 128,
     "avg_exec_us": 10.91, "avg_latency_us": 17.25, "exec_pct": 63.2,
     "avg_head_oh_us": 0.44, "avg_tail_oh_us": 5.90},
    {"func_id": 3, "name": "sv_matmul",       "core_type": "aic", "count": 128,
     "avg_exec_us": 21.39, "avg_latency_us": 30.64, "exec_pct": 69.8,
     "avg_head_oh_us": 4.05, "avg_tail_oh_us": 5.20},
    {"func_id": 4, "name": "online_softmax",  "core_type": "aiv", "count": 128,
     "avg_exec_us": 12.37, "avg_latency_us": 17.28, "exec_pct": 71.6,
     "avg_head_oh_us": 0.43, "avg_tail_oh_us": 4.48}
  ],
  "scheduler": {
    "thread_count": 3,
    "phases": {"complete_pct": 34.6, "dispatch_pct": 21.6, "scan_pct": 0.0, "idle_pct": 43.8},
    "tasks_per_loop": [0.1, 0.2, 0.4],
    "fanout_edges_total": 0, "fanin_edges_total": 0
  },
  "memory": {
    "max_usage_pct": 25.0, "threshold_pct": 70, "all_below_threshold": true,
    "per_kernel": [
      {"kernel": "qk_matmul",   "space": "Right", "used_kb": 16.0, "limit_kb": 64.0, "usage_pct": 25.0},
      {"kernel": "sv_matmul",   "space": "Right", "used_kb": 16.0, "limit_kb": 64.0, "usage_pct": 25.0}
    ]
  },
  "perf_hints": [
    {"id": "PH001", "kind": "TileInnermostDimGranularity", "tile_op": "load",
     "innermost_B": 4,   "recommended_B": 512,
     "site": {"file": "qwen3_32b_decode_scope2.py", "line": 210, "col": 20},
     "frequency": 4,  "structural": true},
    {"id": "PH001", "kind": "TileInnermostDimGranularity", "tile_op": "store",
     "innermost_B": 4,   "recommended_B": 512,
     "site": {"file": "qwen3_32b_decode_scope2.py", "line": 188, "col": 41},
     "frequency": 1,  "structural": true},
    {"id": "PH001", "kind": "TileInnermostDimGranularity", "tile_op": "load",
     "innermost_B": 256, "recommended_B": 512,
     "site": {"file": "qwen3_32b_decode_scope2.py", "line": 79,  "col": 16},
     "frequency": 6,  "structural": true}
  ],
  "bottlenecks": [
    {
      "id": "BK1", "kind": "core_underutilization", "severity": "high",
      "metric": {"used": 18, "available": 20, "loss_pct": 10},
      "root_cause": "block_dim=24 clamped to floor(20/3)*3=18 because scheduler_thread_num=3",
      "lever": [
        {"option": "increase_aicpu_thread_num",
         "param_or_site": "aicpu_thread_num",
         "from": 4, "to": 5,
         "predicted": {"block_dim_used": 20, "loss_pct": 0},
         "risk": "may oversubscribe AICPU; bench required",
         "complexity": "low"}
      ],
      "source": null
    },
    {
      "id": "BK2", "kind": "dispatch_overhead", "severity": "high",
      "metric": {"head_oh_pct": 8.7, "tail_oh_pct": 21.2,
                 "task_count": 528, "avg_exec_us": 16.64},
      "root_cause": "528 short tasks; tail dominated by poll-based completion (~6.9 sched loops to detect)",
      "lever": [
        {"option": "fuse_qk_softmax_sv_into_one_task",
         "param_or_site": "qwen3_32b_decode_scope2.py:157-200",
         "from": null, "to": null,
         "predicted": {"task_count": 176, "tail_oh_pct": "<10"},
         "risk": "increases per-task SRAM footprint",
         "complexity": "high"},
        {"option": "increase_Q_HEAD_BATCH",
         "param_or_site": "Q_HEAD_BATCH",
         "from": 8, "to": 16,
         "predicted": {"task_count": 264},
         "risk": "may exceed cube fractal budget",
         "complexity": "medium"}
      ],
      "source": "qwen3_32b_decode_scope2.py:157,170,192"
    },
    {
      "id": "BK3", "kind": "cache_line_underutilization", "severity": "medium",
      "metric": {"cache_line_B": 512, "innermost_B": 4, "utilization": 0.008,
                 "frequency": 128},
      "root_cause": "online softmax mi/li are single-column FP32 stores",
      "lever": [
        {"option": "horizontally_pack_mi_li_across_sb_blocks",
         "param_or_site": "qwen3_32b_decode_scope2.py:188-189",
         "from": null, "to": null,
         "predicted": {"innermost_B": 128, "utilization": 0.25},
         "risk": "stage-5 accumulation rewrite; correctness-sensitive",
         "complexity": "high"}
      ],
      "source": "qwen3_32b_decode_scope2.py:188-189"
    }
  ],
  "anomalies": [
    {"kind": "perf_record_count_mismatch",
     "metric": {"reported_by_aicpu": 657, "collected": 528, "loss_pct": 19.6},
     "interpretation": "AICPU sub-tasks (sched/orch) excluded; kernel-level stats unaffected",
     "actionable": false}
  ],
  "delta_from_baseline": null,
  "raw_artifacts": {
    "swimlane_merged":      "swimlane_data/merged_swimlane_20260507_112348.json",
    "swimlane_perf_records":"swimlane_data/l2_perf_records_20260507_112348.json",
    "passes_dump_dir":      "passes_dump/",
    "kernels_dir":          "kernels/",
    "memory_report":        "report/memory_after_AllocateMemoryAddr.txt",
    "perf_hints_log":       "report/perf_hints.log",
    "perf_hints_aggregated":"report/perf_hints_aggregated.json"
  }
}
```

---

## 6. 派生工件设计(Tier 1)

### 6.1 `report/perf_hints_aggregated.json`

**问题**:`perf_hints.log` 同 `(file, line, hint_id)` 重复 6×,token 浪费 + 误导严重程度。

**输出**:每条 hint 一行,带 `frequency` 与 `structural` 标记。

```json
[
  {"id": "PH001", "kind": "TileInnermostDimGranularity", "tile_op": "load",
   "innermost_B": 256, "recommended_B": 512,
   "site": {"file": "qwen3_32b_decode_scope2.py", "line": 79, "col": 16},
   "frequency": 6, "structural": true,
   "explanation": "head_dim×2B=256B fixed by model shape"},
  {"id": "PH001", "kind": "TileInnermostDimGranularity", "tile_op": "store",
   "innermost_B": 4, "recommended_B": 512,
   "site": {"file": "qwen3_32b_decode_scope2.py", "line": 188, "col": 41},
   "frequency": 1, "structural": true,
   "explanation": "online softmax mi/li single-column reduction"}
]
```

**`structural` 判定规则**:

```
若 innermost_B ∈ {2, 4, 8} 且 site 在 reduction 缩减点 → structural=true
若 innermost_B ∈ {head_dim_B, half_dim_B, SEQ_TILE_B} → structural=true(由模型形状决定)
否则 → structural=null(让 agent 决定)
```

可在 `tools/agent_feedback.yaml` 中维护**白名单**进一步标注已知 structural site,避免 agent 反复尝试一些不可能的 lever。

### 6.2 `report/passes_diff.json`

**问题**:31 个 pass dump 共 5926 行 IR,全喂爆,选 1 个又看不全。

**输出**:每对相邻 pass 的"net change"。

```json
[
  {"from": "00_frontend",                "to": "01_after_UnrollLoops",
   "ops_added": {"pl.range": -2, "pl.parallel": 0}, "lines_delta": +14},
  {"from": "07_after_SplitChunkedLoops", "to": "08_after_InterchangeChunkLoops",
   "ops_added": {"pl.tile.load": +5, "pl.tile.store": +3}, "lines_delta": +28},
  ...
]
```

让 agent 一眼看出"哪个 pass 引入了大量 tile op"——必要时再用 `read_file` 工具去拉对应的 pass dump。

### 6.3 `swimlane_data/critical_path.json`(可选)

从 `merged_swimlane_*.json` 派生**关键路径**:从最早 dispatch 到最晚 finish 的端到端任务链,只保留 ≤ 20 条记录。

```json
{
  "wall_time_us": 695.6,
  "critical_chain": [
    {"task_id": 1, "func": "rope_kv_cache", "start": 39.4, "end": 49.3, "core": 22},
    {"task_id": 2, "func": "qk_matmul",     "start": 53.8, "end": 76.5, "core": 5,
     "head_oh_us": 4.5, "tail_oh_us": 4.0,
     "blocked_by": [1]},
    ...
  ]
}
```

> **注意**:此项工作量较大(需要正确的 task 依赖图),**Phase 2 再实现**。Phase 1 直接把 raw `merged_swimlane_*.json` 留给 Tier 2 工具检索。

---

## 7. 代码实现方案

### 7.1 文件布局

```
pypto-lib/
├── tools/
│   ├── perf_analyzer.py        (现有,不动)
│   ├── agent_feedback.py       (新增,~400 行)
│   └── agent_feedback.yaml     (新增,structural site 白名单)
├── docs/
│   └── agent_feedback_design.md (本文)
└── golden/
    └── runner.py               (可选改动,~5 行,自动调用)
```

### 7.2 `tools/agent_feedback.py` 模块结构

```python
"""
Agent-friendly feedback emitter for pypto-lib build_output.

Reads a single build_output/<run> dir, derives a compact JSON summary
(~1.5 KB) suitable for direct LLM agent consumption, and optional
detail/aggregation files for on-demand fetch.

Reuses tools.perf_analyzer.ScopeAnalysis for parsing; this module is
purely a derivation/emission layer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Reuse existing parsing layer
sys.path.insert(0, str(Path(__file__).parent))
from perf_analyzer import (  # noqa: E402
    ScopeAnalysis, analyze_scope, discover_scopes,
)

SCHEMA_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------

PERF_HINT_RE = re.compile(
    r"\[perf_hint\s+(?P<id>PH\d+)\]\s+(?P<kind>\w+):\s+tile\.(?P<op>\w+)\s+"
    r"has\s+innermost\s+dim\s+=\s+(?P<innermost>\d+)B;\s+recommended\s+>=\s+"
    r"(?P<recommended>\d+)B[^/]*at\s+(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+)"
)


def aggregate_perf_hints(perf_hints_log: Path,
                         structural_sites: dict[tuple, bool]) -> list[dict]:
    """Aggregate perf_hints.log: dedup by (file, line, hint_id, op, innermost),
    add frequency, classify structural."""
    if not perf_hints_log.exists():
        return []

    counts: dict[tuple, int] = defaultdict(int)
    meta: dict[tuple, dict] = {}

    for raw in perf_hints_log.read_text().splitlines():
        m = PERF_HINT_RE.search(raw)
        if not m:
            continue
        key = (m["file"], int(m["line"]), m["id"], m["op"], int(m["innermost"]))
        counts[key] += 1
        meta[key] = {
            "id": m["id"],
            "kind": m["kind"],
            "tile_op": m["op"],
            "innermost_B": int(m["innermost"]),
            "recommended_B": int(m["recommended"]),
            "site": {"file": Path(m["file"]).name,
                     "line": int(m["line"]), "col": int(m["col"])},
        }

    out = []
    for key, n in counts.items():
        d = dict(meta[key])
        d["frequency"] = n
        d["structural"] = structural_sites.get((d["site"]["file"], d["site"]["line"]),
                                               _classify_structural(d["innermost_B"]))
        out.append(d)
    out.sort(key=lambda x: (x["site"]["file"], x["site"]["line"]))
    return out


def _classify_structural(innermost_B: int) -> bool | None:
    """Heuristic structural classifier (model shape vs algorithm constants)."""
    # Reduction columns (FP32 single-element)
    if innermost_B in (2, 4, 8):
        return True
    # Common shape-dictated sizes: head_dim*2B=256, half_dim*2B=128, SEQ_TILE*2B=128
    if innermost_B in (128, 256):
        return True
    return None


# ---------------------------------------------------------------------------
# Bottleneck inference
# ---------------------------------------------------------------------------

@dataclass
class Bottleneck:
    id: str
    kind: str
    severity: str
    metric: dict[str, Any]
    root_cause: str
    lever: list[dict[str, Any]]
    source: str | None


def infer_bottlenecks(scope: ScopeAnalysis,
                      hw_max_aicore: int,
                      perf_hints: list[dict]) -> list[Bottleneck]:
    out: list[Bottleneck] = []
    bk_id = lambda i: f"BK{i+1}"

    # BK1: core underutilization
    sched_threads = max(scope.aicpu_thread_num - 1, 1)
    requested = scope.block_dim
    used = (hw_max_aicore // sched_threads) * sched_threads
    if used < requested and used < hw_max_aicore:
        loss_pct = round((hw_max_aicore - used) / hw_max_aicore * 100, 1)
        sev = "high" if loss_pct >= 10 else ("medium" if loss_pct >= 5 else "low")
        out.append(Bottleneck(
            id=bk_id(len(out)),
            kind="core_underutilization",
            severity=sev,
            metric={"used": used, "available": hw_max_aicore, "loss_pct": loss_pct},
            root_cause=(f"block_dim={requested} clamped to floor({hw_max_aicore}/"
                        f"{sched_threads})*{sched_threads}={used} "
                        f"because scheduler_thread_num={sched_threads}"),
            lever=_lever_for_aicpu_thread_num(scope, hw_max_aicore),
            source=None,
        ))

    # BK2: dispatch overhead
    head_pct = _per_task_pct(scope, "head_oh")
    tail_pct = _per_task_pct(scope, "tail_oh")
    if head_pct + tail_pct > 25:
        sev = "high" if head_pct + tail_pct > 40 else "medium"
        out.append(Bottleneck(
            id=bk_id(len(out)),
            kind="dispatch_overhead",
            severity=sev,
            metric={"head_oh_pct": head_pct, "tail_oh_pct": tail_pct,
                    "task_count": scope.task_count,
                    "avg_exec_us": _global_avg_exec_us(scope)},
            root_cause=(f"{scope.task_count} short tasks; "
                        "tail dominated by poll-based completion"),
            lever=[],   # populated by template if scope name matches known patterns
            source=None,
        ))

    # BK3: cache line underutilization (from aggregated hints)
    sub32 = [h for h in perf_hints if h["innermost_B"] < 32 and h["frequency"] >= 16]
    for h in sub32:
        out.append(Bottleneck(
            id=bk_id(len(out)),
            kind="cache_line_underutilization",
            severity="medium",
            metric={"cache_line_B": 512, "innermost_B": h["innermost_B"],
                    "utilization": round(h["innermost_B"] / 512, 4),
                    "frequency": h["frequency"]},
            root_cause=f"{h['tile_op']} at {h['site']['file']}:{h['site']['line']}",
            lever=[],
            source=f"{h['site']['file']}:{h['site']['line']}",
        ))

    return out


def _per_task_pct(scope: ScopeAnalysis, kind: str) -> float:
    total_lat = sum(s.total_latency_us for s in scope.task_stats.values())
    total_oh = sum(getattr(s, f"total_{kind}_us") for s in scope.task_stats.values())
    return round(total_oh / total_lat * 100, 1) if total_lat else 0.0


def _global_avg_exec_us(scope: ScopeAnalysis) -> float:
    total = sum(s.total_exec_us for s in scope.task_stats.values())
    cnt = sum(s.count for s in scope.task_stats.values())
    return round(total / cnt, 2) if cnt else 0.0


def _lever_for_aicpu_thread_num(scope: ScopeAnalysis, hw_max: int) -> list[dict]:
    out = []
    for n in (5, 6, 8, 10):
        sched = max(n - 1, 1)
        used = (hw_max // sched) * sched
        if used > _current_used(scope, hw_max):
            out.append({
                "option": "increase_aicpu_thread_num",
                "param_or_site": "aicpu_thread_num",
                "from": scope.aicpu_thread_num, "to": n,
                "predicted": {"block_dim_used": used,
                              "loss_pct": round((hw_max - used) / hw_max * 100, 1)},
                "risk": "may oversubscribe AICPU; bench required",
                "complexity": "low",
            })
    return out


def _current_used(scope: ScopeAnalysis, hw_max: int) -> int:
    sched = max(scope.aicpu_thread_num - 1, 1)
    return (hw_max // sched) * sched


# ---------------------------------------------------------------------------
# Delta vs baseline
# ---------------------------------------------------------------------------

def compute_delta(curr: dict, baseline_path: Path | None) -> dict | None:
    if baseline_path is None or not baseline_path.exists():
        return None
    baseline = json.loads(baseline_path.read_text())
    wt_delta = curr["perf"]["wall_time_us"] - baseline["perf"]["wall_time_us"]
    wt_pct = wt_delta / baseline["perf"]["wall_time_us"] * 100
    curr_hints = {(h["site"]["file"], h["site"]["line"], h["id"]): h
                  for h in curr.get("perf_hints", [])}
    base_hints = {(h["site"]["file"], h["site"]["line"], h["id"]): h
                  for h in baseline.get("perf_hints", [])}
    new = [k for k in curr_hints if k not in base_hints]
    removed = [k for k in base_hints if k not in curr_hints]
    return {
        "baseline_run_id": baseline["run_id"],
        "wall_time_us_delta": round(wt_delta, 2),
        "wall_time_pct_delta": round(wt_pct, 2),
        "task_count_delta": curr["perf"]["task_count"] - baseline["perf"]["task_count"],
        "perf_hint_delta": {"new": new, "removed": removed,
                            "unchanged": len(curr_hints) - len(new)},
        "no_meaningful_change": abs(wt_pct) < 1.0 and not new and not removed,
    }


# ---------------------------------------------------------------------------
# Top-level emit
# ---------------------------------------------------------------------------

def emit_agent_summary(scope_dir: Path,
                       hw_max_aicore: int = 20,
                       baseline: Path | None = None,
                       structural_yaml: Path | None = None) -> dict:
    sa = analyze_scope(scope_dir)
    if sa is None:
        raise RuntimeError(f"Cannot parse scope dir: {scope_dir}")

    # Aggregate perf hints
    structural_sites = _load_structural_yaml(structural_yaml)
    perf_hints = aggregate_perf_hints(scope_dir / "report" / "perf_hints.log",
                                      structural_sites)
    # Persist Tier 1
    (scope_dir / "report" / "perf_hints_aggregated.json").write_text(
        json.dumps(perf_hints, indent=2))

    # Build summary
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": sa.name,
        "scope": _scope_root_name(sa.name),
        "timestamp": _extract_timestamp(sa.name),
        "verdict": _verdict(sa),
        "config": _config(sa, hw_max_aicore),
        "perf": _perf(sa),
        "tasks_by_function": _tasks(sa),
        "scheduler": _sched(sa),
        "memory": _memory(sa),
        "perf_hints": perf_hints,
        "bottlenecks": [asdict(b) for b in infer_bottlenecks(sa, hw_max_aicore, perf_hints)],
        "anomalies": _anomalies(sa),
        "delta_from_baseline": None,    # filled below
        "raw_artifacts": _raw_artifacts(sa),
    }
    summary["delta_from_baseline"] = compute_delta(summary, baseline)

    out = scope_dir / "agent_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    return summary


# helper stubs (impl details elided for brevity in this design doc)
def _load_structural_yaml(p): ...
def _scope_root_name(name): ...
def _extract_timestamp(name): ...
def _verdict(sa): ...
def _config(sa, hw_max): ...
def _perf(sa): ...
def _tasks(sa): ...
def _sched(sa): ...
def _memory(sa): ...
def _anomalies(sa): ...
def _raw_artifacts(sa): ...


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Emit agent-friendly summary.")
    ap.add_argument("path", help="build_output dir or single scope dir")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="Baseline agent_summary.json for delta computation")
    ap.add_argument("--hw-max-aicore", type=int, default=20,
                    help="HW AICore count (auto-detect if omitted)")
    ap.add_argument("--structural", type=Path,
                    default=Path(__file__).with_suffix(".yaml"),
                    help="YAML with known-structural perf_hint sites")
    args = ap.parse_args()

    scopes = discover_scopes(Path(args.path))
    if not scopes:
        print(f"No scope directories found under {args.path}", file=sys.stderr)
        sys.exit(1)

    for s in scopes:
        summary = emit_agent_summary(s, args.hw_max_aicore, args.baseline,
                                     args.structural)
        out = s / "agent_summary.json"
        print(f"[OK] {out}  ({_count_bytes(out)} bytes)")


def _count_bytes(p: Path) -> int:
    return p.stat().st_size


if __name__ == "__main__":
    main()
```

> **实现说明**:上面省略了 `_verdict / _config / _perf / _tasks / _sched / _memory / _anomalies / _raw_artifacts` 这些 helper 的具体内容(都是把 `ScopeAnalysis` 的字段 reshape 成 schema 的子结构,直观)。完整实现约 400 行。

### 7.3 `tools/agent_feedback.yaml`(structural 白名单)

```yaml
# 已知由模型形状/算法决定、无法从用户代码侧修复的 perf_hint 站点
# (file, line) 组合;agent 看到这些不应再尝试 cache-line-utilization 类 lever
known_structural:
  - {file: "qwen3_32b_decode_scope2.py", line: 79,  reason: "head_dim*2B=256 fixed by model"}
  - {file: "qwen3_32b_decode_scope2.py", line: 158, reason: "K tile shape"}
  - {file: "qwen3_32b_decode_scope2.py", line: 188, reason: "online softmax mi/li single-col"}
  - {file: "qwen3_32b_decode_scope2.py", line: 189, reason: "online softmax mi/li single-col"}
  - {file: "qwen3_32b_decode_scope2.py", line: 210, reason: "online softmax mi/li read"}
```

### 7.4 集成钩子(可选)

#### 选项 A — 在 `golden/runner.py` 末尾自动调用(推荐)

```diff
--- a/golden/runner.py
+++ b/golden/runner.py
@@
     # Validate
     with _stage("validate"):
         try:
             validate_golden(device_outputs, golden_outputs, rtol=config.rtol, atol=config.atol)
         except AssertionError as e:
             return _fail(str(e))

+    # Optional agent-feedback emission (best-effort, non-fatal)
+    if config.runtime.get("runtime_profiling"):
+        try:
+            from tools.agent_feedback import emit_agent_summary
+            emit_agent_summary(work_dir)
+        except Exception as exc:
+            print(f"[RUN]   agent_summary skipped: {exc}", flush=True)
+
     total = time.time() - start
     print(f"[RUN] PASS ({total:.2f}s)", flush=True)
```

**优点**:每次 `--runtime-profiling` 的真机运行都自动生成 agent 摘要,零额外命令。
**缺点**:增加 ~50ms 后处理延迟;失败时不阻断主流程(已用 try/except 包裹)。

#### 选项 B — 独立脚本

```bash
python tools/agent_feedback.py build_output/Qwen3Scope2_20260507_112341
# 输出:build_output/Qwen3Scope2_20260507_112341/agent_summary.json (~1.5 KB)

python tools/agent_feedback.py build_output/ \
    --baseline build_output/Qwen3Scope2_baseline/agent_summary.json
# 批量处理 + delta
```

**推荐顺序**:**先实现选项 B**(独立可测),稳定后再考虑选项 A 的自动钩子。

### 7.5 与 `examples/models/qwen-opt/run_comparison.sh` 的整合

现有 `run_comparison.sh` 跑完 baseline + opt 后只对比 PASS/FAIL。可改为:

```bash
# baseline run (产生 baseline 的 agent_summary.json)
python examples/models/qwen3/32b/qwen3_32b_decode_scope2.py -p a2a3 -d 0 --runtime-profiling
BASE_DIR=$(ls -td build_output/Qwen3Scope2_*/ | head -1)
python tools/agent_feedback.py "$BASE_DIR"

# optimized run + delta vs baseline
python examples/models/qwen-opt/qwen3_32b_decode_scope2-opt.py -p a2a3 -d 0 \
    --runtime-profiling --golden-data "$BASE_DIR/data"
OPT_DIR=$(ls -td build_output/Qwen3Scope2_*/ | head -1)
python tools/agent_feedback.py "$OPT_DIR" \
    --baseline "$BASE_DIR/agent_summary.json"

# 现在 OPT_DIR/agent_summary.json 含 delta_from_baseline 字段
# 任何 LLM agent 直接读这个文件,~2 KB,即可作出优化判断
```

---

## 8. baseline / delta 工作流

```
┌─────────────────────┐         ┌─────────────────────┐
│  baseline run       │         │  optimized run      │
│  agent_summary.json │ ──────► │  --baseline=...     │
│  (无 delta 字段)    │         │  agent_summary.json │
└─────────────────────┘         │  (delta 字段已填)   │
                                └─────────────────────┘
```

**Delta 字段规则**:
- `wall_time_us_delta`、`wall_time_pct_delta`:绝对差与百分比
- `task_count_delta`:任务数变化
- `perf_hint_delta.new` / `removed`:hint 站点变化(隐含可能的算子重排)
- `no_meaningful_change`:`|wall_time_pct_delta| < 1%` 且无 hint 变化 → agent 应**直接返回"无变化"**,无需深入分析

> 这一字段是节省 agent 上下文的关键 short-circuit。

---

## 9. 落地 Roadmap

### Phase 1(MVP,~1–2 天)
- [ ] `tools/agent_feedback.py` 主体(emit Tier 0 + perf_hints 聚合)
- [ ] `tools/agent_feedback.yaml` 初始白名单(覆盖 `qwen3/32b/*-scope*.py`)
- [ ] CLI:`python tools/agent_feedback.py <build_output>`
- [ ] 单测:用 `Qwen3Scope2_20260507_112341` 作为 fixture,对照本文 §5.4 实例

### Phase 2(增强,~3 天)
- [ ] delta-vs-baseline 实现 + 对接 `run_comparison.sh`
- [ ] `report/passes_diff.json` 生成
- [ ] `golden/runner.py` 钩子(选项 A)
- [ ] 在 `docs/swimlane_guide.md` 末尾加一个交叉链接到本文

### Phase 3(可选,工作量较大)
- [ ] `swimlane_data/critical_path.json`(从 13523 events 提取关键路径)
- [ ] `bottlenecks` 的更多 `kind`(memory_pressure、scheduler_starvation 等)
- [ ] 对 `passes_dump/*.py` 做 op-count diff(需 IR 解析)

---

## 10. 设计权衡 / FAQ

### Q1:为什么不直接把现有 stdout 表喂给 agent?

stdout 含**散文式 "Key insight: ..."**,会把 agent 锚定到 *作者* 的诊断方向,丧失独立判断。结构化 JSON 故意只给数字,**让 agent 自己组合假设**——这是设计原则 #6 的核心。

### Q2:为什么不做更多预处理(直接告诉 agent "这就是瓶颈")?

预处理是双刃剑:
- 隐藏长尾(只给 mean → P99 异常丢失)
- 预处理器的偏见污染 agent
- 不同 agent 任务需要不同 view(精度调试 vs 性能优化)

折中:**Tier 0 给数字 + 候选 lever,Tier 2 留 raw**。lever 是"agent 可参考的候选",不是"必须采纳的结论"。

### Q3:`structural=true/null` 的判定不够精确?

是的。当前是启发式 + 白名单。准确的方案是让编译器在 emit `perf_hint` 时附带 `is_structural` 元数据(由 IR 形状传播分析得出),`agent_feedback.py` 直接读。这是**编译器侧**的改进,本设计先用启发式。

### Q4:1.5 KB 真够吗?

`Qwen3Scope2` 实例确认 ~1.5 KB。如果 scope 更复杂(例如 prefill 全图,task_count > 5000),`tasks_by_function` + `perf_hints` 会增长。控制方法:
- `tasks_by_function` 只保留 top-N(N=10)
- `perf_hints` 只保留 `frequency ≥ 4` 的
- 其余移入 `agent_summary.detail.json`(Tier 1)

### Q5:跟 MCP / function-calling 怎么配合?

Tier 2 的 raw 不直接进 prompt,而是通过 agent 的工具调用(`read_file` / `grep_search` / 自定义 MCP `pypto.swimlane.query`)按需检索。设计文档里的 `raw_artifacts` 字段就是给 agent 提供"我可以再去查这些"的入口。

### Q6:这套方案对人类还有价值吗?

有。`agent_summary.json` 也是**对 human 友好的**简明摘要:打开就能看 verdict + top-3 bottlenecks + delta,比翻 stdout 快。但泳道图、Perfetto、`perf_analyzer.py` Markdown 报告**仍然保留**——人类需要密集可视化。

---

## 11. 一句话总结

> **泳道图把数据压成"人类秒读的图像"。Agent 的对应物不是"更详细的图",而是 "数字 + 因果 + 可执行 lever + 预期收益" 的紧凑结构化文本,加上按需检索的工具入口。** 本方案在不破坏现有人类视角的前提下,以 ~400 行 Python 实现这个补丁。

---

## 附录 A:术语表

| 术语 | 定义 |
|------|------|
| **Tier 0/1/2** | 三层 feedback 架构;Tier 0 永远在 agent context,Tier 1 按需 fetch,Tier 2 工具检索 raw |
| **lever** | "可改的代码/参数位置 + 改成什么 + 预期收益",agent 可直接行动的最小单元 |
| **counterfactual** | "如果改 X,预计 Y 会发生" — 给 agent 提供决策依据 |
| **structural hint** | 由模型形状或算法结构决定、无法从用户代码侧修复的 perf_hint(eg. head_dim×2B=256) |
| **delta-from-baseline** | 与上一次 baseline 运行的差值,`no_meaningful_change=true` 时 agent 可早退 |

## 附录 B:相关文档

- `docs/swimlane_guide.md` — 泳道图(人类视角)使用指南
- `docs/performance_analysis_guide.md` — 性能分析整体方法论
- `docs/baseline_selection_guide.md` — baseline 选择与对比工作流
- `tools/perf_analyzer.py` — 现有解析层(本方案复用其 `ScopeAnalysis`)
