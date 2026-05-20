# Baseline 选择指南

本文档说明如何在 pypto-lib 中定位并建立优化基线（baseline）版本，用于衡量后续
所有优化的收益。

---

## 1. 核心概念

**Baseline** = 模型最初能 E2E 跑通的版本，即未经任何手动优化的首版代码。
所有后续优化的正确性和性能收益均以此版本为参照基准。

**为什么选最初版本**：项目目标是为任意模型提供自动化优化，开发者不一定了解
如何手动优化。未优化的首版代码代表了优化流水线介入前开发者所能产出的基本水平。

---

## 2. 代码依赖关系与 Baseline 策略

### 2.1 两类模型代码

| 位置 | 内容 | Baseline 粒度 |
|------|------|--------------|
| `llm/examples/` | E2E 推理脚本（加载模型 → 生成文本） | **按目录**（模型作为完整单元一次性引入） |
| `examples/models/` | 单层 / 单 scope 的 kernel 示例 | **按文件**（各 scope 跨多个 PR 逐步新增） |

### 2.2 仓库内依赖关系

`examples/models/` 下的代码**并非自包含**，存在以下依赖：

| 依赖 | 来源 | 说明 |
|------|------|------|
| `pypto.language` | 外部安装的 `pypto` 包 | 核心编程框架，非本仓库代码 |
| `golden` 模块 | 本仓库 `golden/` 目录 | `TensorSpec`、`RunConfig`、`run()` — 编译/执行/验证流程 |
| 文件间交叉引用 | 同目录其他文件 | 如 DS V4 indexer 引用 compressor 的 golden 函数 |

**关键影响**：不能只 checkout 单个文件就运行。`golden/` 模块的 API 随仓库
演进（如 `run()` 的参数签名、`TensorSpec` 的字段），**必须与 example 文件
所在的 commit 版本保持一致**。

### 2.3 最保险的 Baseline 策略：整仓库 Checkout

为确保 baseline 代码能实际运行，**最保险的做法是将整个仓库恢复到目标 commit
的完整状态**，而非单独 checkout 某个文件。

```bash
# 保存当前工作状态
git stash

# 将整个仓库切换到 baseline commit
git checkout <baseline-commit>

# 运行 baseline（此时 golden/ 模块、pyproject.toml 等全部是匹配版本）
python examples/models/qwen3/32b/qwen3_32b_decode_scope3.py -p a2a3sim

# 运行完毕后切回当前分支
git checkout main
git stash pop
```

---

## 3. Baseline Commit 分组与共用

### 3.1 核心发现：多个文件首次出现在同一 commit

通过对 `upstream/main` 全量排查，发现许多 baseline 文件的**首次出现聚集在
少量 commit 上**。这意味着同一个 commit 可以作为多个文件的共用 baseline，
只需 checkout 一次即可采集所有相关文件的 baseline 数据。

### 3.2 完整 Baseline Commit 分组表

以下按时间顺序列出所有需关注的 baseline commit，及其包含的文件：

#### Group 1: `fac9a54` — 2026-04-21 (#145)

> CI: add daily CI and mark draft examples

| 文件 | 模型 |
|------|------|
| `examples/models/kimi/kimi_k2_decode_draft.py` | Kimi K2 |
| `examples/models/milm/milm_decode_draft.py` | MILM |

#### Group 2: `ef18594` — 2026-04-23 (#152)

> Add Qwen3-14B decode scopes and update .gitignore

| 文件 | 模型 |
|------|------|
| `examples/models/qwen3/14b/qwen3_14b_decode.py` | Qwen3-14B 完整层 |
| `examples/models/qwen3/14b/qwen3_14b_decode_scope1.py` | Qwen3-14B scope1 |
| `examples/models/qwen3/14b/qwen3_14b_decode_scope2.py` | Qwen3-14B scope2 |
| `examples/models/qwen3/14b/qwen3_14b_decode_scope3.py` | Qwen3-14B scope3 |
| `examples/models/qwen3/14b/qwen3_14b_prefill.py` | Qwen3-14B prefill |

#### Group 3: `7e887c6` — 2026-04-23 (#160)

> Add PyPTO LLM inference scaffold

| 文件 | 模型 |
|------|------|
| `llm/examples/qwen3_14b_cpu_generate.py` | Qwen3-14B E2E (CPU) |
| `llm/examples/qwen3_14b_npu_generate.py` | Qwen3-14B E2E (NPU) |

#### Group 4: `517a8dc` — 2026-04-23 (#163)

> Adapt Qwen3-32B decode to pl.pipeline; tidy example boilerplate

| 文件 | 模型 |
|------|------|
| `examples/models/qwen3/32b/qwen3_32b_decode.py` | Qwen3-32B 完整层 |
| `examples/models/qwen3/32b/qwen3_32b_decode_scope1.py` | Qwen3-32B scope1 |
| `examples/models/qwen3/32b/qwen3_32b_decode_scope2.py` | Qwen3-32B scope2 |
| `examples/models/qwen3/32b/qwen3_32b_decode_scope3.py` | Qwen3-32B scope3 |
| `examples/models/qwen3/32b/qwen3_32b_decode_tile.py` | Qwen3-32B tile 版本 |
| `examples/models/qwen3/32b/qwen3_32b_prefill.py` | Qwen3-32B prefill |
| `examples/models/qwen3/32b/qwen3_32b_prefill_scope1.py` | Qwen3-32B prefill scope1 |
| `examples/models/qwen3/32b/qwen3_32b_prefill_scope2.py` | Qwen3-32B prefill scope2 |
| `examples/models/qwen3/32b/qwen3_32b_prefill_scope3.py` | Qwen3-32B prefill scope3 |
| `examples/models/qwen3/32b/qwen3_32b_training_draft.py` | Qwen3-32B training draft |
| `examples/models/qwen3/32b/qwen3-32b_draft.py` | Qwen3-32B draft |

> **注意**：Qwen3-32B 文件在 #163 之前就存在于旧路径
> `examples/models/qwen3/qwen3_32b_*.py`，但 #163 将它们移动到
> `examples/models/qwen3/32b/` 并同步升级了 `golden` API 调用方式。
> 因此 **`517a8dc` 是当前路径下的首次出现**，也是最保险的 baseline commit。

#### Group 5: `70b4e31` — 2026-04-27 (#186)

> Refactor: reorganize DeepSeek examples and add V4 decode kernels

| 文件 | 模型 |
|------|------|
| `examples/models/deepseek/v3_2/deepseek_v3_2_decode_back.py` | DS V3.2 decode back |
| `examples/models/deepseek/v3_2/deepseek_v3_2_decode_front_scope1.py` | DS V3.2 front scope1 |
| `examples/models/deepseek/v3_2/deepseek_v3_2_decode_front_scope123.py` | DS V3.2 front scope123 |
| `examples/models/deepseek/v3_2/deepseek_v3_2_decode_front_scope2.py` | DS V3.2 front scope2 |
| `examples/models/deepseek/v3_2/deepseek_v3_2_decode_front_scope3.py` | DS V3.2 front scope3 |
| `examples/models/deepseek/v3_2/deepseek_v3_2_decode_front_scope4.py` | DS V3.2 front scope4 |
| `examples/models/deepseek/v3_2/deepseek_v3_2_prefill_back.py` | DS V3.2 prefill back |
| `examples/models/deepseek/v3_2/deepseek_v3_2_prefill_front_draft.py` | DS V3.2 prefill front |
| `examples/models/deepseek/v4/deepseek_v4_decode_attention_draft.py` | DS V4 attention |
| `examples/models/deepseek/v4/deepseek_v4_decode_compressor_draft.py` | DS V4 compressor |
| `examples/models/deepseek/v4/deepseek_v4_decode_indexer_draft.py` | DS V4 indexer |
| `examples/models/deepseek/v4/deepseek_v4_decode_o_proj_draft.py` | DS V4 output proj |

#### Group 6: `1bceb1c` — 2026-04-27 (#187)

> Add DeepSeek V3.2 decode front kernel

| 文件 | 模型 |
|------|------|
| `examples/models/deepseek/v3_2/deepseek_v3_2_decode_front.py` | DS V3.2 decode front |

#### Group 7: `f7110d0` — 2026-04-28 (#194)

> Refactor: split DeepSeek-V4 decode MoE, add hc_post and single-layer doc

| 文件 | 模型 |
|------|------|
| `examples/models/deepseek/v4/deepseek_v4_decode_hc_post_draft.py` | DS V4 HC post |
| `examples/models/deepseek/v4/deepseek_v4_decode_moe_expert_draft.py` | DS V4 MoE expert |
| `examples/models/deepseek/v4/deepseek_v4_decode_moe_router_draft.py` | DS V4 MoE router |
| `examples/models/deepseek/v4/deepseek_v4_decode_single_layer.md` | DS V4 single layer doc |

#### Group 8: `1c800f1` — 2026-04-29 (#201)

> Refactor: align DSv4 decode attention drafts with Attention.forward

| 文件 | 模型 |
|------|------|
| `examples/models/deepseek/v4/deepseek_v4_decode_qkv_proj_rope_draft.py` | DS V4 QKV proj+RoPE |
| `examples/models/deepseek/v4/deepseek_v4_decode_sparse_attn_draft.py` | DS V4 sparse attention |

#### Group 9: `d8cd44c` — 2026-04-29 (#202)

> Add DeepSeek-V4 HC pre decode example

| 文件 | 模型 |
|------|------|
| `examples/models/deepseek/v4/deepseek_v4_decode_hc_pre.py` | DS V4 HC pre |

### 3.3 分组汇总

| Group | Commit | 日期 | 模型 | 文件数 |
|-------|--------|------|------|-------|
| 1 | `fac9a54` | 04-21 | Kimi K2 + MILM | 2 |
| 2 | `ef18594` | 04-23 | Qwen3-14B (scope) | 5 |
| 3 | `7e887c6` | 04-23 | Qwen3-14B (E2E) | 2 |
| 4 | `517a8dc` | 04-23 | **Qwen3-32B 全部** | 11 |
| 5 | `70b4e31` | 04-27 | DS V3.2 + V4 首批 | 12 |
| 6 | `1bceb1c` | 04-27 | DS V3.2 front | 1 |
| 7 | `f7110d0` | 04-28 | DS V4 MoE 拆分 | 4 |
| 8 | `1c800f1` | 04-29 | DS V4 attention 重构 | 2 |
| 9 | `d8cd44c` | 04-29 | DS V4 HC pre | 1 |

> **关键结论**：只需 checkout **9 个 commit**，即可覆盖仓库中所有模型的
> baseline。其中 Qwen 系列只需 3 次（Group 2/3/4），DeepSeek 系列需 5 次
>（Group 5/6/7/8/9），Kimi + MILM 只需 1 次（Group 1）。

### 3.4 采集流程示例

以 Group 4（Qwen3-32B 全部 11 个文件）为例：

```bash
# 1. 保存当前工作状态
git stash

# 2. 切换到 baseline commit（整个仓库恢复到该版本）
git checkout 517a8dc

# 3. 依次运行每个 scope，采集 golden + perf 数据
for scope in scope1 scope2 scope3; do
    python examples/models/qwen3/32b/qwen3_32b_decode_${scope}.py \
        -p a2a3 --runtime-profiling
done

# 也运行完整层
python examples/models/qwen3/32b/qwen3_32b_decode.py \
    -p a2a3 --runtime-profiling

# 4. 保存所有产物到 baselines/ 目录
# （此步骤在后续 §7 中详细说明）

# 5. 切回当前分支
git checkout main
git stash pop
```

一次 checkout 即可采集 Qwen3-32B 全部 11 个文件的 baseline 数据。

---

## 4. SOW 要求的 Baseline 维度

根据 SOW（"基于PyPTO的多模块SubAgent自动调优智能体"）的要求，Benchmark
基线需要覆盖以下维度：

### 4.1 必须收集的 Baseline

| 维度 | 说明 | 对应 SOW 要求 | 采集方式 |
|------|------|-------------|----------|
| **源代码** | 文件首次出现时的完整源码 | Agent 需理解算子结构，对比优化前后代码 | `git show <commit>:<path>` |
| **正确性（Golden）** | 输入/输出张量 | 验证执行 agent 需交叉比对结果 | `golden/runner.py` 自动保存 |
| **性能（Perf）** | Wall Time、任务统计、调度效率 | 性能诊断 agent 需量化优化收益 | `tools/perf_analyzer.py` |
| **编译参数** | `strategy`、`backend_type`、`dump_passes` 等 | Agent 需调整编译参数 | 从脚本中 `RunConfig.compile` 提取 |
| **运行时参数** | `block_dim`、`aicpu_thread_num`、`platform` | Agent 需调整运行时参数 | 从 `kernel_config.py` 的 `RUNTIME_CONFIG` 提取 |
| **内存使用** | 各内核各内存空间的利用率 | Agent 需判断 tiling 是否有优化空间 | `memory_after_AllocateMemoryAddr.txt` |
| **泳道图原始数据** | 任务时间线、调度器/编排器阶段 | 性能诊断 agent 解析泳道图识别瓶颈 | `perf_swimlane_*.json` |

### 4.2 Baseline Manifest 建议扩展

为满足 SOW 中 Agent 各模块的需求，`baseline.json` 建议扩展为：

```json
{
  "model": "qwen3-32b-decode-scope3",
  "baseline_group": 4,
  "git_commit": "517a8dc",
  "source_file": "examples/models/qwen3/32b/qwen3_32b_decode_scope3.py",
  "platform": "a2a3",
  "golden_dir": "baselines/qwen3-32b-scope3/golden",
  "perf_report": "baselines/qwen3-32b-scope3/perf_report.md",
  "swimlane_json": "baselines/qwen3-32b-scope3/perf_swimlane.json",
  "memory_report": "baselines/qwen3-32b-scope3/memory_report.txt",
  "correctness": {
    "rtol": 3e-3,
    "atol": 3e-3
  },
  "compile_params": {
    "strategy": "default",
    "backend_type": "Ascend910B",
    "dump_passes": true
  },
  "runtime_params": {
    "block_dim": 24,
    "aicpu_thread_num": 4,
    "platform": "a2a3"
  },
  "metrics": {
    "wall_time_us": null,
    "task_count": null,
    "exec_latency_ratio": null,
    "scheduler_idle_pct": null
  }
}
```

---

## 5. 定位 Baseline Commit

### 5.1 以主仓库（upstream）为准

主仓库（`upstream`）拥有权威的提交历史。在 fork 仓库中，`git fetch upstream`
即可将完整历史拉取到本地，无需前往主仓库页面操作。

```bash
# 确保上游历史为最新
git fetch upstream
```

### 5.2 `llm/examples/` — 按目录定位

对 E2E 推理脚本，首次新增该文件的 commit 即为 baseline：

```bash
# 查找模型在 llm/examples/ 中首次出现的 commit
git log --diff-filter=A --oneline upstream/main -- "llm/examples/<model_file>.py"
```

**示例** — Qwen3-14B：

```bash
$ git log --diff-filter=A --oneline upstream/main -- llm/examples/qwen3_14b_npu_generate.py
7e887c6 Add PyPTO LLM inference scaffold (#160)
```

→ Baseline commit：`7e887c6`

### 5.3 `examples/models/` — 按文件定位

Scope 文件是跨多个 PR 逐步新增的（先有 scope1，后有 scope2……），因此必须
**按具体文件**定位 baseline：

```bash
# 查找某个 scope 文件首次被新增的 commit
git log --diff-filter=A --oneline upstream/main -- \
    "examples/models/qwen3/32b/qwen3_32b_decode_scope3.py"
```

如果文件经历过重命名或目录移动（如目录重组），使用 `--follow` 穿越重命名：

```bash
git log --follow --diff-filter=A --oneline upstream/main -- \
    "examples/models/qwen3/32b/qwen3_32b_decode.py" | tail -1
```

**示例** — Qwen3-32B（当前路径 `examples/models/qwen3/32b/`）：

所有文件均在 `517a8dc` (#163) 首次出现在当前路径，属于 Group 4。

若使用 `--follow` 追溯更早的旧路径（`examples/models/qwen3/qwen3_32b_*.py`）：

| 文件 | 旧路径首次出现 | 当前路径首次出现 |
|------|---------------|-----------------|
| `qwen3_32b_decode_scope1.py` | `aa05a5b` (#36) | `517a8dc` (#163) |
| `qwen3_32b_decode_scope2.py` | `409759c` (#52) | `517a8dc` (#163) |
| `qwen3_32b_decode_scope3.py` | `6e13fcb` (#65) | `517a8dc` (#163) |
| `qwen3_32b_decode.py` | `5f79621` (#66) | `517a8dc` (#163) |

> **推荐使用当前路径的 `517a8dc`** 作为 baseline，因为它同时升级了
> `golden` API 调用方式，是整仓库 checkout 后能直接运行的版本。

### 5.4 常用 Git 参数说明

| 参数 | 含义 |
|------|------|
| `--diff-filter=A` | 只显示**新增**该文件的 commit（排除修改和删除） |
| `--follow` | 追踪文件重命名历史 |
| `upstream/main` | 在主仓库的历史中搜索 |
| `tail -1` | 取最早的 commit（git log 默认从新到旧） |

---

## 6. 获取文件首次出现时的状态

定位到 baseline commit 后，需要获取该文件在首次出现时的**完整源码内容**。

### 6.1 查看文件在指定 commit 时的内容

```bash
# 语法：git show <commit>:<path>
git show 6e13fcb:examples/models/qwen3/32b/qwen3_32b_decode_scope3.py
```

该命令输出文件在该 commit 时的完整内容（不影响工作区）。

### 6.2 将文件导出为本地副本

```bash
# 导出到指定路径
git show 6e13fcb:examples/models/qwen3/32b/qwen3_32b_decode_scope3.py \
    > baselines/qwen3-32b-scope3/source_baseline.py
```

### 6.3 将文件还原到工作区（会修改工作区文件）

```bash
# 将工作区的该文件替换为 baseline 版本
git checkout 6e13fcb -- examples/models/qwen3/32b/qwen3_32b_decode_scope3.py

# 用完后恢复为当前版本
git checkout HEAD -- examples/models/qwen3/32b/qwen3_32b_decode_scope3.py
```

---

## 7. 处理已删除的文件

部分文件在后续重构中被删除（如 `qwen3_32b_decode_mixed.py` 在 #147 中被移除）。
这些文件仍然完整保留在 git 历史中，可以恢复。

### 7.1 找到被删除的文件列表

```bash
# 列出某个目录下所有曾被删除的文件
git log --diff-filter=D --name-only --oneline upstream/main -- "examples/models/"
```

**仓库中已删除的文件示例**：

| 被删除的文件 | 删除 commit | 首次新增 commit |
|-------------|-----------|----------------|
| `qwen3_32b_decode_mixed.py` | `7259971` (#147) | `8cfb7c0` (#99) |
| `qwen3_32b_decode_scope1_tile.py` | `fac9a54` (#145) | 需查询 |
| `qwen3_32b_prefill_tilelet.py` | `fac9a54` (#145) | 需查询 |
| `deepseek_v3_2_decode_front_draft.py` | `1bceb1c` (#187) | 需查询 |
| `deepseek_v4_decode_moe_draft.py` | `f7110d0` (#194) | 需查询 |

### 7.2 找到已删除文件的首次出现

```bash
# 对已删除的文件，--diff-filter=A 依然有效
git log --diff-filter=A --oneline upstream/main -- \
    "examples/models/qwen3/qwen3_32b_decode_mixed.py"
# 输出：8cfb7c0 Refactor: Qwen3 decode with 3-scope architecture ...
```

### 7.3 恢复已删除文件在首次出现时的内容

```bash
# 查看内容（不修改工作区）
git show 8cfb7c0:examples/models/qwen3/qwen3_32b_decode_mixed.py

# 导出到本地
git show 8cfb7c0:examples/models/qwen3/qwen3_32b_decode_mixed.py \
    > baselines/qwen3-32b-mixed/source_baseline.py

# 恢复到工作区
git checkout 8cfb7c0 -- examples/models/qwen3/qwen3_32b_decode_mixed.py
```

### 7.4 查看已删除文件的完整生命周期

```bash
# 显示文件从新增到删除的所有 commit（包括修改）
git log --oneline upstream/main --all -- \
    "examples/models/qwen3/qwen3_32b_decode_mixed.py"
```

> **注意**：被删除的文件可能是因为重构（如 scope 拆分、API 更新）而移除的。
> 作为 baseline 使用时，需要确认该文件在首次出现的 commit 上能够独立运行。
> 如果不能（如依赖了同一 commit 中的其他文件），则需要以该 commit 的
> 完整工作区为准。

---

## 8. 建立 Baseline

定位到 baseline commit 后，需要固定以下产物：

### 8.1 打 Git Tag

```bash
git tag baseline/<model>/<scope>/v0 <commit-sha>

# 示例（使用 §3 中的分组 commit）：
git tag baseline/qwen3-14b/e2e/v0 7e887c6         # Group 3
git tag baseline/qwen3-32b/v0 517a8dc             # Group 4
git tag baseline/deepseek-v3.2/v0 70b4e31         # Group 5
```

### 8.2 保存 Golden 数据（正确性基线）

将整仓库切换到 baseline commit，运行目标脚本并持久化输入/输出张量。
`golden/runner.py` 框架会自动将数据保存到 `build_output/`：

```bash
# 切换整仓库到 baseline commit（确保 golden 模块版本匹配）
git stash
git checkout <baseline-commit>

# 运行并收集 golden 数据
python <baseline_script>.py -p <platform>

# 持久化 golden 数据
BASELINE_DIR="baselines/<model>/golden"
mkdir -p "$BASELINE_DIR"
cp -r build_output/<ProgramName>_*/data/* "$BASELINE_DIR/"
```

后续验证优化版本时，直接加载此 golden 数据：

```python
result = run(
    program=build_optimized_program(),
    tensor_specs=build_tensor_specs(),
    golden_fn=golden_fn,
    config=RunConfig(rtol=3e-3, atol=3e-3, ...),
    golden_data="baselines/<model>/golden",  # 固定输入验证
)
```

### 8.3 保存性能基线

```bash
# 在真机上带 profiling 运行
python <baseline_script>.py -p a2a3 --runtime-profiling

# 生成性能报告
python tools/perf_analyzer.py build_output/<ProgramName>_*/ \
    -o baselines/<model>/perf_report.md
```

### 8.4 保存编译产物（泳道图 + 内存报告）

```bash
# 泳道图原始数据
cp build_output/<ProgramName>_*/swimlane_data/perf_swimlane_*.json \
    baselines/<model>/

# 内存使用报告
cp build_output/<ProgramName>_*/report/memory_after_AllocateMemoryAddr.txt \
    baselines/<model>/

# 运行时配置（含 block_dim, aicpu_thread_num 等）
cp build_output/<ProgramName>_*/kernel_config.py \
    baselines/<model>/
```

### 8.5 编写 Baseline Manifest

创建 `baselines/<model>/baseline.json`，便于自动化工具读取：

```json
{
  "model": "qwen3-32b-decode-scope3",
  "git_tag": "baseline/qwen3-32b/scope3/v0",
  "git_commit": "6e13fcb",
  "source_file": "examples/models/qwen3/32b/qwen3_32b_decode_scope3.py",
  "platform": "a2a3",
  "golden_dir": "baselines/qwen3-32b-scope3/golden",
  "perf_report": "baselines/qwen3-32b-scope3/perf_report.md",
  "config": {
    "rtol": 3e-3,
    "atol": 3e-3
  }
}
```

---

## 9. 对比优化结果与 Baseline

### 9.1 正确性对比

```bash
# 使用 baseline golden 数据验证优化版本
python <optimized_script>.py -p <platform> \
    --golden-data baselines/<model>/golden
```

`run()` 函数会从 golden 目录加载固定输入，在设备上执行优化后的程序，然后使用
相同的 rtol/atol 验证输出与保存的 golden 输出是否一致。

### 9.2 性能对比

```bash
# 生成优化版本的性能报告
python tools/perf_analyzer.py build_output/<Optimized>_*/ -o opt_perf.md

# 与 baseline 性能报告对比，关注以下指标：
# - Wall Time：应下降
# - Exec/Latency ratio：应上升
# - Scheduler idle%：应下降
# - 不应出现新的 WARN 诊断
```

完整的验证检查清单见 `docs/opt-design/optimization_correctness_methodology.md`
第 3、8 节。

---

## 10. 命令速查

```bash
# ── 整仓库切换到 baseline commit（最保险的方式） ──
git stash
git checkout <baseline-commit>
# ... 运行 baseline 脚本、采集数据 ...
git checkout main
git stash pop

# ── 定位 llm/examples/ 模型的 baseline ──
git log --diff-filter=A --oneline upstream/main -- "llm/examples/<file>.py"

# ── 定位 examples/models/ 各 scope 的 baseline（按文件） ──
git log --diff-filter=A --oneline upstream/main -- \
    "examples/models/<model>/<size>/<file>.py"

# ── 处理文件重命名 ──
git log --follow --diff-filter=A --oneline upstream/main -- "<path>" | tail -1

# ── 查看文件在指定 commit 时的内容 ──
git show <commit>:<path>

# ── 导出为本地副本 ──
git show <commit>:<path> > baselines/<model>/source_baseline.py

# ── 找已删除文件的首次出现 ──
git log --diff-filter=A --oneline upstream/main -- "<deleted_file_path>"

# ── 恢复已删除文件 ──
git show <commit>:<deleted_file_path> > recovered.py

# ── 列出某目录下所有被删除过的文件 ──
git log --diff-filter=D --name-only --oneline upstream/main -- "<dir>/"

# ── 打 baseline tag ──
git tag baseline/<model>/<scope>/v0 <commit>

# ── 保存 golden 数据 ──
cp -r build_output/<Name>_*/data/* baselines/<model>/golden/

# ── 保存性能基线 ──
python tools/perf_analyzer.py build_output/<Name>_*/ -o baselines/<model>/perf_report.md

# ── 保存泳道图和内存报告 ──
cp build_output/<Name>_*/swimlane_data/perf_swimlane_*.json baselines/<model>/
cp build_output/<Name>_*/report/memory_after_AllocateMemoryAddr.txt baselines/<model>/
cp build_output/<Name>_*/kernel_config.py baselines/<model>/
```
