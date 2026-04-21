# Qwen3-32B Decode Scope 2 性能优化机会盘点

本计划盘点 `@/root/pypto-lib/examples/models/qwen-opt/qwen3_32b_decode_scope2_opt.py` 在 Opaque 编排层与 Tile DSL 重写两条路径上尚未利用的优化机会，以 a2a3sim 可观测信号（任务数 / 编排耗时 / 内核数 / GM 分配）为衡量重点，标注独立项与少数组合项，并为每项标注**改动边界**（仅 `pypto-lib` / 需实测验证 / 需 pypto Python API 扩展 / 需 pto-isa 或 CANN runtime 扩展）。

## 改动边界分类（重要）

`pypto-lib` = 仅 `examples/` 下的示例 `.py` 与 `golden/` 测试框架；真正的 pypto 编译器 / IR / runtime 以 site-package 形式安装在 `.venv/.../site-packages/pypto/`。四档分类：

| 代号 | 含义 | 判定依据 |
|---|---|---|
| **①** `pypto-lib only` | 只改 `examples/models/qwen-opt/*.py`，使用的所有 API 已在其他 example 成功落地 | API 已在 `qwen3_32b_decode_mixed.py` / `...tile.py` / `hello_world.py` 等验证 |
| **②** `pypto-lib + 需实测验证` | API 存在但本条组合未在示例中出现；可能触发 pypto 编译器边界情况 | API 层 OK，但语义交互需实跑编译 + rtol 回归确认；若失败可能回退要求 pypto 修复 |
| **③** `需 pypto Python API 扩展` | pypto `language/` 层没有暴露该能力；需新增 Python API + IR op + lowering pass | 对 pypto 框架本身的代码改动（非 pypto-lib） |
| **④** `需 pto-isa / CANN runtime 扩展` | 需改 Ascend 底层 ISA 指令集、调度器或 runtime | 本清单中暂无此类（tprefetch 指令已存在于 pto-isa，所以 S2-B4 属于 ③ 而非 ④） |

## 0. 现状基线

`scope2_opt.py` vs `scope2.py` 的 diff 仅两处：
- `for gi in pl.parallel(total_q_groups)` — 已落地分析文档中的 **S2-2（group parallel）**。
- `RunConfig.compile.output_dir` — profile 目录定制。

其余完全同原版；`qwen3_optimization_analysis.md` 中 S2-1、S2-4 及本计划新增机会均未实施。

## 1. 执行摘要（优先级排序）

符号：**Opaque** = 当前 `scope2_opt.py` 结构内可直接改；**TileDSL** = 需拆 `InCore + Orchestration`；**Combined** = 需与其他项同时启用才能充分展开。`Sim↑` = 在 a2a3sim 上可观测信号（任务数 / 编排器耗时 / 内核数 / GM）。**改动** = 改动边界（① / ② / ③ / ④ 见上节）。

| P | 编号 | 类型 | 优化 | 模拟器收益 | 成本 | 风险 | **改动** |
|---|---|---|---|---|---|---|---|
| **P0** | S2-A3 | Opaque | QK+Softmax+SV per-sb 融合，消除 2 个全局同步屏障 | **Sim↑↑** incore `3×sb → 1×sb`；去 `all_raw_scores`(256 KB)，压缩 `all_exp_padded` | 中 | 低 | **①** |
| **P0** | S2-A4 | Opaque | 删前置 Q-pad 零初始化循环；`q_padded` 下沉为 per-group 局部 | Sim↑ -120 incore 任务；-512 KB GM | 低 | 极低 | **①** |
| **P0** | S2-A5 | Opaque | 删 `attn_row` 中间缓冲，直写 `attn_out[b,...]` | Sim↑ -16 assemble；-256 KB GM | 低 | 极低 | **①** |
| **P1** | **S2-C1** 组合 | Opaque | S2-A3 + S2-A6：sb 顺序内线式 flash-attention | **Sim↑↑↑** 消除 5 个 per-group GM 张量 (~770 KB/group) + 全部 Stage 5 任务 | 中 | 低–中（精度回归） | **①** |
| **P1** | S2-A1 | Opaque | `for b in pl.parallel(batch, chunk=...)` 批次并行 | Sim↑ (b,gi)=128 任务打满 24 AIC | 低 | 低 | **②** |
| **P1** | **S2-C3** 组合 | Opaque | S2-A7+A8+A9：Stage 1 RoPE 向量化到 `[8,64]=2 KB` TILELET MAX | Sim↑ TILELET 13–25% → 100%；-1920 个 Q-pad assemble | 中 | 低 | **②** (A9 需验证) |
| **P2** | S2-A7 | Opaque | K-RoPE 跨 8 KV-head 批量 | Sim↑ -42 vec op/batch | 低 | 低 | **①** |
| **P2** | S2-A8 | Opaque | Q-RoPE 跨 8 Q-head 批量 + 整块 assemble `q_padded` | Sim↑ -15 assemble/group × 128 | 低 | 低 | **①** |
| **P2** | S2-A9 | Opaque | K-RoPE `rot_lo/rot_hi` 合并 `[1,head_dim]` cast+assemble | Sim↑ -1 op/head × 16 | 低 | 低（依赖 tile concat 支持） | **②** |
| **P2** | S2-B1 | TileDSL | `pl.auto_incore(split=UP_DOWN)` 包裹融合体，AIC/AIV 跨核流水 | Sim↑ 关键路径 AIC+AIV+AIC → max(AIC,AIV) | 中–高 | 中 | **②** |
| **P2** | S2-B3 | TileDSL | 显式 `q_padded → L1` 驻留，跨 sb 复用 | Sim↑ 减 GM 读 `ctx_blocks×`（~32 MB） | 中 | 低 | **①** |
| **P3** | S2-A11 | Opaque | sb 循环 `chunk` 参数调优 | 依赖组合 | 低 | 低 | **①** |
| **P3** | S2-A12 | Opaque | 去 `exp_scores` BF16↔FP32 round-trip | -1 cast/sb；需精度回归 | 低 | **中** | **①** |
| **P3** | S2-B2 | TileDSL | 整体拆 5 个 InCore kernel（同 `qwen3_32b_decode_tile.py`） | Sim↑ 结构清晰 | 高 | 中 | **①** |
| **P3** | S2-B4 | TileDSL | K/V tile double-buffer 预取 | Sim↑ swimlane 交叠 | 高 | 中 | **③** |
| — | S2-A2 | Opaque | group parallel | **已落地** | — | — | ① |
| — | S2-A10 | Opaque | cos/sin 预切分复用 | 编译器自动 hoist | — | — | ① |

## 2. Category A — Opaque 层独立优化

### S2-A1　Batch 循环并行化
- **现状**：`for b in pl.range(batch)` 顺序；16 批次内部 gi 并行，批次间存在不必要同步。
- **观察**：每批次读 `seq_lens[b]` / `rope_cos[pos]`，写 `k_cache[b*...]` / `v_cache[b*...]` / `attn_out[b,:]`，区域不相交。
- **改动**：`for b in pl.parallel(batch, chunk=4 或 8)`，与已有 S2-A2（group parallel）叠加。
- **风险点**：`all_raw_scores` / `all_exp_padded` / `all_oi_tmp` / `all_cur_mi` / `all_cur_li` / `all_q_padded` 作为 per-(b,gi) 中间物在 batch 并行后需 pypto 每迭代实例化；若不支持则降级 `chunk=1`。
- **依赖**：独立，但与 S2-C1 组合后天然形成"外并行 + 内串行"。

### S2-A3　QK + Softmax + SV per-sb 融合（= 分析文档 S2-4）
- **现状**：三个独立 `pl.at(level=CORE_GROUP, optimization=chunked_loop_optimizer)` 各自 `for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH)`。中间结果 `all_raw_scores`、`all_exp_padded` 落 GM，两个全局同步屏障。
- **改动**：合并为单个 `pl.at` 块，body 依次 QK → softmax → SV；sb 仍 parallel `chunk=SB_BATCH`。`raw_scores`、`exp_scores` 保留在 chunk-loop 的 anonymous incore 内部（不落 GM）。
- **硬件约束**（对齐 `qwen3_tilelet.md`）：Mat 峰值 20 KB（K/V 交替驻留），Acc ≤ 8 KB，Vec scores `[8,64] FP32 = 2 KB TILELET MAX`，L0A 4 KB / L0B 16 KB，均远低于上限。
- **模拟器信号**：内核数 -2；`ctx_blocks × 2 × groups × batch` 个屏障任务消失；`all_raw_scores` 256 KB GM 移除。
- **正确性风险**：低，每 sb 内部数学与原版一致。

### S2-A4　删除 Q-pad 预初始化循环
- **现状**：
  ```python
  all_q_padded = pl.create_tensor([batch * total_q_groups * Q_HEAD_PAD, head_dim], BF16)
  with pl.at(level=pl.Level.CORE_GROUP):
      for idx in pl.range(batch * total_q_groups):  # 128 次
          all_q_padded = pl.assemble(all_q_padded, pl.cast(pl.full(...), BF16),
                                     [idx*Q_HEAD_PAD + Q_HEAD_BATCH, 0])
  ```
- **改动**：效仿 `qwen3_32b_decode_mixed.py` / `..._tile.py`，`q_padded` 下沉为 `for gi in pl.parallel(...)` 内局部 `pl.create_tensor([Q_HEAD_PAD, head_dim], BF16)`；padding 行填零改为 per-group 单次 `pl.assemble(..., [Q_HEAD_BATCH, 0])`，从 batch × TQG = 128 次降到 TQG = 8 次。
- **模拟器信号**：-120 incore 任务；-1 个 512 KB GM 张量。
- **正确性风险**：极低。`q_padded` 只在 (b,gi) 内被读写。

### S2-A5　删除 `attn_row` 中间缓冲
- **现状**：每 `b` 先 `attn_row = pl.create_tensor([1, hidden], BF16)`，gi 循环 assemble 各 group 的 `ctx_flat_bf16`，最后 `attn_out = pl.assemble(attn_out, attn_row, [b, 0])`。
- **改动**：直接 `attn_out = pl.assemble(attn_out, ctx_flat_bf16, [b, q_base*head_dim])`。
- **模拟器信号**：-16 final assemble；少一个 per-batch 临时张量。
- **正确性风险**：极低。group 间写入 `attn_out` 列区域不重叠。

### S2-A6　Stage 5 融入 sb 序列化增量更新
- **现状**：Stage 5 独立 `pl.at`，从 GM 读 `all_oi_tmp[sb]` / `all_cur_mi[sb]` / `all_cur_li[sb]`，sb 顺序更新 `oi/mi/li`。
- **改动（配合 S2-A3）**：sb 改为顺序循环（loop-carried `oi/mi/li`）放入单个 incore：
  ```python
  with pl.at(level=pl.Level.CORE_GROUP):
      for sb in pl.range(ctx_blocks):              # 顺序，loop-carried oi/mi/li
          # QK → softmax → SV（融合体，S2-A3 内容）
          if sb == 0:
              oi, mi, li = oi_tmp, cur_mi, cur_li
          else:
              mi_new = pl.maximum(mi, cur_mi)
              alpha = pl.exp(pl.sub(mi, mi_new)); beta = pl.exp(pl.sub(cur_mi, mi_new))
              li = pl.add(pl.mul(alpha, li), pl.mul(beta, cur_li))
              oi = pl.add(pl.row_expand_mul(oi, alpha), pl.row_expand_mul(oi_tmp, beta))
              mi = mi_new
      ctx = pl.row_expand_div(oi, li)
      attn_out = pl.assemble(attn_out, pl.cast(pl.reshape(ctx, [1, QHB*HD]), BF16),
                             [b, q_base * head_dim])
  ```
- **模拟器信号**：消除 `all_oi_tmp`(~512 KB) + `all_cur_mi`(2 KB) + `all_cur_li`(2 KB) 3 个 per-group GM 张量；Stage 5 的 `ctx_blocks - 1` 个 slice/assemble 任务全部消失；`pl.at` 块从 4 降到 1。
- **正确性风险**：低–中。必须保留原 Stage 3 的 BF16 round-trip：`exp_scores → BF16 → 同一 BF16 tile 做 SV 的 A 输入`，同时 `cur_li = row_sum(cast(BF16→FP32))` 保证与 SV 量化同源。需固定输入 golden 回归。
- **依赖**：必须与 S2-A3 同时启用（否则 sb 无法线性化）。
- **并行度代价**：sb 变顺序，需由 (b,gi) 外层补齐 → 即 S2-C2 = S2-C1 + S2-A1。

### S2-A7　K-RoPE 跨 8 KV-head 批量化
- **现状**：`for ki in pl.parallel(num_kv_heads, chunk=8)` 每 head 做 `[1, half_dim]=256 B` FP32 RoPE，TILELET 利用率 13%。
- **改动**：8 KV-head 的 `k_proj` 拼成 `[8, head_dim]` FP32，批量做 `[8, 64]=2 KB TILELET MAX` 的 col_expand_mul/add/sub；scatter 到 8 个 `cache_row`（因 `cache_row` 按 MS=4096 分布，仍需 8 次 assemble，但算术核心一次完成）。
- **模拟器信号**：vec ops `8×6 → 6`，-42 ops/batch。
- **正确性风险**：低。

### S2-A8　Q-RoPE 跨 8 Q-head 批量化 + 整块写
- **现状**：`for qi in pl.range(Q_HEAD_BATCH=8)` 每 head `[1, half_dim]` FP32；每 qi 两次 `pl.assemble`（lo/hi 半）→ 16 次 assemble/group。
- **改动**：
  1. 8 Q-head `q_proj` 拼 `[8, head_dim]` FP32，批量 RoPE = `[8, half_dim]=2 KB TILELET MAX`。
  2. 单次 `pl.cast([8, head_dim] FP32 → BF16)`。
  3. 单次 assemble 写 `all_q_padded[qpad_base : qpad_base + Q_HEAD_BATCH, 0:head_dim]`（或配合 S2-A4 的局部 `q_padded`）。
- **模拟器信号**：per-group 16 → 1 assemble；全程 `batch × TQG × 15 = 1920` 个 assemble 任务消失。**这是对 orchestrator 最大的一次性削减**。
- **正确性风险**：低。

### S2-A9　K-RoPE `rot_lo/rot_hi` 合并 cast+assemble
- **现状**：`rot_lo FP32 → cast BF16 → assemble [cache_row, 0]`；`rot_hi` 同法 `[cache_row, half_dim]`。2 cast + 2 assemble / head。
- **改动**：若 pypto 支持 Vec 上下文 tile 级拼接（`pl.assemble` 作用在 tile 而非 tensor 上），把 `rot_lo/rot_hi` 合成 `[1, head_dim]` FP32，单次 `pl.cast` + 单次 assemble。
- **不确定性**：需验证 pypto tile 级拼接能力；不支持则保持现状。
- **模拟器信号**：-16 cast + -16 assemble /batch。
- **正确性风险**：低。

### S2-A11　sb 循环 `chunk` 参数
- **启用 S2-C1 时**：sb 顺序；`chunk=SB_BATCH` 或直接 `pl.range` 均可。
- **仅 S2-A3 不 S2-A6 时**：sb 仍 parallel；`chunk=1` / `chunk=8` 提供额外 sb 维并行，但任务数按 `ctx_blocks / chunk` 增加 → 需 profiling。
- 纯 profiling 项，无结构性变化。

### S2-A12　去 `exp_scores` BF16↔FP32 round-trip
- **改动 A（激进）**：`cur_li = pl.row_sum(exp_scores)` 直接 FP32 求和；`li` 与 SV 量化不再 bit-identical，差异 ≤ BF16 噪声 × SEQ_TILE。
- **改动 B（保守）**：写为 `pl.row_sum(pl.cast(exp_scores_bf16, FP32))` 融合形式，语义等价现状。
- **建议**：默认只做 B；A 作为可选实验，需 rtol=1e-3 精度回归。

## 3. Category B — Tile DSL 重写（更高天花板）

### S2-B1　`pl.auto_incore(split=UP_DOWN)` 跨核流水
- **目标**：融合后的 per-sb 体 `QK(AIC) → softmax(AIV) → SV(AIC) → update(AIV)` 自动 AIC/AIV 切分 + ping-pong。
- **参考**：`qwen3_32b_decode_mixed.py` Scope 1 的 `pl.auto_incore(split=pl.SplitMode.UP_DOWN)`。
- **模拟器信号**：swimlane 上 AIC / AIV 泳道交叠；per-sb 关键路径 `AIC+AIV+AIC → max(AIC, AIV)`。
- **风险**：sb loop-carried 依赖（S2-A6）与 auto_incore 的兼容性需验证。
- **依赖**：S2-A3 必须先做。

### S2-B2　整体拆为 InCore + Orchestration
- `qwen3_32b_decode_tile.py` 已展示 Scope 2 的 tile DSL 形态（5 个 InCore kernel）。
- **建议**：避免重复 tile DSL 重写。优先吃透 Category A 的 Opaque 优化；仅当需要 S2-B3/B4 时才动。

### S2-B3　显式 `q_padded` L1 驻留
- **改动**：`q_l1 = pl.load(q_padded, target_memory=Mat)` 一次，`q_l0a = pl.move(q_l1, Left)` 在 sb 外保持；sb 只重复 load K/V。
- **模拟器信号**：`all_q_padded` GM 读次数 `ctx_blocks×` → `1×`/group；全局 ≈ 32 MB GM 带宽节省。

### S2-B4　K/V tile double-buffer 预取
- **改动**：`pl.load(k_cache[sb+1])` 到 L1 第二 buffer，与 `k_cache[sb]` 的 QK 并行；SV 同构。
- **模拟器信号**：swimlane 上 K load / compute 泳道交叠；per-sb latency 下降。
- **依赖**：S2-B2 结构。

## 4. Category C — 组合优化（协同效应）

### S2-C1 = S2-A3 + S2-A6 — sb 单次 pass Flash-Attention
- 单独 S2-A3 只消除 2 个同步屏障；单独 S2-A6 无法实现（Stage 5 需要读 Stage 4 的 GM 结果）。两者合并后：
  1. 消除 `all_raw_scores` / `all_exp_padded` / `all_oi_tmp` / `all_cur_mi` / `all_cur_li` 共 5 个 per-group GM 张量（~770 KB × 128 group ≈ **98 MB** GM 削减）。
  2. attention per-(b,gi) 压缩为 1 个 incore scope；sb 顺序 + loop-carried `oi/mi/li`。
  3. orchestrator 的 `pl.at` 块从 4 降到 1；`alloc/params/fanin` 计数按 `ctx_blocks` 倍缩减。
- **Scope 2 最大收益单点**。

### S2-C2 = S2-C1 + S2-A1 (+ 已有 S2-A2) — "外并行 + 内串行"
- S2-C1 使 sb 顺序，失去 sb 并行；由 (b,gi) 外层补齐 → `batch × total_q_groups = 128` 并行任务，24 AIC 下约 6 轮。
- **模拟器信号**：并行任务数 `8 → 128`；orchestrator `params/fanin` 吞吐曲线变"宽矮"。

### S2-C3 = S2-A7 + S2-A8 + S2-A9 — "Stage 1 TILELET 满载"
- 三项都是把 Stage 1 `[1, half_dim]` 零散向量指令换成 `[8, 64] = 2 KB` 批量指令。单独做任一项，dataflow 桥接成本仍由零散指令主导。
- **模拟器信号**：Stage 1 `Avg Exec` 显著下降；per-batch 任务数 ↓ 数十；per group 的 Q-pad assemble 16 → 1。

## 4.5 改动边界详解

本节说明上表中每项的边界判定依据；②/③ 项额外列出"若 pypto 不支持时"的回退路径。

### ① `pypto-lib only` 项（高信心，直接改示例）

| 编号 | 依据 |
|---|---|
| S2-A3 (QK+Soft+SV 融合) | 单纯重排 `with pl.at(CORE_GROUP, optimization=pl.chunked_loop_optimizer)` 块与共享 `for sb in pl.parallel` 的内容；该模式在 `qwen3_32b_decode_scope1.py` Stage 2 的 "single incore" 形态已用。 |
| S2-A4 (删 Q-pad 初始化) | `pl.create_tensor([Q_HEAD_PAD, head_dim], BF16)` 在 `for gi in pl.parallel(...)` 内的 per-iteration 实例化已在 `qwen3_32b_decode_mixed.py:296 q_padded = pl.create_tensor(...)` 和 `qwen3_32b_decode_tile.py` 多次使用。 |
| S2-A5 (删 attn_row) | `pl.assemble(attn_out, tile, [b, col])` 在 `pl.parallel(b)` 体内直接写 tensor 是基础语义，如 `hello_world.py:43`、`matmul.py:57` 等。 |
| S2-A6 (Stage 5 融入 sb) | `pl.range` 顺序循环内 loop-carried 变量（`oi/mi/li = pl.add(...)`）与 Scope 1 的 matmul-acc 累加同构（`qwen3_32b_decode_scope1.py:109-115`）。 |
| S2-A7 / A8 (RoPE 批量化) | 纯算法重写：`pl.slice` 一次取 `[1, NUM_KV_HEADS*head_dim]` 连续切片 + `pl.reshape([NUM_KV_HEADS, head_dim])` + 批量 `pl.col_expand_mul`；所有 API 均在 example 中广泛使用。 |
| S2-A11 / A12 | 纯参数调优 / 算法改写。 |
| S2-B2 (拆 InCore kernel) | `qwen3_32b_decode_tile.py` 已整体展示相同形态。 |
| S2-B3 (显式 L1 驻留) | `pl.load(target_memory=pl.MemorySpace.Mat)` + `pl.move(target_memory=pl.MemorySpace.Left)` 在 `qwen3_32b_decode_scope1_tile_opt.py:121-124` 已有同构用法。 |

### ② `pypto-lib + 需实测验证` 项

#### S2-A1 — Batch 并行化
- **假设**：pypto 在 `for b in pl.parallel(batch, chunk=...)` 内对"外层 create 的 per-(b,gi) 中间 tensor"（`all_raw_scores`、`all_exp_padded`、`all_oi_tmp`、`all_cur_mi`、`all_cur_li`、`all_q_padded`）能正确处理写入分区（因为当前写入索引含 `b`，本应分区无冲突）。
- **风险**：pypto 可能在 `pl.parallel(b)` 外层下对这些 GM tensor 的生命周期管理按"单实例"处理，导致并行写入被冲突检测器拒绝或生成串行屏障。
- **实测方法**：先对仅启用 S2-C1 + S2-A1 的版本跑 `--platform a2a3sim --golden-data`，观察是否编译通过且精度 OK；若失败查看 `passes_dump/` 中具体哪个 pass 报错。
- **回退路径**：
  1. 若 pypto 不支持 → 先做 S2-C1（消除 5 个中间 tensor），再 S2-A1 自然解除冲突（因为中间 tensor 消失）。这是**推荐顺序**。
  2. 若即使消除中间 tensor 仍有冲突 → 降级 `chunk=1`（每 chunk 内 batch 串行，只在 chunk 间并行，有限并行度）。
  3. 最坏情况：仅靠 gi parallel（当前 8 路），放弃 batch 并行。

#### S2-A9 — K-RoPE `rot_lo/rot_hi` 合并
- **假设**：`pl.concat` 在 Vec 上下文对两个 `[1, 64] FP32` tile 可编排成单条向量指令（`pypto.language.op.unified_ops.concat` + `pypto.language.op.tile_ops.concat` 均已存在）。
- **风险**：Vec 上下文的 `pl.concat` 是否强制生成额外 `move` 指令、能否被后续 `pl.cast` 融合，需看编译器 pass。
- **实测方法**：单独构造一个 `concat([1,64], [1,64]) → cast → assemble` 小例子跑 sim，对比任务数和生成 C++ 源码。
- **回退路径**：若 concat 成本 ≥ 两次独立 cast+assemble，则保持原 2 次 cast/2 次 assemble，**放弃本项**（S2-C3 仅保留 A7+A8 仍有大部分收益）。

#### S2-B1 — `pl.auto_incore(split=UP_DOWN)` 跨核包裹 sb 融合体
- **假设**：pypto 能跨 AIC/AIV 切分一个 `for sb in pl.range(...)` 顺序循环，且 loop-carried `oi/mi/li` 变量在 ping-pong 切分中仍保持正确。
- **风险点 A**：现有 `pl.auto_incore(UP_DOWN)` 在 `qwen3_32b_decode_mixed.py:205` Scope 1 中包裹的循环只有 matmul-acc 这种**单核内** loop-carried（`q_acc` 只在 AIC 侧流转），而 `oi/mi/li` 同时被 AIV（softmax）和 AIC（SV matmul）访问，跨核 round-trip 每 sb 必发生。
- **风险点 B**：auto_incore 可能要求 loop-carried 变量的类型 / shape 简单（`[Q_HEAD_BATCH, head_dim]` FP32 tile 在 Vec 内），是否可自动 stage 到 Mat / Vec 跨核传递未知。
- **实测方法**：先实现 S2-C1（sb 顺序融合无 auto_incore），通过 rtol 后再尝试叠加 `pl.auto_incore(UP_DOWN)`。
- **回退路径**：
  1. 若 auto_incore 失败 → 保持 S2-C1 形态但单 core group（不跨核切分），牺牲 AIC/AIV 流水并行度但保结构正确。
  2. 或改为 S2-B2 全盘 Tile DSL，手工写 5 个 InCore kernel + 显式 sync（与 `qwen3_32b_decode_tile.py` 同构）。

#### S2-C2 / S2-C3 — 组合项
- S2-C2 = S2-C1 (①) + S2-A1 (②) → 归类为 **②**，实测风险来自 S2-A1；推荐按 "先 S2-C1 再 S2-A1" 顺序逐步叠加。
- S2-C3 = S2-A7+A8+A9 → 含 S2-A9 (②)，归类为 **②**；但若 S2-A9 回退，A7+A8 仍可独立落地，C3 整体收益不会大幅下降。

### ③ `需 pypto Python API 扩展` 项

#### S2-B4 — K/V tile double-buffer 预取
- **现状**：pypto `language/` 层未发现 `double_buffer=True` 参数、`pl.async_load`、或 `pl.prefetch`。grep 全量 site-package 仅在 `pto-isa/docs/...` 中找到 `tprefetch` ISA 指令，即**底层 ISA 已支持**但 Python API 层未暴露。
- **需要改动**：
  1. pypto `language/op/tile_ops.py` 新增 `async_load(tensor, ..., buffer_id: int)` 或在 `load(...)` 加 `double_buffer=True` 参数。
  2. pypto `ir/op/tile_ops.py` + `ir/builder.py` 新增对应 IR op。
  3. lowering pass：生成 `tprefetch` ISA 指令 + 双 buffer 槽位分配。
- **不改 pypto 时可做的**：可依赖 pypto 编译器**已有**的自动双缓冲 pass（若存在）——需查 `passes_dump/` 中是否有相关 pass 名（如 `DoubleBufferInsertion`）。若已自动处理，则示例层无需改动；若未处理，纯 pypto-lib 无法显式控制，本项**跳过**。
- **结论**：若用户只想优化 `pypto-lib`，**S2-B4 不在范围内**。

### ④ `需 pto-isa / CANN runtime 扩展` 项
- 本清单暂无。`tprefetch` 已在 pto-isa 存在，AICPU 调度器、core mapping 等 runtime 组件也已有能力；本 plan 中未识别出需要改 ISA 或 runtime 的优化。

### 实测勘误（2026-04-21 首轮编译反馈）

首轮按 ① 一次性落地后，编译器在 A2/A3 平台报出 3 个错误，揭示**原 ① 标注偏乐观**。修正后分类：

| 编号 | 原标注 | 实测结果 | 根因 / 证据 |
|---|---|---|---|
| **S2-A7** K-RoPE 批量 | ① | **②⁻ 失败** | `pl.reshape([1, kv_hidden] → [num_kv_heads, head_dim])` 后再 `pl.slice` 半幅，在 Vec tile 上触发 `pto.textract`；A2/A3 要求 `textract` src=Mat。`intermediate/rope.py:65` 有反面注释："so each becomes a separate tile.load (no textract)"。所有生产模型（`mixed.py`、`tile.py`、`qwen3_32b_decode.py`、`kimi`、`milm`）均用 per-head 直接从 GM tensor slice |
| **S2-A8** Q-RoPE 批量 | ① | **②⁻ 失败** | 同上，reshape 后对 Vec tile 的二次 slice 不支持 |
| **S2-A5** 直写 `attn_out` | ① | **② 需验证** | 无生产模型在 `pl.parallel(gi)` 体内直接 `pl.assemble(attn_out, ...)`；全部使用 per-batch `attn_row` scratch 后在 gi 循环外一次 `pl.assemble(attn_out, attn_row, [b, 0])` |
| **S2-C1** 融合 | ① | **① 但结构必须按 `mixed.py` 拆** | 不能把 cube（`pl.matmul`）+ vec（`pl.row_max`/`pl.exp`/`pl.cast`）塞进单个 `pl.at(CORE_GROUP)`；A2/A3 要求 mixed kernel 显式 `split=UP_DOWN`。`mixed.py:328-386` 的正确做法：sb 循环内 4 个独立 `pl.at(CORE_GROUP)` 子块（QK 纯 cube / Softmax 纯 vec / SV 纯 cube / update 纯 vec），通过外层 sb Vec-resident `oi/mi/li` 传递状态——仍然消除 5 个大 GM 中间张量，sim 信号仍在 |

**当前 `qwen3_32b_decode_scope2_opt.py` 已修正为**：S2-A4 + S2-C1（按 mixed.py 结构）。S2-A7/A8/A5 回滚到 per-head + attn_row scratch。

**对后续优化的启示**：
1. A2/A3 `pl.slice` 需直接作用于 GM tensor；任何"先 load/reshape 再二次 slice"的模式都会触发 textract-on-Vec 失败。批量向量化只能在 GM layout 允许一次连续切片 `[N, D]` 的前提下直接做（本例中 k_proj/q_proj 的 half-dim 在各 head 间不连续，无法单次 GM slice `[num_heads, half_dim]`）。
2. Opaque 层的跨 core-type 融合**必须**拆成多个 `pl.at(CORE_GROUP)` 子块（各自纯 cube 或纯 vec），或整体用 `pl.auto_incore(split=UP_DOWN)` 包裹（S2-B1 路径）。
3. `pl.parallel(gi)` 下对 `attn_out` 的直写未经验证；保守做法用 `attn_row` scratch。

## 5. 模拟器可观测信号速查表

| 优化 | Task count | Orch duration | Kernel count | GM 分配 | 备注 |
|---|---|---|---|---|---|
| S2-A1 | ↔ / ↑(实例复制) | ↓ 关键路径 | ↔ | 可能 ↑ | sim 看任务拓扑，wall time 不可信 |
| S2-A3 | ↓ `2×ctx_blocks×groups×batch` | ↓↓ | ↓ 2 | ↓ `all_raw_scores` 256 KB | 最强 sim 信号之一 |
| S2-A4 | ↓ 120 | ↓ | ↓ 1 | ↓ 512 KB | 立竿见影 |
| S2-A5 | ↓ 16 | ↓ | ↔ | ↓ ~256 KB | 小而确定 |
| **S2-C1** | ↓↓ | ↓↓↓ | ↓ 3–4 | ↓↓ ~770 KB/group × 128 | **最强 sim 信号** |
| S2-C2 | ↓↓ | ↓↓ 关键路径 | ↓ 3–4 | ↔ | 并行度 8 → 128 |
| S2-C3 | ↓ ~1920 Q-pad assemble | ↓ | ↔ | ↔ | 主要看 `Avg Exec` |
| S2-A11 | ↔ / ↑ | ↔ | ↔ | ↔ | 纯 profiling |
| S2-A12 | ↓ ctx_blocks | ↓ | ↔ | ↔ | 需精度回归 |
| S2-B1 | ↑（拆 AIC/AIV） | ↓ 关键路径 | ↑ | ↔ | swimlane 交叠 |
| S2-B3 | ↔ | ↔ | ↔ | ↔ | sim 只看 load count，带宽收益靠真机 |
| S2-B4 | ↔ | ↓ 关键路径 | ↔ | ↔ | swimlane 交叠 |

## 6. 实现次序建议

每步都必须独立通过 rtol=1e-3（含 `--golden-data` 固定输入）回归：

**只改 pypto-lib 即可完成的顺序（①类为主，②类带回退）**：

1. **P0 独立项** [①]：S2-A4（删 q-pad init）→ S2-A5（删 attn_row）。低风险 + 立即可量化 orchestrator 任务数削减。
2. **P1 核融合** [①]：S2-A3（仅合并 QK+Softmax+SV，保留 Stage 5 不变），验证正确性与任务数变化。
3. **P1 单次 pass** [①]：在 2 通过后引入 S2-A6，形成 **S2-C1** 完整 Flash-Attention 形态。
4. **P1 batch parallel** [②]：叠加 S2-A1 形成 **S2-C2**；此时 sim 上 orchestrator 峰值并发从 8 升到 128。**若 pypto 冲突检测拒绝则参见 §4.5 的回退路径**。
5. **P2 Stage 1 向量化** [A7/A8: ① · A9: ②]：**S2-C3**（三项一次做更直观，与 1–4 步正交可并行演进）。A9 如受限可只做 A7+A8。
6. **P2 TileDSL 可选** [②]：若 C2/C3 仍不够，引入 S2-B1（auto_incore UP_DOWN）；失败则回退到 S2-B2 + 手工跨核（①）。
7. **P3 实验性** [①]：S2-A12 去 round-trip、S2-A11 chunk 调优、S2-B3 显式 L1 驻留。

**需 pypto 底层协作才能落地的项**（不在 pypto-lib 修改范围内）：

- **[③] S2-B4** K/V double-buffer 预取：需新增 `pl.async_load` / `pl.load(..., double_buffer=True)` Python API + IR op + lowering 到 tprefetch ISA 指令；**若仅改 pypto-lib 则跳过本项**。若发现 pypto 已有 `DoubleBufferInsertion` pass（通过 `passes_dump/` 检查），则相当于 ① 白给，但需先验证。

## 7. 验证与风险护栏

- **精度基线**：每迭代必须跑 `--golden-data $GOLDEN` 固定输入比对，`rtol=atol=1e-3`。
- **边界样本**：覆盖 `ctx_len = 1 / 64 / 128 / 4095 / 4096`，确保 S2-A3/A6 的 `valid_len` padding 路径正确。
- **性能基线**：每步独立 `build_output/Qwen3Scope2_opt_<tag>/` 目录（通过 `RunConfig.compile.output_dir`），`perf_report.md` + `memory_after_AllocateMemoryAddr.txt` 记入 `opt-results/S2-*.md`。
- **文档规范**：遵循现有 `opt-records/S*.md` + `opt-results/S*.md` 双文件结构（参考 S1-1、S3-1）。

## 8. 使用说明

- 本文档仅为**机会盘点**，不含代码改动。
- 每条 `S2-Ax` / `S2-Bx` / `S2-Cx` 独立生成 `opt-records/S2-*.md` + `opt-results/S2-*.md`。
- 组合 bundle（S2-C1/C2/C3）按其依赖顺序分拆到对应的子记录。
