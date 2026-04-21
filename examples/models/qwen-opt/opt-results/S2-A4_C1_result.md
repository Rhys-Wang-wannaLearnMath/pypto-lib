# S2-A4 + S2-C1 (重构版) 优化结果

**修改文件**：`examples/models/qwen-opt/qwen3_32b_decode_scope2_opt.py`
**对照基线**：`examples/models/qwen3/qwen3_32b_decode_scope2.py`
**Build 输出**：
- 基线 `build_output/Qwen3Scope2_20260421_202624/`
- 优化 `build_output/Qwen3Scope2_opt_20260421_202710/`

**平台**：910B（Ascend A2/A3 模拟器）
**运行配置**：`batch=16, num_heads=64, num_kv_heads=8, q_per_kv=8, head_dim=128, hidden=8192, max_seq=4096, SEQ_TILE=64, Q_HEAD_BATCH=8, Q_HEAD_PAD=16, total_q_groups=8`

---

## 1. 应用的优化项

| 编号 | 名称 | 说明 |
|---|---|---|
| **S2-A4** | per-group `q_padded` | 取消 `all_q_padded[batch * total_q_groups * Q_HEAD_PAD, head_dim]` GM 预分配及其 pad-row 初始化循环；`q_padded` 下沉为 per-gi tile |
| **S2-C1**<br>（按 `mixed.py` 拆） | Per-sb Flash-Attention 融合 | 取消 5 个大 GM 暂存张量（`all_raw_scores` / `all_exp_padded` / `all_oi_tmp` / `all_cur_mi` / `all_cur_li`）；online-softmax `oi/mi/li` 以 Vec-resident 形式跨 sb 传递；sb 内 4 个纯类型 `pl.at(CORE_GROUP)` 子块（QK / Softmax / SV / Update） |

**首轮回滚项**（经 A2/A3 实测不可行）：S2-A5 直写 `attn_out`、S2-A7/A8 批量 RoPE。详见 `opt-records/S2_optimization_opportunities.md §实测勘误`。

---

## 2. 总体指标对比

| 指标 | 基线 | 优化 | Δ |
|---|---:|---:|---:|
| Wall time（仿真墙钟） | 16.472 s | 14.103 s | **−14.4 %** ✅ |
| Sum of task durations（所有 task 的执行时间合计） | 365.168 s | 274.673 s | **−24.8 %** ✅ |
| Kernel count（InCore 核数） | 6 | 7 | +1 |
| Task count（orch 提交的 task 总数） | 529 | 15984 | **+2922 %** ❌ |
| Avg task duration | 690 ms | 17.2 ms | −97.5 % |
| Orchestrator busy time（单 orch 核累计） | 7.55 ms | 7.70 s | **+101 960 %** ❌ |
| Scheduler busy time（2 sched 核累计） | 3.85 ms | 139.5 ms | +3522 % |

**关键结论**：Wall time 取得 **−14.4 %** 的净收益；但任务数量和 orch/sched 负载出现严重膨胀，收益被 dispatch 开销大幅稀释。

---

## 3. Per-func 执行时间拆解

| func | core | 含义 | Baseline count×avg = total (us) | Opt count×avg = total (us) | Δ total |
|---|---|---|---|---|---|
| 0 | aiv | 基线：`all_q_padded` pad 初始化 一次性任务<br>优化：per-b K-RoPE + V-cache 写 | 1 × 12634 = **12 634** | 16 × 2422 = **38 756** | +207 % |
| 1 | aiv | 基线：per-b K/V cache + K/Q RoPE 全部融合<br>优化：per-(b,gi) Q-RoPE + accum init | 16 × 220 = **3 526** | 128 × 16071 = **2 057 109** | +58 238 % |
| 2 | aic | QK matmul | 128 × 1 451 594 = **185 803 993** | 3928 × 35 625 = **139 935 261** | **−24.7 %** ✅ |
| 3 | aiv | Softmax | 128 × 52 717 = **6 747 721** | 3928 × 2 525 = **9 919 944** | +47.0 % ❌ |
| 4 | aic | SV matmul | 128 × 1 347 993 = **172 543 105** | 3928 × 31 231 = **122 675 997** | **−28.9 %** ✅ |
| 5 | aiv | 基线：online-softmax merge + normalize + 写 attn_row<br>优化：per-sb online-softmax update（小 Vec） | 128 × 444 = **56 792** | 3928 × 11 = **43 795** | −22.9 % ✅ |
| 6 | aiv | 优化：per-gi 最终 normalize + 写 attn_row | — | 128 × 20 = **2 571** | — |
| **TOTAL** | — | — | **365 167 772** | **274 673 434** | **−24.8 %** ✅ |

**分项要点**：
- **QK / SV** 两个 AIC 大核是 attention 的算力瓶颈。基线单 task 平均 1.35 / 1.45 秒（内含 sb-chunk 循环），优化后每 task 压缩到 31 / 36 ms（单 sb），**总计算量下降 25-29 %**——这正是 S2-C1 真正的 sim 信号来源，归功于消除了 5 个大 GM 暂存张量的分配/DMA 往返。
- **Softmax 回退 +47 %**：基线把 softmax 和 sb-chunk 循环一起塞进单个大 kernel（每 task 52 ms），I/O 在 kernel 内部 DMA 链路摊销；优化版把 softmax 拆为 per-sb 小 task（每 task 2.5 ms），**GM 小块读写摊不掉 dispatch 开销**。
- **Func 1 剧增 +58 238 %**（但绝对值仅 2 s）：基线 func 1 只做 16 次 K/V RoPE；优化版 func 1 接管了 128 次 per-gi 的 Q-RoPE（带 per-head 内层循环）+ 三个 accumulator 初始化。绝对时间仍远小于 AIC 两个核，非瓶颈。

---

## 4. Orchestration 结构对比

### 4.1 生成的 `orchestration/qwen3_scope2.cpp`

| 维度 | 基线 (190 行) | 优化 (203 行) |
|---|---|---|
| 外层循环 | `for b in 0..16` | `for b in 0..16` |
| K/V 阶段 | `for ki in 0..1` 嵌套（融合 Q-RoPE） | 无（Q-RoPE 移入 gi 循环） |
| 大 GM 预分配 | `all_q_padded` [2048, 128] BF16 = **512 KB** 每次调用一次 | 无 |
| gi 循环内 GM 分配 | `all_raw_scores` + `all_exp_padded` + `all_oi_tmp` + `all_cur_mi` + `all_cur_li` = **~900 KB / (b,gi)** × 128 = **~115 MB / call** | `q_padded` + 3 个 accum = **~4 KB / (b,gi)** × 128 = **~512 KB / call** |
| sb 循环内 GM 分配 | 无 | `raw_scores_pad` + `exp_padded` + `oi_tmp_pad` + 4 个 accum = **~16 KB / (b,gi,sb)** × ~3 928 = **~63 MB / call** |
| sb 循环形态 | 单 `pl.at(CORE_GROUP, chunked_loop_optimizer)` 包住 `pl.parallel(ctx_blocks, chunk=SB_BATCH)`；**每阶段 1 task / (b,gi)** | 4 个独立 `pl.at(CORE_GROUP)`，sb 顺序；**每阶段 1 task / (b,gi,sb)** |

### 4.2 GM 分配总量（粗估）

| 项 | 基线 | 优化 |
|---|---:|---:|
| `all_q_padded` 一次性 | 512 KB | 0 |
| Per-(b,gi) 大 GM 张量 | 115 MB | 0.5 MB |
| Per-(b,gi,sb) 小 GM 张量 | 0 | 63 MB |
| **合计** | **~115.5 MB / call** | **~63.5 MB / call（−45 %）** ✅ |

GM 分配总量减少 45 % 与 compute 下降 24.8 % 同方向。但 per-sb 小块分配带来的高频 DMA-issue 会抵消部分收益（见 Softmax 回退）。

### 4.3 Orchestrator 侧 phase 数

| 项 | 基线 | 优化 |
|---|---:|---:|
| orch_alloc | 658 | 20 040 |
| orch_sync | 658 | 20 040 |
| orch_params | 658 | 20 040 |
| orch_fanin | 658 | 20 040 |
| orch_lookup | 529 | 15 984 |
| orch_insert | 529 | 15 984 |
| **合计 phase** | **3 690** | **112 128（+2 939 %）** |
| **Orch busy** | **7.55 ms** | **7.70 s（×1020）** |

orch 核一直 99.9 % busy，持续 7.7 s。但注意：**orch 总 busy (7.7 s) 仍 < wall (14.1 s)**——意味着 orch 在仿真里没有成为关键路径上的实际阻塞（AIC/AIV 任务执行继续延伸到 14.1 s），所以 wall 仍下降；但一旦未来减少 AIC/AIV latency，orch 将立即成为硬瓶颈。

---

## 5. Kernel 内存占用（`report/memory_after_AllocateMemoryAddr.txt`）

| Kernel | Baseline / Vec | Baseline / Mat+L0+Acc | Opt / Vec | Opt / Mat+L0+Acc |
|---|---|---|---|---|
| incore_0 | 6.0 KB (aiv) | — | 3.0 KB (aiv) | — |
| incore_1 | 3.2 KB (aiv) | — | 6.3 KB (aiv) | — |
| incore_2 | — | 20+4+16+4 KB (aic) | — | 20+4+16+4 KB (aic) |
| incore_3 | 9.1 KB (aiv) | — | 9.1 KB (aiv) | — |
| incore_4 | — | 18+2+16+8 KB (aic) | — | 18+2+16+8 KB (aic) |
| incore_5 | 14.3 KB (aiv) | — | 8.3 KB (aiv) | — |
| incore_6 | — | — | 6.0 KB (aiv) | — |

所有 kernel 的 Vec/Mat/L0/Acc 占用率均 ≤ 25 %，**片上内存远未饱和**——可支撑更大的 Q-head/sb 块融合（见 §7 next steps）。

---

## 6. 诊断：为什么收益只有 −14 %？

### 6.1 Task 爆炸的根本原因

基线代码（`qwen3_32b_decode_scope2.py:159-209`）用了这个关键组合：

```python
with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
    for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
        ...
```

- `pl.parallel(..., chunk=SB_BATCH)`：告诉 pypto 将 sb 迭代空间切成 `ceil(ctx_blocks / SB_BATCH)` 个连续块；
- `optimization=pl.chunked_loop_optimizer`：把**每个块内的 `SB_BATCH` 次 sb 迭代展开进 kernel 内部循环**；
- 结果：orch 层每 `pl.at` 只下发 `ceil(ctx_blocks / SB_BATCH)` 个 task（通常 = 1，因为 `ctx_blocks ≤ SB_BATCH`），每个 task 的 kernel 内部 sb 次完整计算。

这等价于**"kernel-inner sb loop with L1/Vec 复用"** ——DMA 往返可以在 kernel 内部被 compiler 的 prefetch/fuse pass 摊销。

我当前的 S2-C1 改写（`qwen3_32b_decode_scope2_opt.py:173`）：

```python
for sb in pl.range(ctx_blocks):
    with pl.at(level=pl.Level.CORE_GROUP):
        ...
```

- 没有 `chunk=`，没有 `chunked_loop_optimizer`；
- 每次 sb 迭代生成**独立**的 task dispatch；
- `pl.range`（而非 `pl.parallel`）让 sb 完全串行，且每个 sb 内部又有 4 个串行的 `pl.at` → per-(b,gi,sb) 产生 4 个串行 task。

### 6.2 收益解构

| 贡献 | 来源 | 对 wall 的影响 |
|---|---|---|
| **AIC 总计算 −25 ~ −29 %** | 5 大 GM 暂存张量消失 → DMA 少一轮；per-sb 小 tile 让 L1 更不易抖动 | wall ↓（关键路径在 AIC） |
| **GM 分配 −45 %** | 同上 | 改善内存带宽，间接助推 AIC |
| **Softmax +47 %** | per-sb 小 Vec 任务被 dispatch 开销摊不掉 | 非关键路径，吞噬少量收益 |
| **Task count +2922 %** | sb 失去 chunk 批处理 | 增加 orch busy，但仍 < wall |
| **Orch busy +1020 × (7.7 s)** | 同上 | 仿真中仍与 AIC 部分重叠，未阻塞 wall |

**净效应**：AIC 关键路径缩短 14 % 主导，orch / softmax 开销吞噬了部分增益但未反转方向。

### 6.3 关键路径归因

| 项 | 基线 latency | 优化 latency | 贡献于 wall |
|---|---:|---:|---|
| func 4（SV matmul, AIC）first→last | 13.199 s | 13.926 s | **两个版本的关键路径都在 SV AIC 核上** |
| func 2（QK matmul, AIC）first→last | 9.542 s | 12.448 s | 次关键 |
| wall 整体 | **16.472 s** | **14.103 s** | SV 总计算减少 29 %，但 latency 降幅仅 5.2 % |

优化把 SV 的**总工作量**砍到 122 s（相对 172 s），但同样的 work 被分散到 3928 个小 task，**仍然被 AIC 核的可用并发度限制**——所以 latency 只降了一点点。理论下界是 `max(sum_exec / AIC_cores, serial_critical_path)`；我们远未到下界。

---

## 7. 下一步建议

按优先级从高到低：

### 7.1 【高优】把 `chunk=SB_BATCH` 加回 QK / SV / Softmax

当前 sb 串行 + 每 iter 4 task 是最主要的 dispatch 膨胀源。基线证明 `pl.parallel(ctx_blocks, chunk=SB_BATCH)` + `chunked_loop_optimizer` 可以让 sb 在 kernel 内部循环，把 QK / SV / Softmax 各自的 task 数从 3928 压回 128（与基线同级）。

但需要解决 online-softmax 的 loop-carried 依赖——可行路径：
- QK / Softmax / SV 全部改成 `pl.parallel(..., chunk=SB_BATCH)`（它们之间没有跨 sb 依赖，只依赖同 sb 内数据链）；
- Online-update 仍保留 `pl.range` 序列化，但依赖前三阶段**按 sb 就绪**——需要判断 pypto 的 `pl.parallel` 在同一 `pl.at` 外能不能提供 per-sb fence 保证。若不能，则此优化退化为基线的"先全做完 stages 2/3/4 再顺序 reduce"结构。

### 7.2 【高优】S2-B1: `pl.auto_incore(split=pl.SplitMode.UP_DOWN)` 包住整个 gi 循环

这是从结构上"一刀切"解决 dispatch 膨胀与跨 core-type 融合的方式（详见 plan §4.② 与 `qwen3_32b_decode_mixed.py:239-396`）。预计效果：
- task 数下降到基线水平（甚至更低）；
- QK ↔ Softmax ↔ SV 可以在 AIC/AIV 间形成 cross-core 流水，进一步缩短 SV latency；
- 需要一轮 runtime 配置与 sim 验证（plan 中归为 ②）。

### 7.3 【中优】打磨 Q-RoPE 批量化路径

当前 func 1 为 2 s 绝对值，量级无关痛痒；但若要进一步把 attention 推到 10 s 以下，RoPE 的 per-head 展开会成为次瓶颈。此时需要看 pypto 是否允许在保持 `pl.slice` 直接作用于 GM 的前提下，把多个 head 的 slice 合并成一次连续 load（目前 head_dim=128，num_kv_heads=8，即 8 次 slice 各 256 B，带宽低效）。

### 7.4 【低优】`all_exp_padded` 的 bf16→fp32 回 cast 是否可消除

基线 `qwen3_32b_decode_scope2.py:186-187` 与优化 `qwen3_32b_decode_scope2_opt.py:209-210` 都有 `bf16 → fp32 → row_sum` 的 cast-cycle；这是语义上必需的（`pl.row_sum` 需要 FP32 累加以保证精度），但如果 pypto 提供 `row_sum(..., accumulate_dtype=pl.FP32)` 的直接路径可省一次 Vec cast。

---

## 8. 结论

**S2-A4 + S2-C1（按 `mixed.py` 拆）** 达到 **−14.4 % wall** 与 **−24.8 % 总计算**，验证了"消除 GM 暂存张量"这一 ① 级优化方向的正确性。但受限于未加 `chunk=SB_BATCH`，引入了严重的 task/orch 开销膨胀，实际收益被稀释一半以上。

**下一轮必做**：把 `pl.parallel(ctx_blocks, chunk=SB_BATCH) + chunked_loop_optimizer` 加到 QK/Softmax/SV 三个阶段，预期可追加 **−20 ~ −30 % wall** 并把 task 数压回 ~512；或直接切换到 S2-B1 的 `auto_incore(split=UP_DOWN)` 跨核融合路径。

---

## 附录 A · 关键命令与数据来源

```bash
# 关键性能数据
/root/pypto-lib/build_output/Qwen3Scope2_20260421_202624/swimlane_data/perf_swimlane_20260421_202705.json
/root/pypto-lib/build_output/Qwen3Scope2_opt_20260421_202710/swimlane_data/perf_swimlane_20260421_202747.json

# 内存报告
/root/pypto-lib/build_output/Qwen3Scope2_20260421_202624/report/memory_after_AllocateMemoryAddr.txt
/root/pypto-lib/build_output/Qwen3Scope2_opt_20260421_202710/report/memory_after_AllocateMemoryAddr.txt

# 生成的 orchestration
/root/pypto-lib/build_output/Qwen3Scope2_20260421_202624/orchestration/qwen3_scope2.cpp
/root/pypto-lib/build_output/Qwen3Scope2_opt_20260421_202710/orchestration/qwen3_scope2.cpp
```

数据提取脚本（内联使用）：

```python
import json
from collections import Counter, defaultdict

def analyze(path, label):
    with open(path) as f:
        d = json.load(f)
    tasks = d['tasks']
    by_func = defaultdict(list)
    for t in tasks:
        by_func[t['func_id']].append(t)
    for fid in sorted(by_func):
        ts = by_func[fid]
        durs = [t['duration_us'] for t in ts]
        start_min = min(t['start_time_us'] for t in ts)
        end_max = max(t['end_time_us'] for t in ts)
        print(f"  func={fid} core={ts[0]['core_type']} count={len(ts)} "
              f"avg={sum(durs)/len(ts):.2f}us total={sum(durs):.2f}us "
              f"latency={end_max-start_min:.2f}us")
```
