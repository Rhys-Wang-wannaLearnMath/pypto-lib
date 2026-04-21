# Round 2 优化结果 —— **Round 1 的回归**

**修改文件**：`examples/models/qwen-opt/qwen3_32b_decode_scope2_opt.py`
**Build 输出**：
- Baseline `build_output/Qwen3Scope2_20260421_202624/`
- Round 1 `build_output/Qwen3Scope2_opt_20260421_202710/`
- **Round 2 `build_output/Qwen3Scope2_opt_20260421_205434/`**（本次）

**备份文件**：
- `qwen3_32b_decode_scope2_opt.py.round1`（Round 1 per-sb 融合版）
- `qwen3_32b_decode_scope2_opt.py.bak`（Round 0 原始 opt）

---

## TL;DR

**Round 2 在 wall time 上是 Round 1 的回归**，编译成功但仿真结果显示：

| 维度 | vs Baseline | vs Round 1 |
|---|---|---|
| Wall time | **−0.1 %**（16.47 → 16.45 s，基本持平） | **+16.6 % 回归** ❌ |
| Sum compute | **−2.9 %**（365.17 → 354.53 s） | **+29.1 % 回归** ❌ |
| Task count | +24 % | −95.9 % ✓ |
| Orch busy | +17 329 % | −83 % ✓ |

预言的"orch 是 wall 瓶颈"是**错的**。真正的瓶颈一直是 AIC 计算（见 §5 诊断）。

**结论**：应回滚到 Round 1 结构，下一轮从另一维度（`auto_incore(split=UP_DOWN)` 或更激进的 tile-level 重写）切入。

---

## 1. 三方对比总览

| 指标 | Baseline | Round 1（per-sb fused） | **Round 2（per-gi staging）** | R2 vs R1 |
|---|---:|---:|---:|---:|
| Wall time | 16.472 s | **14.103 s** | 16.449 s | **+2.346 s** |
| Sum of task durations | 365.168 s | 274.673 s | 354.529 s | +79.856 s |
| Kernel count | 6 | 7 | 6 | −1 |
| Task count | 529 | 15 984 | 656 | −15 328 |
| AIC task count | 256 | 7 856 | 256 | −7 600 |
| AIV task count | 273 | 8 128 | 400 | −7 728 |
| Avg task duration | 690 ms | 17.2 ms | 540 ms | +523 ms |
| Orch busy (single core) | 7.55 ms | 7.70 s | **1.32 s** | **−6.38 s** |
| Sched busy (sum 2 cores) | 3.85 ms | 139.5 ms | 5.85 ms | −133.6 ms |

**关键观察**：Round 2 几乎收敛到 baseline 的执行特征（kernel 数、task 数、orch 级都回到 baseline 量级），但付出了与 baseline 相当的 compute 代价，也就丢掉了 Round 1 的 `−24.8 %` compute 红利。

---

## 2. Per-func 三方对比

### 2.1 任务数量

| func | core | Baseline | Round 1 | Round 2 | R2 vs Baseline |
|---|---|---:|---:|---:|---|
| 0 | aiv | 1 | 16 | 16 | +15 |
| 1 | aiv | 16 | 128 | 128 | +112 |
| 2 | aic | 128 | 3 928 | **128** | ↔ |
| 3 | aiv | 128 | 3 928 | **128** | ↔ |
| 4 | aic | 128 | 3 928 | **128** | ↔ |
| 5 | aiv | 128 | 3 928 | **128** | ↔ |
| 6 | aiv | — | 128 | — | — |
| **合计** | — | **529** | **15 984** | **656** | +127 |

Stages 2/3/4/5 每阶段任务数成功从 3928 压回 128（与 baseline 一致）——chunk 化生效。
新增的 +127 tasks 来自：Stage 2a（per-gi Q-RoPE，128 tasks）被独立出来，没有像 baseline 那样融合进 Stage 1。

### 2.2 每 task 执行时间（avg）

| func | Baseline | Round 1 | Round 2 | 说明 |
|---|---:|---:|---:|---|
| 0 | 12.63 ms | 2.42 ms | 7.52 ms | Stage 1: K/V cache + K-RoPE (+ baseline 含 Q-pad init) |
| 1 | 0.22 ms | 16.07 ms | 36.28 ms | Stage 2a: Q-RoPE（baseline 融合进 0；R1/R2 独立 per-gi） |
| 2 | **1 451.59 ms** | 35.63 ms | **1 376.35 ms** | QK matmul（R2 恢复"每 gi 单 task 内部循环 sb"） |
| 3 | 52.72 ms | 2.53 ms | 20.91 ms | Softmax |
| 4 | **1 347.99 ms** | 31.23 ms | **1 334.86 ms** | SV matmul（R2 恢复 baseline 粒度） |
| 5 | 0.44 ms | 0.011 ms | 0.42 ms | Online merge（R2 恢复 baseline 粒度） |
| 6 | — | 0.020 ms | — | R1 独有的 normalize + writeback |

**要点**：
- Round 2 的 QK / SV / Softmax 每任务 avg 回到 baseline 量级（**证实 chunk=SB_BATCH + chunked_loop_optimizer 生效**）
- 但 QK/SV 的 **total compute** 也回到 baseline 量级（QK 176.2s vs baseline 185.8s；SV 170.9s vs baseline 172.5s）——**Round 1 的 25-29 % 省出来的计算被 staging DMA 又吃回去了**

### 2.3 每 func 总计算（total_us）

| func | Baseline | Round 1 | Round 2 | R2 vs Baseline | R2 vs R1 |
|---|---:|---:|---:|---:|---:|
| 0 | 12 634 | 38 756 | 120 372 | **+853 %** ❌ | +210 % |
| 1 | 3 526 | 2 057 109 | 4 644 098 | **+131 613 %** ❌ | +126 % |
| 2 (QK, AIC) | 185 804 k | 139 935 k | **176 173 k** | **−5.2 %** | **+25.9 %** ❌ |
| 3 (Softmax) | 6 748 k | 9 920 k | 2 676 k | **−60 %** ✓ | −73 % |
| 4 (SV, AIC) | 172 543 k | 122 676 k | **170 862 k** | **−1.0 %** | **+39.3 %** ❌ |
| 5 (Merge) | 56 792 | 43 795 | 53 427 | −5.9 % | +22.0 % |
| 6 | — | 2 571 | — | — | — |
| **合计** | **365 168 k** | **274 673 k** | **354 529 k** | **−2.9 %** | **+29.1 %** ❌ |

**致命观察**：func 2（QK AIC）的 total 在 Round 2 **回升到 176 s**，几乎完全回到 baseline 的 185 s。func 4（SV AIC）同样。Round 1 的 −45/−50 s 计算红利**完全被 staging 吃掉**。

**Q-RoPE 膨胀（func 1）**：baseline 把 Q-RoPE 融进 Stage 1（220 µs / task × 16 task = 3.5 ms 总耗时）。Round 2 独立 per-gi（36 ms / task × 128 task = 4.6 s 总耗时）——**1 300 × 膨胀**。这是 S2-A4 的隐性代价：Q-RoPE 从 AIV vector tile 的一次性 fused ROP 退化为 per-gi 独立任务，每任务附加了 kernel launch + GM alloc + load/store overhead。

### 2.4 每 func 延迟（latency，first→last span）

| func | Baseline | Round 1 | Round 2 | 说明 |
|---|---:|---:|---:|---|
| 2 (QK, AIC) | 9.54 s | 12.45 s | **10.15 s** | R2 AIC 利用率优于 R1（128 大 task 在 24 core 上排得更紧） |
| 4 (SV, AIC) | 13.20 s | 13.93 s | **12.65 s** | 同上 |
| 3 (Softmax) | 9.25 s | 12.43 s | 9.83 s | 类似 |
| 5 (Merge) | 10.12 s | 13.84 s | 10.51 s | 类似 |
| **wall** | **16.47 s** | **14.10 s** | **16.45 s** | — |

**观察**：Round 2 的**各 func 延迟确实比 Round 1 短**（SV 12.65 vs 13.93，−1.28 s）——说明 chunk 化 + 大粒度 task 给 AIC 更好的调度填充。但**总 compute 更多**，所以 wall 反而更长。

---

## 3. AIC 利用率分析

| 版本 | func 4 total | func 4 latency | AIC 平均并行度 |
|---|---:|---:|---:|
| Baseline | 172.5 s | 13.20 s | **13.07 cores** / 24 |
| Round 1 | 122.7 s | 13.93 s | **8.81 cores** / 24（48 %） |
| Round 2 | 170.9 s | 12.65 s | **13.51 cores** / 24（**56 %**，最好） |

**Round 2 AIC 利用率最高**，但同时 AIC 总工作量也最大。杠杆方向错了：我们用**更多的 AIC 工作**换来了**略少的 AIC 延迟**。

---

## 4. 内存使用（`memory_after_AllocateMemoryAddr.txt`）

Round 2 和 Baseline 完全同构：

| Kernel | R2 Vec | R2 Mat/L0/Acc |
|---|---|---|
| incore_0 | 3.0 KB | — |
| incore_1 | 2.2 KB | — |
| incore_2 | — | 20+4+16+4 KB（= baseline） |
| incore_3 | 9.1 KB（= baseline） | — |
| incore_4 | — | 18+2+16+8 KB（= baseline） |
| incore_5 | 14.3 KB（= baseline） | — |

片上资源使用率仍 ≤ 25 %，与 baseline 一致。

---

## 5. 根本原因诊断 —— 我预言错了

### 5.1 Round 1 结果表中的"orch 是瓶颈"推论是**错的**

Round 1 的数据：
```
wall = 14.10 s   orch_busy = 7.70 s   sum_compute = 274.67 s
func 4 (SV AIC) latency = 13.93 s ≈ wall
```

我当时推断：
> orch busy 7.7s，约 wall 的 55 %；若能降低 orch，wall 会进一步下降。

**真相**：orch busy 只占单个 orch 核的时间，而 wall 上限是 `max(任意 core 的忙时)`。AIC 在 func 4 期间连续忙了 13.93 s（≈ wall），**AIC 才是关键路径**。orch 只要 < wall 就不阻塞，无论是 7.7 s 还是 1.3 s。

Round 2 直接验证了这一点：orch busy 从 7.70 s 降到 1.32 s，**wall 反而从 14.10 s 升到 16.45 s**——orch 省下的时间完全没有传导到 wall。

### 5.2 Round 1 的 `−24.8 %` compute 红利**真实且有效**

Round 1 相对 baseline：
```
sum_compute 365.17 → 274.67 s     (−90.5 s)
wall         16.47 →  14.10 s     (−2.37 s)
```

90 s compute ÷ 2.37 s wall ≈ **38× 等效并行度**。Round 1 并行实际用了约 38 个 core 来吸收 compute 节省——这正是 ~24 AIC + ~25 AIV 的并行度上限。**compute 节省的 2.37 s 就是 Round 1 wall 相对 baseline 的全部收益**。

Round 2 把这 90 s compute 又加了 80 s 回来（274 → 354），对应 wall +2.35 s——**数量级完全匹配**，证实 compute 就是 wall 的一阶决定因素。

### 5.3 Round 2 为什么 compute 会涨回去？

Staging tensor 带来的 GM traffic：

| GM 读写 | Round 1 | Round 2 |
|---|---|---|
| QK 输出 | per-sb Vec-local（0 GM） | per-sb 写 `all_raw_scores`（~256 KB / gi） |
| Softmax 输入 | per-sb Vec-local | per-sb 读 `all_raw_scores` |
| Softmax 输出 | per-sb Vec-local | per-sb 写 `all_exp_padded` + `all_cur_mi` + `all_cur_li` |
| SV 输入 | per-sb Vec-local | per-sb 读 `all_exp_padded` |
| SV 输出 | per-sb Vec-local | per-sb 写 `all_oi_tmp` |
| Merge 输入 | loop-carried Vec | 读 `all_oi_tmp` + `all_cur_mi` + `all_cur_li` |

每 sb 迭代 Round 2 多了 ~1.4 MB GM 读写（Q_HEAD_PAD=16, SEQ_TILE=64, head_dim=128, FP32+BF16 混合）。× 31 sb × 128 gi ≈ **5.5 GB GM traffic / call**，这就是 AIC 多出的 ~50 s compute 去向。

### 5.4 Round 2 的**真正**收益

| 维度 | 收益 |
|---|---|
| Orch busy 回到 ms 级 | ✓ 但不影响 wall |
| AIC 利用率 +55 % | ✓ 但被总量 +29 % 抵消 |
| per-func latency 略降 | ✓ 但 func 间关键路径不变 |
| Task 数量可控 | ✓ 对未来的调度器健壮性有好处 |

**真正对 wall 有利的只有 AIC 利用率**，但不足以抵消 compute 上升。

---

## 6. 认知更新：Round 1 的"负面信号"解读纠正

在 Round 1 结果 doc §7.1 我写过：

> 【高优】把 `chunk=SB_BATCH` 加回 QK / SV / Softmax
> 当前 sb 串行 + 每 iter 4 task 是最主要的 dispatch 膨胀源……预期可追加 −20 ~ −30 % wall。

**修正**：`chunk=SB_BATCH` 本身没错，错在**同时恢复了 staging tensor**。正确的做法应该是：

> ✅ 保留 Round 1 的 per-sb Vec-local 数据流（无 GM staging）
> ✅ 同时把 sb 序列化 + 4 子块"折叠"进更大的 task 单元

两者兼得的唯一路径是 **S2-B1 `pl.auto_incore(split=pl.SplitMode.UP_DOWN)`**：让 pypto 自动把 QK-Softmax-SV-Update 的整个 sb 循环融合进一个跨 AIC/AIV 的 incore kernel，期间无 GM staging，同时只产生 1 task（per gi，与 baseline 同量级）。

---

## 7. 行动建议

### 7.1 **立刻回滚到 Round 1**

Round 1 在 wall 上优于 Round 2 **16.6 %**。作为中间稳定基线：

```bash
cp /root/pypto-lib/examples/models/qwen-opt/qwen3_32b_decode_scope2_opt.py.round1 \
   /root/pypto-lib/examples/models/qwen-opt/qwen3_32b_decode_scope2_opt.py
```

`Round 1 的 14.10 s wall 是当前已验证的最佳值`。

### 7.2 Round 3 候选方案（按优先级）

#### A. `pl.auto_incore(split=pl.SplitMode.UP_DOWN)` 包住整个 gi 循环（S2-B1）

```python
for gi in pl.parallel(total_q_groups):
    with pl.auto_incore(split=pl.SplitMode.UP_DOWN):
        # Q-RoPE → QK → Softmax → SV → online update → normalize → write attn_row
        ...
```

- **期望**：把 Round 1 的 4 个 sb 子块 + final normalize 折叠进单个跨核 incore kernel，无 GM staging + 仅 1-2 task/gi。理论上 **compute 保持 Round 1 水平（274 s），latency 接近 baseline 量级（task 数 ~128-256），wall 目标 10 s 以下**。
- **风险**：`auto_incore` 对 online-softmax 的 loop-carried 状态（`oi/mi/li`）是否能保持 Vec-resident 尚未验证；可能需要显式 `pl.at` 标注每段属于 AIC 还是 AIV。
- **回退**：若不行，退到 Round 1。

#### B. Round 1 + Stage 1 融合 Q-RoPE（减 1 kernel）

Round 2 揭示 `per-gi Q-RoPE` 付出 4.6 s 代价。若把 Q-RoPE 融回 Stage 1（像 baseline）：
- 代价：需要保留全局 `all_q_padded`（512 KB GM），放弃 S2-A4
- 收益：Q-RoPE 任务数 128 → 16，总耗时 4.6 s → 3 ms 级
- **但** Q-RoPE 非关键路径（func 1 latency = 1.6 s << wall 14-16 s），收益不会传导到 wall

结论：**B 收益低，不优先**。

#### C. 减少 num_heads 方向的 AIC 工作（算法级）

SV matmul 的 `[Q_HEAD_PAD=16, SEQ_TILE=64] × [SEQ_TILE=64, head_dim=128]` 内积维度只 64，AIC 的 Cube 单元按 `[16, 16, 16]` 基本块算，利用率天然不高。考虑：
- 合并多个 gi 的 Q：把 `Q_HEAD_BATCH` 从 8 提到 16（如果 num_heads 允许），减少 gi 循环次数；但改变了逻辑结构。
- 或调整 SEQ_TILE 从 64 → 128：让 AIC Cube 内积维度 × 2，单任务做更多工作。

这些需要重新推导 tiling 安全约束，属于 ② 级。

---

## 8. 结论

Round 2 **编译成功、功能正确、架构更接近 baseline**，但在最核心的 wall time 指标上**相对 Round 1 回归 16.6 %**。

**经验教训**：
1. 不要信任"orch busy 看起来很多"这种**次级**信号；必须用 per-func latency 对齐 wall 找**主要关键路径**。
2. GM staging tensor 的 DMA 成本**实实在在反映在 AIC compute 上**——Round 2 用 80 s AIC compute 回退证明了这一点。
3. Round 1 的 per-sb Vec-local 数据流**是正确的 compute 节省路径**，只是打包粒度（per-sb 4 task）偏小；正确的下一步是用 `auto_incore` 把整个链路打成一个大 task，保持 Vec-local 数据流的同时消除 dispatch overhead。

**建议**：回滚到 Round 1，Round 3 尝试 `pl.auto_incore(split=pl.SplitMode.UP_DOWN)`。

---

## 附录 A · 数据来源

```bash
# Round 2 性能数据
/root/pypto-lib/build_output/Qwen3Scope2_opt_20260421_205434/swimlane_data/perf_swimlane_20260421_205513.json
/root/pypto-lib/build_output/Qwen3Scope2_opt_20260421_205434/report/memory_after_AllocateMemoryAddr.txt
/root/pypto-lib/build_output/Qwen3Scope2_opt_20260421_205434/orchestration/qwen3_scope2.cpp

# 对比数据
/root/pypto-lib/build_output/Qwen3Scope2_20260421_202624/         # Baseline
/root/pypto-lib/build_output/Qwen3Scope2_opt_20260421_202710/     # Round 1

# 版本备份
/root/pypto-lib/examples/models/qwen-opt/qwen3_32b_decode_scope2_opt.py          # Round 2（当前）
/root/pypto-lib/examples/models/qwen-opt/qwen3_32b_decode_scope2_opt.py.round1   # Round 1（14.10 s 版）
/root/pypto-lib/examples/models/qwen-opt/qwen3_32b_decode_scope2_opt.py.bak      # Round 0
```

## 附录 B · 复现分析脚本

```python
import json
from collections import Counter, defaultdict

paths = {
    'BASELINE': '/root/pypto-lib/build_output/Qwen3Scope2_20260421_202624/swimlane_data/perf_swimlane_20260421_202705.json',
    'ROUND1':   '/root/pypto-lib/build_output/Qwen3Scope2_opt_20260421_202710/swimlane_data/perf_swimlane_20260421_202747.json',
    'ROUND2':   '/root/pypto-lib/build_output/Qwen3Scope2_opt_20260421_205434/swimlane_data/perf_swimlane_20260421_205513.json',
}

for label, path in paths.items():
    with open(path) as f:
        d = json.load(f)
    tasks = d['tasks']
    wall = max(t['end_time_us'] for t in tasks) - min(t['start_time_us'] for t in tasks)
    compute = sum(t['duration_us'] for t in tasks)
    by_core = Counter(t['core_type'] for t in tasks)
    print(f"{label}: tasks={len(tasks)}, wall={wall:.0f}us, "
          f"compute={compute:.0f}us, {dict(by_core)}")
```
