# Qwen3-32B Decode 优化计划（当前设备更新版）

> **更新日期**：2026-05-05  
> **适用代码**：`examples/models/qwen3/32b/` 与 `examples/models/qwen-opt/`  
> **当前设备**：8 × Ascend 910B4 / `IT21HMDA_Bin6x`，每卡 20 AICore，单卡有效 `block_dim=18`  
> **核心结论**：先完成环境与基线复测，再做 Scope 3 tile sweep；不要再把 Gate/Up 融合或显式 `pl.parallel` 作为首要优化。

---

## 0. 结论先行

本文件替代早期“机会清单式”计划。早期计划中部分候选已被 `qwen-opt/opt-results/`
下的实验结果证伪，且当前 checkout 的 `qwen3/32b` 代码已经和旧报告假设不完全一致。

### 0.1 本次更新后的优先级

| 优先级 | 编号 | 行动 | 原因 | 产出 |
|--------|------|------|------|------|
| **P0** | **E0** | 修复可运行环境 | 当前默认 Python=3.9.9，缺 `torch`/`pypto`，`PTOAS_ROOT`/`PTO_ISA_ROOT` 缺失 | 能跑 simulator 与真机 profiling |
| **P0** | **V0** | 修正验证入口与路径 | `run_comparison.sh`/`testing_guide.md` 仍引用旧路径和 `_opt.py` 文件名 | 一键 golden + profiling 可复现 |
| **P1** | **B0** | 在当前 checkout 上重建基线 | 旧报告中的 Scope 3 常量与当前 `qwen3/32b` 不一致 | 当前设备上的真实 span/task/core 分布 |
| **P1** | **S3-T** | Scope 3 tile sweep：`Q_OUT_CHUNK`/`MLP_OUT_CHUNK` | 当前 `scope3.py` 为 `Q=64, MLP=256`，`qwen-opt` 为 `Q=128, MLP=128`，不能直接沿用旧结论 | 选择本设备真实最优 tile |
| **P2** | **S2-B** | Scope 2 保持基线，仅做 ctx_len sweep | QK→Softmax→SV 融合受 acc→vec 编译器限制 | 明确短/长上下文下是否需要 batch/group 调整 |
| **P2** | **S1-B** | Scope 1 保持基线，仅复测 | `K_CHUNK=512` 降到 128 已出现回退；L0B 打满是约束也是当前最优点 | 避免重复无收益实验 |
| **P3** | **F0** | 框架级优化 | 需要 runtime 依赖分析或 incore acc→vec 支持 | 后续 PyPTO/runtime 任务 |

### 0.2 已从首要优化中移除的候选

| 候选 | 旧优先级 | 当前处理 | 依据 |
|------|----------|----------|------|
| Gate/Up 投影融合 | P0 | **移除为默认方案** | v3 结果显示保持 gate/up 分离可维持 runtime 自动 overlap；融合会破坏跨迭代流水或引入共享 inout 序列化风险 |
| QK+Softmax+SV per-sb 融合 | P1 | **降为框架任务** | 当前编译器不支持单个 incore 中 AIC `acc` 到 AIV `vec` 的数据连接 |
| Scope 1 `K_CHUNK=512→128` | P3 | **不再优先尝试** | 历史实测回退约 35.6%，RMSNorm/循环开销增加 |
| 对共享输出张量使用 chunked `pl.parallel` | P1 | **禁止作为默认方向** | runtime 会对共享 `inout` 做保守依赖分析，可能串行到单核 |

---

## 1. 当前设备实测信息

### 1.1 主机与 NPU

| 项 | 实测值 | 说明 |
|----|--------|------|
| CPU 架构 | `aarch64` | `lscpu` |
| CPU 型号 | Kunpeng-920 | 4 socket × 48 cores |
| CPU 数量 | 192 | 8 NUMA nodes，`0-191` |
| 系统内存 | 约 1.5 TiB | `free -h`，无 swap |
| NPU 数量 | 8 | `npu-smi info` |
| NPU 名称 | `910B4` | 每张卡同类 |
| Board/Product | `IT21HMDA_Bin6x` | `npu-smi info -i 0 -t board` |
| Board ID | `0x67` | 区别于旧文档里的 `0x62` |
| 驱动/软件版本 | `24.1.0.3` | `npu-smi` |
| 单卡 HBM | 32768 MB | 当前空闲，基础占用约 2.65 GB |
| 单卡 AICore Count | 20 | `npu-smi info -i 0 -t common` |
| AICore 频率 | 1650 MHz，当前 800 MHz | 空闲状态下当前频率较低 |

### 1.2 PyPTO 有效 `block_dim`

PyPTO runtime 会按 AICPU 调度线程数对 `block_dim` 做截断：

```text
scheduler_thread_num = aicpu_thread_num - 1 = 3
hw_max = rtGetAiCoreCount() = 20
effective block_dim = floor(hw_max / 3) * 3 = 18
```

因此本文中所有“本机”并行轮数均按 **18 AIC** 推算，而不是按 20 或 24。

| N blocks | 满配 24 AIC 轮数 | 本机 18 AIC 轮数 |
|----------|------------------|------------------|
| 64 | 3 | 4 |
| 100 | 5 | 6 |
| 128 | 6 | 8 |
| 200 | 9 | 12 |
| 400 | 17 | 23 |

### 1.3 当前软件环境状态

| 项 | 当前状态 | 影响 |
|----|----------|------|
| CANN toolkit | `/usr/local/Ascend/ascend-toolkit/latest` 可检测到 | 可作为 Ascend 环境入口 |
| 容器权限 | `CAP_SYS_RAWIO` 存在 | 真机 kernel 不应因该权限 hang |
| 默认 Python | `/usr/bin/python`，Python 3.9.9 | `pyproject.toml` 要求 `>=3.10` |
| Python 包 | 默认环境缺 `torch`/`torch_npu`/`pypto` | 当前不能直接运行样例 |
| Ascend 环境变量 | `ASCEND_HOME_PATH`/`ASCEND_OPP_PATH` 未设置 | 需要 source 或写入 shell |
| PTOAS_ROOT | `/data1/home/ziruiwang/ptoas-bin` 缺失 | `setup_npu_env.sh --dry-run` 失败 |
| PTO_ISA_ROOT | `/data1/home/ziruiwang/pto-isa` 缺失 | `setup_npu_env.sh --dry-run` 失败 |

**P0 结论**：当前设备硬件资源充足，但开发环境尚未完成。优化计划第一步必须是修复环境，
否则无法在这台新设备上得到可信 profiling。

---

## 2. 当前代码与旧文档的差异

### 2.1 目录与文件名

当前用户指定的原始代码目录是：

```text
examples/models/qwen3/32b/
```

当前优化代码目录是：

```text
examples/models/qwen-opt/
```

需要注意两处陈旧引用：

| 文件 | 当前问题 | 影响 |
|------|----------|------|
| `examples/models/qwen-opt/run_comparison.sh` | 原始路径写成 `examples/models/qwen3/qwen3_32b_decode_scope*.py`，缺少 `/32b/` | 批量验证会 SKIP/FAIL |
| `run_comparison.sh` 与 `testing_guide.md` | 优化文件名写成 `scope*_opt.py`，但实际文件是 `scope*-opt.py` | 找不到优化文件 |

**V0 行动**：在做任何性能判断前，先修正或绕开这些验证入口，使用真实路径手动跑一次。

### 2.2 当前关键 tiling 常量

| 文件 | 角色 | 关键常量 | 备注 |
|------|------|----------|------|
| `qwen3/32b/qwen3_32b_decode_scope1.py` | Scope 1 基线 | `K_CHUNK=512`, `Q_OUT_CHUNK=64`, `KV_OUT_CHUNK=64` | L0B 对 `[512,64]` 权重 tile 打满 |
| `qwen-opt/qwen3_32b_decode_scope1-opt.py` | Scope 1 opt | 同基线 | 主要是类名/构造参数差异 |
| `qwen3/32b/qwen3_32b_decode_scope2.py` | Scope 2 基线 | `Q_HEAD_BATCH=8`, `Q_HEAD_PAD=16`, `SEQ_TILE=64`, `SB_BATCH=64` | 历史结果显示接近基线最优 |
| `qwen-opt/qwen3_32b_decode_scope2-opt.py` | Scope 2 opt | 同基线 | 早期融合尝试已回退 |
| `qwen3/32b/qwen3_32b_decode_scope3.py` | Scope 3 当前基线 | `K_CHUNK=128`, `Q_OUT_CHUNK=64`, `MLP_OUT_CHUNK=256` | 与旧计划中“MLP=64”假设不一致 |
| `qwen-opt/qwen3_32b_decode_scope3-opt.py` | Scope 3 opt v3 | `K_CHUNK=128`, `Q_OUT_CHUNK=128`, `MLP_OUT_CHUNK=128` | 使用分离 gate/up 与顺序 `pl.range` |
| `qwen3/32b/qwen3_32b_decode.py` | 完整 decode | `Q_OUT_CHUNK=256`, `SEQ_TILE=256`, `MLP_OUT_CHUNK=256` | 不应直接套用 scope 级结论 |

### 2.3 旧报告可继承与不可继承的部分

| 内容 | 是否继承 | 说明 |
|------|----------|------|
| 本机有效 `block_dim=18` 的推导 | 是 | 当前设备仍是 20 AICore Bin，公式相同 |
| 单核 UB/L1/L0A/L0B/L0C 约束 | 是 | binned 影响核心数，不改变单核片上容量 |
| “避免共享 inout 导致串行化” | 是 | 属于 runtime 行为经验 |
| 2026-04-23 Scope 3 v3 的 -26.1% 结果 | **仅作历史参考** | 当前 `scope3.py` 常量已不同，必须复测 |
| “Gate/Up 融合作为 P0” | 否 | 已被后续 v3 结论取代 |
| “S2-4 per-sb 融合作为 P1” | 否 | 需要编译器支持 acc→vec 后再讨论 |

---

## 3. 历史实验结论摘要

### 3.1 v3 已验证结果（历史参考）

`examples/models/qwen-opt/opt-results/qwen3_scope_optimization_v3_results.md` 记录：

| Scope | 基线 | OPT v3 | 变化 | 结论 |
|-------|------|--------|------|------|
| Scope 1 | 407.6 us | 407.6 us | +0.0% | 保持基线 |
| Scope 2 | 879.7 us | 871.0 us | -1.0% | 保持基线，差异接近噪声 |
| Scope 3 | 3568.6 us | 2636.7 us | **-26.1%** | tile 变大后任务数下降，收益明确 |

v3 的核心经验不是“融合更多算子”，而是：

1. **Scope 3 使用更大的 tile 可以减少调度任务数**。
2. **Gate/Up 保持分离优于手动融合**，runtime 能自动实现跨迭代 overlap。
3. **Scope 1/2 的已有结构已经接近当前 runtime 下的较优点**。

### 3.2 已否决实验

| 实验 | 结果 | 当前结论 |
|------|------|----------|
| Gate/Up 融合 | 破坏 gate/up 跨迭代 overlap，或引入共享中间张量风险 | 不作为默认优化 |
| 含共享 `inout` 的 chunked `pl.parallel` | 可能被 runtime 序列化到单核 | 仅在确认无共享写依赖时使用 |
| Scope 1 `K_CHUNK=512→128` | 历史实测回退约 35.6% | 不优先 |
| Scope 1 Q proj `chunk=8` | 历史实测回退约 31.0% | 不优先 |
| Scope 2 QK+Softmax+SV 融合 | 编译器不支持 AIC acc 到 AIV vec 的单 incore 连接 | 框架任务 |

---

## 4. 更新后的具体执行计划

### 4.1 P0-E0：修复运行环境

目标：让 simulator、编译器、真机 runtime 都能在当前 shell 中工作。

| 步骤 | 检查点 |
|------|--------|
| 安装或切换到 Python >= 3.10 | `python -V` 满足 `pyproject.toml` |
| 安装项目依赖 | `import torch`, `import pypto` 成功 |
| 配置 CANN 环境变量 | `ASCEND_HOME_PATH`/`ASCEND_OPP_PATH`/`LD_LIBRARY_PATH` 有效 |
| 准备 PTOAS 与 PTO-ISA | `PTOAS_ROOT`/`PTO_ISA_ROOT` 指向存在目录 |
| dry-run 环境脚本 | `bash setup_npu_env.sh --dry-run` 不再报错 |
| 真机权限检查 | `CAP_SYS_RAWIO` 已存在，继续保留 |

### 4.2 P0-V0：修正验证入口

当前建议先使用手动命令验证真实文件路径：

```bash
python examples/models/qwen3/32b/qwen3_32b_decode_scope3.py -p a2a3 -d 0 --runtime-profiling
GOLDEN=$(ls -td build_output/Qwen3Scope3_*/data | head -1)
python examples/models/qwen-opt/qwen3_32b_decode_scope3-opt.py -p a2a3 -d 0 --runtime-profiling --golden-data "$GOLDEN"
```

随后再修正 `run_comparison.sh` 的映射：

| 基线 | 优化版 |
|------|--------|
| `examples/models/qwen3/32b/qwen3_32b_decode_scope1.py` | `examples/models/qwen-opt/qwen3_32b_decode_scope1-opt.py` |
| `examples/models/qwen3/32b/qwen3_32b_decode_scope2.py` | `examples/models/qwen-opt/qwen3_32b_decode_scope2-opt.py` |
| `examples/models/qwen3/32b/qwen3_32b_decode_scope3.py` | `examples/models/qwen-opt/qwen3_32b_decode_scope3-opt.py` |

### 4.3 P1-B0：重建当前基线

旧数据不能替代当前 checkout 的基线。需要至少记录：

| 指标 | 用途 |
|------|------|
| total span | 直接性能目标 |
| task 数量 | 判断调度开销 |
| AIC/AIV core 分布 | 识别是否串行到单核 |
| scheduler phase 数 | 识别调度器退化 |
| `memory_after_AllocateMemoryAddr.txt` | 检查 UB/L1/L0* 是否逼近上限 |
| golden PASS/FAIL | 正确性底线 |

建议先跑单卡 device 0；多卡只用于吞吐并发，不改变单个 scope kernel 的片上优化结论。

### 4.4 P1-S3-T：Scope 3 tile sweep

由于当前 `scope3.py` 与 `qwen-opt` v3 的常量不一致，下一步不是直接接受某个值，
而是在当前设备上做受控 sweep。

| 实验 | `Q_OUT_CHUNK` | `MLP_OUT_CHUNK` | 预期观察 |
|------|---------------|-----------------|----------|
| 当前基线 | 64 | 256 | 作为当前 checkout 真实 baseline |
| v3 opt | 128 | 128 | 验证历史 -26.1% 是否仍成立 |
| Q-only 变大 | 128 | 256 | 分离 output projection 收益与 MLP tile 影响 |
| MLP-only 变小 | 64 | 128 | 判断 MLP=128 是否因 task 数/调度改善而优于 256 |

选择规则：

1. golden 必须 PASS。
2. total span 必须稳定优于当前基线。
3. AIC/AIV core 分布不能退化为单核串行。
4. 若 `MLP_OUT_CHUNK=256` 已优于 128，则不要为了复用 v3 代码强行改小。
5. 若 `Q_OUT_CHUNK=128` 明显改善 output projection 且不伤害后续阶段，则优先保留。

### 4.5 P2-S2-B：Scope 2 保持结构，补上下文长度 sweep

Scope 2 当前结构：

```text
RoPE/KV cache → QK matmul → Softmax → SV matmul → online softmax
```

当前计划：

| 项 | 决策 |
|----|------|
| QK+Softmax+SV 融合 | 不做，等待编译器支持 acc→vec |
| batch/group 并行 | 仅在短 ctx_len 下复测，避免长 ctx_len 已饱和时增加 GM 中间压力 |
| `SEQ_TILE` | 先保持 64；完整 decode 的 256 需要单独评估 |

### 4.6 P2-S1-B：Scope 1 保持结构

当前 Scope 1 的 `K_CHUNK=512` 让 `[512,64]` BF16 权重 tile 正好占满 64 KB L0B。
虽然看起来没有余量，但历史实测说明降低 `K_CHUNK` 会增加循环与 RMSNorm 开销。

当前计划：

1. 保留 `K_CHUNK=512`。
2. 不再优先尝试 `chunk=8`。
3. 仅在当前设备复测后，如果 Scope 1 成为总瓶颈，再考虑 tile DSL 或框架级实验。

### 4.7 P3-F0：框架级后续方向

这些方向不是当前模型代码的短期优化，而是 PyPTO/runtime 能力建设：

| 方向 | 解锁内容 |
|------|----------|
| 基于 tensor view offset 的写依赖分析 | 允许不同 tile 写同一大 tensor 的不重叠区域时并行 |
| incore 内 AIC acc → AIV vec 数据移动 | 解锁 Scope 2 QK+Softmax+SV per-sb 融合 |
| `pl.parallel` per-iteration alloc | 避免共享 `inout` 触发保守串行化 |
| profiling 自动摘要 | 自动输出 span/task/core/memory 对比，减少手动读 swimlane |

---

## 5. 当前设备下的风险与验证标准

### 5.1 风险

| 风险 | 表现 | 应对 |
|------|------|------|
| 环境未就绪 | import 或编译失败 | 先完成 P0-E0 |
| 旧路径导致假失败 | `run_comparison.sh` 找不到文件 | 先完成 P0-V0 |
| 旧报告与当前代码不一致 | 优化结论无法复现 | 当前 checkout 重新 profiling |
| 共享 `inout` 串行化 | AIC 只跑 core 0 | 检查 core 分布，避免该编排模式 |
| binned 核数误判 | 按 24 核估算过高 | 所有轮数按 18 核计算 |

### 5.2 接受标准

任一代码优化进入“推荐”前必须满足：

1. **正确性**：golden PASS，容差不放宽到掩盖误差。
2. **稳定性**：同一设备至少重复 3 次，排除频率/温度/后台进程波动。
3. **性能**：total span 明确下降，且不是只减少 task 数但实际 span 上升。
4. **调度**：AIC/AIV core 分布符合预期，没有意外单核串行。
5. **资源**：UB/L1/L0A/L0B/L0C 不超过限制，HBM 中间张量可接受。

---

## 6. 最短执行路径

按当前状态，建议顺序如下：

1. **修环境**：Python >= 3.10、`torch`/`pypto`、CANN env、`PTOAS_ROOT`、`PTO_ISA_ROOT`。
2. **修验证入口**：更新 `run_comparison.sh` 路径和 `-opt.py` 文件名。
3. **跑当前基线**：Scope 1/2/3 各自 golden + profiling。
4. **跑现有 qwen-opt**：确认当前 `qwen-opt` 是否仍能在新设备和当前代码下 PASS。
5. **Scope 3 sweep**：只围绕 `Q_OUT_CHUNK`/`MLP_OUT_CHUNK` 做最小矩阵实验。
6. **再考虑完整 decode**：完整 `qwen3_32b_decode.py` 常量不同，必须单独 profiling。

---

## 7. 附录：模型尺寸

| 参数 | 值 |
|------|-----|
| BATCH | 16 |
| MAX_SEQ | 4096 |
| HIDDEN | 8192 |
| KV_HIDDEN | 1024 |
| INTERMEDIATE | 25600 |
| NUM_HEADS | 64 |
| NUM_KV_HEADS | 8 |
| HEAD_DIM | 128 |

常用张量大小：

| Tensor | Shape | DType | Size |
|--------|-------|-------|------|
| hidden_states | `[16, 8192]` | BF16 | 256 KB |
| q_proj | `[16, 8192]` | FP32 | 512 KB |
| k_proj / v_proj | `[16, 1024]` | FP32 | 64 KB each |
| k_cache / v_cache | `[524288, 128]` | BF16 | 128 MB each |
| wo | `[8192, 8192]` | BF16 | 128 MB |
| w_gate / w_up | `[8192, 25600]` | BF16 | 400 MB each |
| w_down | `[25600, 8192]` | BF16 | 400 MB |
| post_norm_tile | `[16, 8192]` | BF16 | 256 KB |
| resid1_tile | `[16, 8192]` | FP32 | 512 KB |
| mlp_tile | `[16, 25600]` | BF16 | 800 KB |
