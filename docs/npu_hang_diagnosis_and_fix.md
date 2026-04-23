# NPU Hang 诊断与修复报告

> **环境**: Ascend 910B3 (Bin6, Board ID 0x62) · CANN 8.5.0 · Driver 25.5.0 · Python 3.11  
> **日期**: 2026-04-22  
> **影响范围**: 所有使用 `block_dim=24` 的 PyPTO-Lib 示例在非满配 910B3 设备上运行时会无限挂起

---

## 1. 问题现象

运行 `hello_world.py` 和 `qwen3_32b_decode_scope1.py` 时，程序在以下位置永久挂起：

```
=== rtStreamSynchronize stream_aicpu_ ===
```

进程不返回、不报错，只能通过 `kill` 或 `timeout` 终止。

---

## 2. 诊断过程

### 2.1 排除的假设

| 假设 | 排除依据 |
|------|----------|
| CANN/Driver 安装损坏 | PyTorch NPU `torch.matmul` 测试通过 |
| `pip install` 构建不完整 | `ccec`、交叉编译器均存在，`libhost_runtime.so`、`libaicpu_kernel.so`、`aicore_kernel.o` 均在 `build/lib/` 下且为新编译 |
| AICore 内核 ELF 格式不兼容 | `readelf -h` 对比用户编译的 `.o` 文件与 CANN 参考内核（如 `hccl_ar_superkernel_float32.o`），Machine type 均为 `0x1029`，ELF flags 一致 |
| `ptoas` / `pto-isa` 缺失 | 已正确设置 `PTOAS_ROOT` 和 `PTO_ISA_ROOT` 环境变量 |
| AICPU↔AICore 启动顺序导致调度死锁 | 尝试在 `device_runner.cpp` 中交换启动顺序（AICore 先于 AICPU main），挂起依旧 |

### 2.2 CANN 日志分析

启用 `ASCEND_GLOBAL_LOG_LEVEL=0` 后，从 host plog 和 device log 中提取到关键时间线：

**Host 侧 (`plog-*.log`)**：
```
[RUNTIME] rtAicpuKernelLaunchExWithArgs: kernel_type=5, flags=0           # AICPU init → stream 47, task 0
[RUNTIME] LaunchKernelWithHandle: stream_id=46, task_id=0,
          kernel_name=aicore_kernel_0_mix_aic, mixType=3, block_dim=24    # AICore → stream 46, task 0
[RUNTIME] rtAicpuKernelLaunchExWithArgs: kernel_type=5, flags=0           # AICPU main → stream 47, task 1
[RUNTIME] StreamSynchronize: stream_id=47, timeout=-1ms                   # 等待 stream 47 完成
```

- **AICore task 已正确提交到 TSCH**（`AllocTaskAndSendStars: task_type=0(KERNEL_AICORE), sendSqeNum=1`）
- **AICPU init (task 0) 正常完成**（`TaskFinished: stream_id=47, task_id=0, task_type=1 (KERNEL_AICPU)`）
- **AICore task 从未返回 CQE 完成事件** — 无任何 `TaskFinished` 或错误报告
- **AICPU main (task 1) 从未完成** — 因等待 AICore 握手而阻塞

**Device 侧 (`device-*.log`)**：
```
[AICPU] DynTileFwkBackendKernelServerInit "[kernel.cpp:52] Runtime Executor Init: Initializing AICPU kernel"
[AICPU] DynTileFwkBackendKernelServer "[kernel.cpp:95] Calling aicpu_execute with Runtime"
[AICPU] handshake_all_cores "Handshaking with 72 cores"    ← 关键行！
```

AICPU 主内核启动后尝试与 **72 个 AICore 核心**握手（= `block_dim × 3` = `24 × 3`），然后日志停止 — 永久阻塞。

### 2.3 定位根因

查询设备实际 AICore 数量：

```python
import ctypes
lib = ctypes.CDLL("libruntime.so")
cnt = ctypes.c_uint32()
lib.rtGetAiCoreCount(ctypes.byref(cnt))
print(cnt.value)  # → 20
```

**此设备为 910B3 Bin6（`IT21HMDA_Bin6`），仅有 20 个计算集群，而非标准的 24 个。**

每个计算集群包含 1 个 AIC（cube）核 + 2 个 AIV（vector）核 = 3 核：

| 配置 | 计算集群 | 总核心数 | AICPU 期望握手 |
|------|---------|---------|----------------|
| `block_dim=24`（配置值） | 24 | 72 | 72 |
| 实际硬件 | **20** | **60** | — |
| **缺口** | **4** | **12** | **12 核永远不会响应** |

TSCH 将 AICore task 分发到 24 个 block，但只有 20 个物理存在。AICPU 期望 72 个核心的握手信号，但只有 60 个能响应 → **永久死锁**。

---

## 3. 修复方案

### 3.1 修改位置

**文件**：`pypto/runtime/device_runner.py` → `execute_on_device()` 函数

### 3.2 修改内容

在 `execute_on_device()` 入口处添加运行时硬件检测和 `block_dim` 自动校正：

```python
# Cap block_dim at actual AICore count to avoid TSCH dispatch hang
# on binned devices (e.g. 910B3 Bin6 has 20 cores, not 24).
# Also round down to nearest multiple of scheduler_thread_num so the
# C++ runtime can evenly distribute blocks across scheduler threads.
if not platform.endswith("sim"):
    try:
        import ctypes
        _rt = ctypes.CDLL("libruntime.so")
        _cnt = ctypes.c_uint32()
        if _rt.rtGetAiCoreCount(ctypes.byref(_cnt)) == 0 and _cnt.value > 0:
            hw_max = _cnt.value
            if block_dim > hw_max:
                # scheduler_thread_num = aicpu_thread_num - 1 (one thread is orchestrator)
                sched_threads = max(aicpu_thread_num - 1, 1)
                new_bd = (hw_max // sched_threads) * sched_threads
                if new_bd < 1:
                    new_bd = sched_threads
                print(f"[WARN] block_dim={block_dim} exceeds AICore count={hw_max}; "
                      f"clamping to {new_bd} (divisible by {sched_threads}).")
                block_dim = new_bd
    except Exception:
        pass  # best-effort; fall through to user-specified block_dim
```

### 3.3 校正逻辑

1. **查询硬件**：通过 `rtGetAiCoreCount()` 获取当前设备的实际 AICore 集群数
2. **上界截断**：如果 `block_dim > hw_max`，则截断至硬件上限
3. **对齐调度线程数**：C++ 运行时要求 `block_dim % scheduler_thread_num == 0`，其中 `scheduler_thread_num = aicpu_thread_num - 1`（1 个线程留给 orchestrator）。向下取整到最近的合法倍数
4. **仅限物理设备**：模拟器平台（`*sim`）跳过此检查

**示例**（本设备）：
- 输入：`block_dim=24`, `aicpu_thread_num=4`
- `hw_max=20`, `sched_threads=3`
- `new_bd = (20 // 3) * 3 = 18`
- 输出：`block_dim=18`

### 3.4 验证结果

| 测试 | 修复前 | 修复后 |
|------|--------|--------|
| `hello_world.py` | 永久挂起 | **PASS** (5.25s) |
| `qwen3_32b_decode_scope1.py` | 永久挂起 | **PASS** (7.09s) |

---

## 4. 调试过程中的临时修改（已确认无效）

### 4.1 `device_runner.cpp` 启动顺序交换

在诊断过程中，曾修改 `device_runner.cpp` 中的内核启动顺序：

**原始顺序**：
```
AICPU init → AICPU main → AICore
```

**临时修改**：
```
AICPU init → AICore → AICPU main
```

**结论**：此修改 **对问题无影响**，原因如下：
- AICPU 和 AICore 位于不同的 TSCH stream（`stream_id=47` vs `stream_id=46`），使用不同的 SQ 队列（`sq_id=2` vs `sq_id=3`）
- Host 端的提交顺序不决定 device 端的调度顺序，TSCH 对不同 stream 的任务并行调度
- 真正的阻塞原因是 AICore task 因硬件集群不足而无法全部执行，与提交时序无关
- **此修改仅作用于源文件，未重新编译为 `libhost_runtime.so`**，因此实际运行的二进制使用的仍是原始启动顺序

**建议**：将 `device_runner.cpp` 恢复为原始启动顺序，并移除调试用的 `std::cout` 输出。

### 4.2 调试用的 `std::cout` 输出

`device_runner.cpp` 中添加了多处 `std::cout` 调试输出（如 `=== launch_aicpu_kernel ===`）。这些输出仅在源文件中，未编译到运行二进制。建议一并清理。

---

## 5. 影响与建议

### 5.1 受影响的设备

所有非满配（binned）Ascend 910B 系列设备：
- **910B3 Bin6**（本例）：20 AICore 集群
- 其他可能的 binned 型号：AICore 数量可能为 16、20 等非 24 的值

### 5.2 长期建议

1. **上游修复**：将 `block_dim` 自动检测逻辑合入 PyPTO-Lib 主线代码，而非依赖硬编码默认值 24
2. **`kernel_config.py` 生成**：编译器生成 `RUNTIME_CONFIG` 时应查询目标设备的实际核心数，或标记 `block_dim="auto"`
3. **C++ 运行时增强**：`device_runner.cpp` 的 `run()` 函数可在启动前主动检测 `block_dim > hardware_max` 并自动校正或报错，而非依赖 Python 层
4. **文档补充**：在 README 中说明 binned 设备的兼容性注意事项

---

## 6. 关键诊断工具与命令

```bash
# 查询 AICore 集群数量
python3 -c "
import ctypes
lib = ctypes.CDLL('libruntime.so')
cnt = ctypes.c_uint32()
lib.rtGetAiCoreCount(ctypes.byref(cnt))
print(f'AICore count: {cnt.value}')
"

# 启用 CANN 详细日志
export ASCEND_GLOBAL_LOG_LEVEL=0  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

# 日志位置
#   Host plog: ~/ascend/log/run/plog/plog-*.log
#   Device log: ~/ascend/log/run/device-0/device-*.log
#   Debug log:  ~/ascend/log/debug/

# 检查设备状态
npu-smi info

# ELF 头检查
readelf -h <kernel.o>
readelf -s <kernel.o> | grep FUNC
```
