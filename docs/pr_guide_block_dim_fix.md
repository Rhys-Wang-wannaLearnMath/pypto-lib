# Pull Request 提交指南：NPU 运行时修复（block_dim + swimlane）

> **目标仓库**: `https://github.com/hw-native-sys/pypto`

---

# Pull Request #1：block_dim 自动检测修复

> **涉及文件**: `python/pypto/runtime/device_runner.py`（仅 1 个文件）  
> **变更类型**: Bug fix — 修复 binned 设备上 NPU 挂死问题

---

## 0. 前置条件

```bash
# 必须安装
git --version          # >= 2.x
gh --version           # GitHub CLI（用于命令行创建 PR）

# 如果没有 gh CLI，安装：
#   Linux:  sudo apt install gh  或参考 https://cli.github.com/
#   然后登录：
gh auth login          # 选择 GitHub.com → HTTPS → 浏览器授权
```

---

## 1. Fork 仓库

在浏览器中操作：

1. 打开 https://github.com/hw-native-sys/pypto
2. 点击右上角 **Fork** 按钮
3. 选择你的 GitHub 用户名作为 fork 目标
4. 记下你的 fork 地址：`https://github.com/<你的用户名>/pypto`

---

## 2. 克隆 Fork 并配置 upstream

```bash
# 克隆你的 fork（不需要 submodule，只改 Python 文件）
git clone https://github.com/<你的用户名>/pypto.git /tmp/pypto-pr
cd /tmp/pypto-pr

# 添加上游仓库
git remote add upstream https://github.com/hw-native-sys/pypto.git
git fetch upstream

# 确认 remote 配置
git remote -v
# origin    https://github.com/<你的用户名>/pypto.git (fetch)
# origin    https://github.com/<你的用户名>/pypto.git (push)
# upstream  https://github.com/hw-native-sys/pypto.git (fetch)
# upstream  https://github.com/hw-native-sys/pypto.git (push)
```

---

## 3. 创建分支

```bash
# 从最新的 upstream/main 创建分支
git fetch upstream
git checkout -b fix/block-dim-clamp-binned-device upstream/main
```

分支命名规范（项目使用 `fix/`、`feat/`、`docs/` 等前缀）：
- `fix/block-dim-clamp-binned-device` ✅

---

## 4. 修改代码

只需修改一个文件：`python/pypto/runtime/device_runner.py`

### 4.1 定位修改位置

```bash
grep -n "def execute_on_device" python/pypto/runtime/device_runner.py
# 输出类似: 462:def execute_on_device(
```

### 4.2 应用修改

在 `execute_on_device()` 函数的 docstring 之后、`worker = Worker(...)` 之前插入以下代码：

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

完整的修改 diff 应该类似：

```diff
--- a/python/pypto/runtime/device_runner.py
+++ b/python/pypto/runtime/device_runner.py
@@ -484,6 +484,23 @@ def execute_on_device(
         enable_profiling: Enable runtime profiling.
         runtime_env: Optional per-example environment variable overrides.
     """
+    # Cap block_dim at actual AICore count to avoid TSCH dispatch hang
+    # on binned devices (e.g. 910B3 Bin6 has 20 cores, not 24).
+    # Also round down to nearest multiple of scheduler_thread_num so the
+    # C++ runtime can evenly distribute blocks across scheduler threads.
+    if not platform.endswith("sim"):
+        try:
+            import ctypes
+            _rt = ctypes.CDLL("libruntime.so")
+            _cnt = ctypes.c_uint32()
+            if _rt.rtGetAiCoreCount(ctypes.byref(_cnt)) == 0 and _cnt.value > 0:
+                hw_max = _cnt.value
+                if block_dim > hw_max:
+                    sched_threads = max(aicpu_thread_num - 1, 1)
+                    new_bd = (hw_max // sched_threads) * sched_threads
+                    if new_bd < 1:
+                        new_bd = sched_threads
+                    print(f"[WARN] block_dim={block_dim} exceeds AICore count={hw_max}; "
+                          f"clamping to {new_bd} (divisible by {sched_threads}).")
+                    block_dim = new_bd
+        except Exception:
+            pass  # best-effort; fall through to user-specified block_dim
+
     worker = Worker(level=2, device_id=device_id, platform=platform, runtime=runtime_name)
     worker.init()
```

### 4.3 验证修改

```bash
# 确认只改了一个文件
git diff --stat
#  python/pypto/runtime/device_runner.py | 17 +++++++++++++++++
#  1 file changed, 17 insertions(+)

# 查看完整 diff
git diff
```

---

## 5. 本地测试

### 5.1 代码风格检查

项目使用 ruff，行宽 120：

```bash
# 如果有 ruff（项目规范）
pip install ruff
ruff check python/pypto/runtime/device_runner.py
ruff format --check python/pypto/runtime/device_runner.py
```

### 5.2 功能测试（在 NPU 设备上）

```bash
# 安装修改后的 pypto
cd /tmp/pypto-pr
uv pip install .
uv pip install ./runtime

# 设置环境变量
export PTOAS_ROOT=$HOME/ptoas-bin
export PTO_ISA_ROOT=$HOME/pto-isa
export ASCEND_HOME_PATH=<你的CANN路径>

# 运行测试
cd /path/to/pypto-lib
python examples/beginner/hello_world.py -p a2a3 -d <设备号>
python examples/models/qwen3/qwen3_32b_decode_scope1.py -p a2a3 -d <设备号>
```

预期输出：
```
[WARN] block_dim=24 exceeds AICore count=20; clamping to 18 (divisible by 3).
...
[RUN] PASS
```

### 5.3 模拟器测试（确保不影响 sim 路径）

```bash
# sim 平台应跳过检测（platform.endswith("sim") 为 True）
python examples/beginner/hello_world.py -p a2a3sim
```

---

## 6. 提交

```bash
cd /tmp/pypto-pr

git add python/pypto/runtime/device_runner.py

git commit -m "fix(runtime): clamp block_dim to hardware AICore count on binned devices

On binned Ascend 910B3 variants (e.g. Bin6 with 20 AICore clusters instead
of 24), launching with block_dim=24 causes TSCH to dispatch tasks to
non-existent cores. The AICPU executor then waits forever for handshakes
from cores that don't exist, resulting in a permanent hang at
rtStreamSynchronize.

Query the actual AICore count via rtGetAiCoreCount at runtime and clamp
block_dim to the hardware maximum. Also round down to the nearest multiple
of scheduler_thread_num (aicpu_thread_num - 1) to satisfy the C++ runtime's
even distribution requirement.

The check is skipped on simulator platforms (*sim) where the API is
unavailable, and wrapped in a try/except for graceful fallback."
```

### Commit message 规范

项目遵循 Conventional Commits 风格：
- **格式**: `type(scope): subject`
- **类型**: `fix`, `feat`, `refactor`, `docs`, `test`, `chore`
- **scope**: 改动所在模块（如 `runtime`, `compiler`, `frontend`）
- **正文**: 解释 what/why，不需要 how（代码已经说明）

---

## 7. Rebase 到最新 upstream

```bash
git fetch upstream
git rebase upstream/main

# 如果有冲突：
#   解决冲突 → git add <file> → git rebase --continue
# 如果想放弃：
#   git rebase --abort
```

---

## 8. Push 到 Fork

```bash
# 首次 push
git push --set-upstream origin fix/block-dim-clamp-binned-device

# 如果 rebase 后需要强制推送
git push --force-with-lease origin fix/block-dim-clamp-binned-device
```

> **注意**：始终用 `--force-with-lease`，不要用 `--force`。

---

## 9. 创建 Pull Request

### 方式 A：使用 gh CLI（推荐）

```bash
gh pr create \
  --repo hw-native-sys/pypto \
  --title "fix(runtime): clamp block_dim to hardware AICore count on binned devices" \
  --body "$(cat <<'EOF'
## Summary

On binned Ascend 910B3 variants (e.g. IT21HMDA_Bin6, Board ID 0x62), the
hardware has fewer AICore clusters than the standard 24 (this device has 20).
The runtime's default `block_dim=24` causes TSCH to dispatch AICore tasks to
non-existent compute clusters. The AICPU executor then waits indefinitely for
handshake responses from 72 cores (24×3), but only 60 (20×3) can respond,
resulting in a permanent hang at `rtStreamSynchronize`.

## Changes

- Query actual AICore count via `rtGetAiCoreCount()` in `execute_on_device()`
- Clamp `block_dim` to `min(block_dim, hw_aicore_count)`
- Round down to nearest multiple of `scheduler_thread_num` (= `aicpu_thread_num - 1`)
  to satisfy the C++ runtime's even block distribution requirement
- Skip check on simulator platforms (`*sim`) and wrap in try/except for graceful fallback

## Root Cause Analysis

1. `block_dim=24` dispatches AICore kernel to 24 blocks via TSCH
2. Binned device only has 20 physical compute clusters
3. AICPU orchestrator calls `handshake_all_cores(72)` (24 blocks × 3 cores each)
4. Only 60 cores respond → AICPU busy-waits forever → `rtStreamSynchronize` hangs

## Testing

- **910B3 Bin6 (20 AICore clusters)**: `hello_world.py` and `qwen3_32b_decode_scope1.py` both PASS
- **Simulator path**: Unaffected (check skipped for `*sim` platforms)
- block_dim auto-adjusted: 24 → 18 (nearest multiple of 3 ≤ 20)

## Affected Devices

All binned/reduced Ascend 910B series variants where AICore count < 24.
EOF
)"
```

### 方式 B：浏览器手动创建

1. 打开 `https://github.com/hw-native-sys/pypto/compare/main...<你的用户名>:fix/block-dim-clamp-binned-device`
2. 点击 **Create pull request**
3. 填写 Title 和 Body（参考方式 A 中的内容）
4. 点击 **Create pull request**

---

## 10. PR 提交后

### 10.1 等待 CI

PR 创建后，GitHub Actions 会自动运行 CI：
- 代码风格检查（ruff）
- 单元测试
- 编译测试

### 10.2 响应 Review

maintainer 可能会提出修改意见，按需修改后：

```bash
# 修改代码
vim python/pypto/runtime/device_runner.py

# 追加提交（或 amend）
git add python/pypto/runtime/device_runner.py
git commit --amend --no-edit    # 修改同一个 commit
git push --force-with-lease     # 更新 PR
```

### 10.3 可能的 Review 反馈及应对

| 可能的反馈 | 应对 |
|-----------|------|
| "应该在 C++ 层做检测" | 可以同意并补充 C++ 层检测，但 Python 层作为 defense-in-depth 仍有价值 |
| "ctypes 调用不够优雅" | 可改为在 simpler 的 nanobind 绑定中暴露 `get_aicore_count()` |
| "print 改为 logging" | 合理，改为 `import logging; logger = logging.getLogger(__name__)` |
| "需要单元测试" | 可以补充 mock `rtGetAiCoreCount` 返回不同值的测试 |

---

## 完整命令速查

```bash
# === 一次性完整流程 ===

# 1. Fork（在浏览器操作）

# 2. 克隆 + 配置
git clone https://github.com/<你的用户名>/pypto.git /tmp/pypto-pr
cd /tmp/pypto-pr
git remote add upstream https://github.com/hw-native-sys/pypto.git
git fetch upstream
git checkout -b fix/block-dim-clamp-binned-device upstream/main

# 3. 修改（用你喜欢的编辑器）
vim python/pypto/runtime/device_runner.py

# 4. 检查
git diff
ruff check python/pypto/runtime/device_runner.py

# 5. 提交
git add python/pypto/runtime/device_runner.py
git commit -m "fix(runtime): clamp block_dim to hardware AICore count on binned devices"

# 6. Rebase + Push
git fetch upstream && git rebase upstream/main
git push --set-upstream origin fix/block-dim-clamp-binned-device

# 7. 创建 PR
gh pr create --repo hw-native-sys/pypto \
  --title "fix(runtime): clamp block_dim to hardware AICore count on binned devices" \
  --body "..."
```

---
---

# Pull Request #2：Swimlane 自动转换修复

> **涉及文件**: `runtime/CMakeLists.txt` + `python/pypto/runtime/runner.py`（2 个文件）  
> **变更类型**: Bug fix — 修复真机 profiling 后 `merged_swimlane_*.json` 不自动生成

---

## 问题描述

在真机运行（`-p a2a3`）时，profiling 结束后只有 `perf_swimlane_*.json`，
`merged_swimlane_*.json` 不会自动生成。`_generate_swimlane()` 静默跳过，无任何报错。

## 根因

1. **打包缺失**：`runtime/CMakeLists.txt` 只安装了 `src/` 和 `build/lib/` 到 wheel 的
   `simpler_setup/_assets/` 目录，漏掉了 `tools/`（包含 `swimlane_converter.py`）。
2. **静默失败**：`pypto/runtime/runner.py` 中 `_generate_swimlane()` 找不到脚本时
   直接 `return`，无任何警告信息。

## 创建分支

```bash
cd /tmp/pypto-pr
git fetch upstream
git checkout -b fix/swimlane-converter-packaging upstream/main
```

## 修改代码

### 修改 1：`runtime/CMakeLists.txt` — 添加 `tools/` 安装

定位 `if(SKBUILD_MODE)` 块，在 `build/lib/` install 之后添加 `tools/`：

```cmake
# AFTER (fixed)
if(SKBUILD_MODE)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/
            DESTINATION simpler_setup/_assets/src)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib/
            DESTINATION simpler_setup/_assets/build/lib
            OPTIONAL)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/tools/
            DESTINATION simpler_setup/_assets/tools)
endif()
```

Diff：

```diff
--- a/runtime/CMakeLists.txt
+++ b/runtime/CMakeLists.txt
@@ -43,5 +43,7 @@ if(SKBUILD_MODE)
     install(DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib/
             DESTINATION simpler_setup/_assets/build/lib
             OPTIONAL)
+    install(DIRECTORY ${CMAKE_SOURCE_DIR}/tools/
+            DESTINATION simpler_setup/_assets/tools)
 endif()
```

### 修改 2：`python/pypto/runtime/runner.py` — 添加缺失脚本警告

定位 `_generate_swimlane()` 中 `swimlane_script.exists()` 检查，添加 warning：

```python
# AFTER (warns)
swimlane_script = simpler_root / "tools" / "swimlane_converter.py"
if not swimlane_script.exists():
    print(
        f"Warning: swimlane_converter.py not found at {swimlane_script}, "
        f"skipping auto-conversion. Run manually with: "
        f"python examples/swimlane_converter.py <perf_swimlane_*.json>"
    )
    return
```

Diff：

```diff
--- a/python/pypto/runtime/runner.py
+++ b/python/pypto/runtime/runner.py
@@ swimlane_script = simpler_root / "tools" / "swimlane_converter.py"
  if not swimlane_script.exists():
-     return
+     print(
+         f"Warning: swimlane_converter.py not found at {swimlane_script}, "
+         f"skipping auto-conversion. Run manually with: "
+         f"python examples/swimlane_converter.py <perf_swimlane_*.json>"
+     )
+     return
```

## 验证修改

```bash
git diff --stat
#  runtime/CMakeLists.txt           | 2 ++
#  python/pypto/runtime/runner.py   | 5 ++++-
#  2 files changed, 6 insertions(+), 1 deletion(-)
```

## 本地测试

```bash
# 重新安装 wheel
cd /tmp/pypto-pr
uv pip install ./runtime

# 确认 tools/ 被打包
ASSETS=$(python -c "from simpler_setup.environment import PROJECT_ROOT; print(PROJECT_ROOT)")
ls "$ASSETS/tools/swimlane_converter.py"  # 应存在

# 真机 profiling 测试
python examples/beginner/matmul.py -p a2a3 --runtime-profiling
# 预期：build_output/*/swimlane_data/ 下同时出现
#   perf_swimlane_*.json
#   merged_swimlane_*.json
```

## 提交

```bash
git add runtime/CMakeLists.txt python/pypto/runtime/runner.py

git commit -m "fix(runtime): package tools/ dir and warn on missing swimlane converter

runtime/CMakeLists.txt only installs src/ and build/lib/ into the wheel's
simpler_setup/_assets/ directory, but omits tools/ which contains
swimlane_converter.py. This causes _generate_swimlane() to silently skip
the perf-to-Perfetto conversion on real-device profiling runs.

- Add tools/ to the SKBUILD_MODE install list in CMakeLists.txt
- Print a warning in _generate_swimlane() when the converter script is
  missing instead of silently returning"
```

## Push + 创建 PR

```bash
git push --set-upstream origin fix/swimlane-converter-packaging

gh pr create \
  --repo hw-native-sys/pypto \
  --title "fix(runtime): package tools/ dir and warn on missing swimlane converter" \
  --body "$(cat <<'EOF'
## Summary

On real-device profiling runs (`-p a2a3 --runtime-profiling`), `merged_swimlane_*.json`
is never auto-generated. Only raw `perf_swimlane_*.json` is produced.

## Root Cause

1. **Packaging bug**: `runtime/CMakeLists.txt` installs `src/` and `build/lib/` into the
   wheel but omits `tools/`, so `swimlane_converter.py` is absent at runtime.
2. **Silent failure**: `_generate_swimlane()` in `runner.py` silently returns when the
   converter script is not found — no warning, no error.

## Changes

- **`runtime/CMakeLists.txt`**: Add `install(DIRECTORY ... tools/ ...)` to SKBUILD_MODE block
- **`python/pypto/runtime/runner.py`**: Print a warning message before returning when
  `swimlane_converter.py` is not found

## Testing

- Real-device profiling: `merged_swimlane_*.json` is now auto-generated
- Simulator path: Unaffected (swimlane conversion only runs on real-device platforms)
- Verified `tools/swimlane_converter.py` exists in installed wheel assets
EOF
)"
```

## 可能的 Review 反馈及应对

| 可能的反馈 | 应对 |
|-----------|------|
| "用 `logging` 代替 `print`" | 合理，改为 `logger.warning(...)` |
| "tools/ 太宽泛，只装 swimlane_converter.py" | 可改为 `install(FILES ... tools/swimlane_converter.py)` |
| "应该加测试" | 可补充 CI 检查确认 wheel 包含 `tools/swimlane_converter.py` |
