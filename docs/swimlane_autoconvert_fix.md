# Swimlane Auto-Convert Fix

## Problem

When running on real NPU device (`-p a2a3`), `merged_swimlane_*.json` is not
automatically generated after profiling. Only `perf_swimlane_*.json` exists in
`build_output/*/swimlane_data/`.

According to `docs/swimlane_guide.md`, real-device runs should auto-invoke
`swimlane_converter.py` to produce `merged_swimlane_*.json`.

## Root Cause

Two issues at different layers:

### 1. Packaging bug (root cause) — `runtime/CMakeLists.txt`

`runtime/CMakeLists.txt` line 40–46 only installs `src/` and `build/lib/` into
the wheel's `simpler_setup/_assets/` directory. The `tools/` directory (which
contains `swimlane_converter.py`) is **not installed**:

```cmake
# BEFORE (broken)
if(SKBUILD_MODE)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/
            DESTINATION simpler_setup/_assets/src)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib/
            DESTINATION simpler_setup/_assets/build/lib
            OPTIONAL)
endif()
```

At runtime, `_generate_swimlane()` in `pypto/runtime/runner.py` resolves
`simpler_root` via `get_simpler_root()` → `simpler_setup/_assets/`, then looks
for `tools/swimlane_converter.py` under it. Since `tools/` was never installed,
the script is not found.

### 2. Silent failure — `pypto/runtime/runner.py`

When `swimlane_converter.py` is missing, `_generate_swimlane()` silently
returns without printing any warning:

```python
# BEFORE (silent)
swimlane_script = simpler_root / "tools" / "swimlane_converter.py"
if not swimlane_script.exists():
    return
```

This makes the issue very hard to diagnose — no error, no warning, nothing in
the output to indicate the conversion was skipped.

## Call Chain

```
runner.py::_execute_on_device()
  → _collect_swimlane_data()
    → platform.endswith("sim") ? skip : _generate_swimlane()
      → simpler_root = get_simpler_root()
        → returns .venv/.../simpler_setup/_assets/
      → swimlane_script = simpler_root / "tools" / "swimlane_converter.py"
      → swimlane_script.exists() == False  ← tools/ never installed
      → return  ← silently skipped
```

## Fix

### Fix 1: `runtime/CMakeLists.txt` — add `tools/` to install

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

### Fix 2: `pypto/runtime/runner.py` — add warning on missing script

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

### Temporary workaround (already applied)

For existing environments without rebuilding the wheel:

```bash
# Copy swimlane_converter.py into the expected location
ASSETS=$(python -c "from simpler_setup.environment import PROJECT_ROOT; print(PROJECT_ROOT)")
mkdir -p "$ASSETS/tools"
cp /tmp/pypto/runtime/tools/swimlane_converter.py "$ASSETS/tools/"
```

## Verification

After fix, run any example with profiling on real device:

```bash
python examples/beginner/matmul.py -p a2a3 --runtime-profiling
```

Expected output should include:

```
Swimlane JSON written to: build_output/.../swimlane_data/merged_swimlane_*.json
```

And `build_output/*/swimlane_data/` should contain both:
- `perf_swimlane_*.json` (raw data)
- `merged_swimlane_*.json` (Perfetto-ready, auto-converted)
