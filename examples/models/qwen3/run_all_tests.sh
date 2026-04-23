#!/usr/bin/env bash
# ============================================================================
# Qwen3 全量测试脚本（带泳道图采集）
#
# 用法:
#   bash examples/models/qwen3/run_all_tests.sh [-p PLATFORM] [--profiling]
#
# 示例:
#   bash examples/models/qwen3/run_all_tests.sh                       # 默认 a2a3sim，不采集泳道图
#   bash examples/models/qwen3/run_all_tests.sh --profiling           # 默认 a2a3sim，采集泳道图
#   bash examples/models/qwen3/run_all_tests.sh -p a2a3 --profiling   # 真实设备 + 泳道图
#   bash examples/models/qwen3/run_all_tests.sh -p a5sim              # A5 模拟器
# ============================================================================
set -euo pipefail

# ── 项目根目录（脚本所在位置向上三级） ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── 激活 uv 虚拟环境 ──
VENV_DIR="$PROJECT_ROOT/.venv"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: 找不到虚拟环境 $VENV_DIR/bin/activate"
    echo "请先执行: uv venv .venv --python 3.10 && uv pip install ..."
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "已激活虚拟环境: $VIRTUAL_ENV"

# ── 加载 ~/.profile 中的环境变量（PTOAS_ROOT, PTO_ISA_ROOT 等） ──
if [[ -f "$HOME/.profile" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.profile" 2>/dev/null || true
fi

# ── 加载 CANN 工具链环境（ccec 编译器等） ──
# ~/.profile 可能将 ASCEND_HOME_PATH 设置为 Ascend 安装根目录而非版本子目录，
# 导致找不到 ccec。优先 source CANN 官方 set_env.sh 来修正。
_CANN_SET_ENV="${ASCEND_HOME_PATH:-$HOME/Ascend}/cann-8.5.0/set_env.sh"
if [[ ! -f "$_CANN_SET_ENV" ]]; then
    # 如果 ASCEND_HOME_PATH 已指向版本目录，直接用它
    _CANN_SET_ENV="${ASCEND_HOME_PATH:-$HOME/Ascend}/set_env.sh"
fi
if [[ -f "$_CANN_SET_ENV" ]]; then
    # shellcheck disable=SC1090
    source "$_CANN_SET_ENV" 2>/dev/null || true
fi
unset _CANN_SET_ENV

# ── 验证关键环境变量 ──
_missing=()
[[ -z "${ASCEND_HOME_PATH:-}" ]] && _missing+=(ASCEND_HOME_PATH)
[[ -z "${PTOAS_ROOT:-}" ]]       && _missing+=(PTOAS_ROOT)
[[ -z "${PTO_ISA_ROOT:-}" ]]     && _missing+=(PTO_ISA_ROOT)
if [[ ${#_missing[@]} -gt 0 ]]; then
    echo "ERROR: 缺少环境变量: ${_missing[*]}"
    echo "请设置后重试，或检查 ~/.profile 和 CANN set_env.sh"
    exit 1
fi
if [[ ! -f "$ASCEND_HOME_PATH/bin/ccec" ]]; then
    echo "ERROR: ccec 编译器不存在: $ASCEND_HOME_PATH/bin/ccec"
    echo "请检查 ASCEND_HOME_PATH=$ASCEND_HOME_PATH 是否指向正确的 CANN 版本目录"
    exit 1
fi
unset _missing

# ── 解析参数 ──
PLATFORM="a2a3sim"
PROFILING=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -p) PLATFORM="$2"; shift 2 ;;
        --profiling) PROFILING=true; shift ;;
        *) echo "用法: $0 [-p PLATFORM] [--profiling]"; exit 1 ;;
    esac
done

EXTRA_ARGS=()
if $PROFILING; then
    EXTRA_ARGS+=(--runtime-profiling)
fi

# ── 按推荐顺序排列的测试文件（scope 级 → 完整层 → tile 变体 → prefill → 训练） ──
TESTS=(
    # Decode scope 级（最小范围，编译最快）
    qwen3_32b_decode_scope1.py
    qwen3_32b_decode_scope2.py
    qwen3_32b_decode_scope3.py
    # Decode 完整层
    qwen3_32b_decode.py
    # Decode tile / mixed 变体
    qwen3_32b_decode_scope1_tile.py
    qwen3_32b_decode_tile.py
    qwen3_32b_decode_mixed.py
    # Prefill scope 级
    qwen3_32b_prefill_scope1.py
    qwen3_32b_prefill_scope3.py
    # Prefill 完整层
    qwen3_32b_prefill.py
    qwen3_32b_prefill_tilelet.py
    # 训练
    qwen3_32b_training_forward_and_backward.py
)

# ── 执行测试 ──
TOTAL=${#TESTS[@]}
PASSED=0
FAILED=0
FAILED_LIST=()

echo ""
echo "=========================================="
echo " Qwen3 全量测试  平台: $PLATFORM"
if $PROFILING; then
    echo " 泳道图采集: 已开启 (--runtime-profiling)"
else
    echo " 泳道图采集: 未开启 (如需开启请传入 --profiling)"
fi
echo " 测试数量: $TOTAL"
echo "=========================================="
echo ""

for i in "${!TESTS[@]}"; do
    TEST_FILE="$SCRIPT_DIR/${TESTS[$i]}"
    SEQ=$((i + 1))

    if [[ ! -f "$TEST_FILE" ]]; then
        echo "[$SEQ/$TOTAL] SKIP (文件不存在): ${TESTS[$i]}"
        continue
    fi

    echo "[$SEQ/$TOTAL] 运行: ${TESTS[$i]} ..."
    echo "  命令: python $TEST_FILE -p $PLATFORM ${EXTRA_ARGS[*]:-}"
    echo "  开始: $(date '+%Y-%m-%d %H:%M:%S')"

    if python "$TEST_FILE" -p "$PLATFORM" "${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}}"; then
        echo "  结果: PASS"
        PASSED=$((PASSED + 1))
    else
        echo "  结果: FAIL"
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("${TESTS[$i]}")
    fi

    echo "  结束: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
done

# ── 汇总 ──
echo "=========================================="
echo " 测试汇总"
echo "=========================================="
echo " 平台:   $PLATFORM"
echo " 总计:   $TOTAL"
echo " 通过:   $PASSED"
echo " 失败:   $FAILED"
if [[ $FAILED -gt 0 ]]; then
    echo " 失败项:"
    for f in "${FAILED_LIST[@]}"; do
        echo "   - $f"
    done
fi
echo "=========================================="
if $PROFILING; then
    echo ""
    echo "泳道图产物位于各测试的 build_output/<name>_<timestamp>/swimlane_data/ 目录下"
    echo "查看方式: 在浏览器中打开 https://ui.perfetto.dev/ 加载 perf_swimlane_*.json 或 merged_swimlane_*.json"
fi

exit $FAILED
