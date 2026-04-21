#!/usr/bin/env bash
# ============================================================================
# run_comparison.sh — Batch comparison test: original vs optimized scopes
#
# Usage:
#   bash examples/models/qwen-opt/run_comparison.sh [platform] [device]
#
# Examples:
#   bash examples/models/qwen-opt/run_comparison.sh a2a3sim       # simulator
#   bash examples/models/qwen-opt/run_comparison.sh a2a3 0        # real NPU
#   bash examples/models/qwen-opt/run_comparison.sh a2a3sim 0 --runtime-profiling
#
# Must be run from pypto-lib repository root.
# ============================================================================

set -euo pipefail

PLATFORM="${1:-a2a3sim}"
DEVICE="${2:-0}"
shift 2 2>/dev/null || true
EXTRA_ARGS="$*"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Verify we are in repo root ──
if [[ ! -f "pyproject.toml" ]] || [[ ! -d "examples/models/qwen-opt" ]]; then
    echo -e "${RED}[ERROR]${NC} Must run from pypto-lib repository root."
    exit 1
fi

# ── Define original → optimized pairs ──
# Format: "original_path|optimized_path"
PAIRS=(
    "examples/models/qwen3/qwen3_32b_decode_scope1.py|examples/models/qwen-opt/qwen3_32b_decode_scope1_opt.py"
    "examples/models/qwen3/qwen3_32b_decode_scope1_tile.py|examples/models/qwen-opt/qwen3_32b_decode_scope1_tile_opt.py"
    "examples/models/qwen3/qwen3_32b_decode_scope2.py|examples/models/qwen-opt/qwen3_32b_decode_scope2_opt.py"
    "examples/models/qwen3/qwen3_32b_decode_scope3.py|examples/models/qwen-opt/qwen3_32b_decode_scope3_opt.py"
)

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

for pair in "${PAIRS[@]}"; do
    orig="${pair%%|*}"
    opt="${pair##*|}"
    scope_name=$(basename "$orig" .py)

    echo ""
    echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  Testing: ${scope_name}${NC}"
    echo -e "${BOLD}══════════════════════════════════════════════════${NC}"

    # Check files exist
    if [[ ! -f "$orig" ]]; then
        echo -e "${RED}[SKIP]${NC} Original not found: $orig"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi
    if [[ ! -f "$opt" ]]; then
        echo -e "${RED}[SKIP]${NC} Optimized not found: $opt"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    # ── Run original ──
    echo -e "${CYAN}── Original: $orig ──${NC}"
    if ! python "$orig" -p "$PLATFORM" -d "$DEVICE" $EXTRA_ARGS; then
        echo -e "${RED}[FAIL]${NC} Original failed: $orig"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # ── Find golden data (most recent build_output with data/ dir) ──
    GOLDEN=$(ls -td build_output/*/data 2>/dev/null | head -1)
    if [[ -z "$GOLDEN" ]] || [[ ! -d "$GOLDEN" ]]; then
        echo -e "${RED}[FAIL]${NC} Could not locate golden data directory"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    echo -e "${CYAN}   Golden data: $GOLDEN${NC}"

    # ── Run optimized with same golden data ──
    echo -e "${CYAN}── Optimized: $opt ──${NC}"
    if python "$opt" -p "$PLATFORM" -d "$DEVICE" --golden-data "$GOLDEN" $EXTRA_ARGS; then
        echo -e "${GREEN}[PASS]${NC} $scope_name — both versions match"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo -e "${RED}[FAIL]${NC} Optimized failed: $opt"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

# ── Summary ──
echo ""
echo -e "${BOLD}══════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}PASS: $PASS_COUNT${NC}  ${RED}FAIL: $FAIL_COUNT${NC}  SKIP: $SKIP_COUNT"
echo -e "${BOLD}══════════════════════════════════════════════════${NC}"

[[ $FAIL_COUNT -eq 0 ]] && exit 0 || exit 1
