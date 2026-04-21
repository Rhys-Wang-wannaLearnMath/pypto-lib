#!/usr/bin/env bash
# ============================================================================
# setup_npu_env.sh — Configure pypto-lib environment variables for NPU devices
#
# This script:
#   1. Auto-detects the CANN toolkit installation path
#   2. Verifies PTOAS_ROOT and PTO_ISA_ROOT directories exist
#   3. Detects available NPU device(s) via npu-smi
#   4. Persists environment variables into ~/.bashrc (before the early-return guard)
#   5. Runs a final verification check
#
# Usage:
#   bash setup_npu_env.sh                          # auto-detect everything
#   bash setup_npu_env.sh --cann /path/to/cann     # manually specify CANN path
#   bash setup_npu_env.sh --dry-run                 # preview without writing
#   bash setup_npu_env.sh --uninstall               # remove env block from ~/.bashrc
# ============================================================================

set -euo pipefail

# ── Color helpers ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Constants ──
ENV_MARKER_BEGIN="# ── pypto-lib environment (inserted by setup_npu_env.sh) ──"
ENV_MARKER_END="# ── end pypto-lib environment ──"
BASHRC="$HOME/.bashrc"

# ── Parse arguments ──
DRY_RUN=false
UNINSTALL=false
MANUAL_CANN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --uninstall) UNINSTALL=true; shift ;;
        --cann)      MANUAL_CANN="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash setup_npu_env.sh [--cann /path/to/cann] [--dry-run] [--uninstall]"
            exit 0
            ;;
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Helper: remove env block using fixed-string line matching ──
remove_env_block() {
    local file="$1"
    # Use awk for reliable fixed-string matching (sed regex can break on special chars)
    awk -v begin="$ENV_MARKER_BEGIN" -v end="$ENV_MARKER_END" '
        $0 == begin { skip=1; next }
        $0 == end   { skip=0; next }
        !skip
    ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
}

# ── Uninstall mode ──
if $UNINSTALL; then
    if [[ ! -f "$BASHRC" ]]; then
        warn "$BASHRC does not exist, nothing to remove."
        exit 0
    fi
    if grep -qF "$ENV_MARKER_BEGIN" "$BASHRC"; then
        remove_env_block "$BASHRC"
        ok "Removed pypto-lib environment block from $BASHRC"
    else
        info "No pypto-lib environment block found in $BASHRC, nothing to remove."
    fi
    exit 0
fi

echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  pypto-lib NPU Environment Setup${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo

# ============================================================================
# Step 0: Container capability check (NPU kernel execution needs CAP_SYS_RAWIO)
# ============================================================================
info "Step 0: Checking container capabilities..."

if [[ -f /proc/1/status ]]; then
    cap_bnd_hex=$(grep '^CapBnd:' /proc/1/status | awk '{print $2}')
    cap_bnd=$((16#${cap_bnd_hex}))
    # CAP_SYS_RAWIO = bit 17
    if (( (cap_bnd >> 17) & 1 )); then
        ok "CAP_SYS_RAWIO present in Bounding Set"
    else
        warn "CAP_SYS_RAWIO (bit 17) is NOT in container Bounding Set (CapBnd: 0x$cap_bnd_hex)"
        warn "NPU kernel execution will HANG without this capability."
        warn "Simulator tests (-p a2a3sim) will still work."
        warn "To fix: restart container with --privileged or --cap-add=SYS_RAWIO"
        echo
    fi
else
    info "Cannot read /proc/1/status — skipping capability check."
fi
echo

# ============================================================================
# Step 1: Detect CANN installation path
# ============================================================================
info "Step 1: Detecting CANN installation path..."

detect_cann_path() {
    # Priority order:
    #   1. Manual --cann argument
    #   2. /usr/local/Ascend/ascend-toolkit/latest  (symlink, most common)
    #   3. /usr/local/Ascend/ascend-toolkit          (without latest/)
    #   4. /usr/local/Ascend/cann-*                  (versioned, pick newest)
    #   5. Fallback: search for set_env.sh

    if [[ -n "$MANUAL_CANN" ]]; then
        if [[ -d "$MANUAL_CANN" ]]; then
            echo "$MANUAL_CANN"
            return 0
        else
            err "--cann path does not exist: $MANUAL_CANN"
            return 1
        fi
    fi

    local ascend_base="/usr/local/Ascend"

    if [[ ! -d "$ascend_base" ]]; then
        err "Ascend base directory $ascend_base does not exist."
        err "Is CANN installed on this machine?"
        return 1
    fi

    # Try ascend-toolkit/latest
    if [[ -d "$ascend_base/ascend-toolkit/latest" ]]; then
        echo "$ascend_base/ascend-toolkit/latest"
        return 0
    fi

    # Try ascend-toolkit (without latest/)
    if [[ -d "$ascend_base/ascend-toolkit" ]] && [[ -f "$ascend_base/ascend-toolkit/set_env.sh" ]]; then
        echo "$ascend_base/ascend-toolkit"
        return 0
    fi

    # Try cann-* directories (pick the one with highest version)
    local cann_dirs
    cann_dirs=$(find "$ascend_base" -maxdepth 1 -type d -name 'cann-*' 2>/dev/null | sort -V | tail -1)
    if [[ -n "$cann_dirs" ]] && [[ -d "$cann_dirs" ]]; then
        echo "$cann_dirs"
        return 0
    fi

    # Fallback: find set_env.sh under ascend-toolkit or cann-*
    local set_env
    set_env=$(find "$ascend_base" -maxdepth 3 -name "set_env.sh" -path "*/ascend-toolkit/*" 2>/dev/null | head -1)
    if [[ -n "$set_env" ]]; then
        echo "$(dirname "$set_env")"
        return 0
    fi

    set_env=$(find "$ascend_base" -maxdepth 3 -name "set_env.sh" -path "*/cann-*" 2>/dev/null | head -1)
    if [[ -n "$set_env" ]]; then
        echo "$(dirname "$set_env")"
        return 0
    fi

    err "Could not auto-detect CANN path under $ascend_base"
    err "Please re-run with: bash setup_npu_env.sh --cann /path/to/your/cann"
    return 1
}

CANN_PATH=$(detect_cann_path)
ok "CANN path: $CANN_PATH"

# Verify it looks valid (should contain bin/ or set_env.sh)
if [[ -f "$CANN_PATH/set_env.sh" ]]; then
    ok "Found set_env.sh at $CANN_PATH/set_env.sh"
elif [[ -d "$CANN_PATH/bin" ]]; then
    ok "Found bin/ directory at $CANN_PATH/bin"
else
    warn "CANN path has no set_env.sh or bin/ — might be incorrect."
    warn "You can re-run with --cann to specify manually."
fi
echo

# ============================================================================
# Step 2: Check PTOAS_ROOT and PTO_ISA_ROOT
# ============================================================================
info "Step 2: Checking PTOAS_ROOT and PTO_ISA_ROOT..."

PTOAS_DIR="$HOME/ptoas-bin"
PTO_ISA_DIR="$HOME/pto-isa"

check_ok=true

if [[ -d "$PTOAS_DIR" ]]; then
    if [[ -x "$PTOAS_DIR/ptoas" ]] || [[ -x "$PTOAS_DIR/bin/ptoas" ]]; then
        ok "PTOAS_ROOT: $PTOAS_DIR (binary found)"
    else
        warn "PTOAS_ROOT directory exists ($PTOAS_DIR) but ptoas binary not found/not executable."
        warn "Expected: $PTOAS_DIR/ptoas or $PTOAS_DIR/bin/ptoas"
        check_ok=false
    fi
else
    err "PTOAS_ROOT directory not found: $PTOAS_DIR"
    err "Please install ptoas first (see guide section 2.2)."
    check_ok=false
fi

if [[ -d "$PTO_ISA_DIR" ]]; then
    if [[ -d "$PTO_ISA_DIR/.git" ]]; then
        local_commit=$(cd "$PTO_ISA_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        ok "PTO_ISA_ROOT: $PTO_ISA_DIR (commit: $local_commit)"
    else
        ok "PTO_ISA_ROOT: $PTO_ISA_DIR"
    fi
else
    err "PTO_ISA_ROOT directory not found: $PTO_ISA_DIR"
    err "Please clone pto-isa first (see guide section 2.3)."
    check_ok=false
fi

if ! $check_ok; then
    err "Pre-checks failed. Fix the issues above before re-running."
    exit 1
fi
echo

# ============================================================================
# Step 3: Detect NPU devices
# ============================================================================
info "Step 3: Detecting NPU devices..."

if command -v npu-smi &>/dev/null; then
    # Extract NPU IDs from npu-smi info output
    # The NPU ID is the first number in lines like: "| 2     910B4  ..."
    npu_ids=$(npu-smi info 2>/dev/null | grep -E '^\|\s*[0-9]+\s+[0-9A-Za-z]+' | awk '{print $2}' || true)

    if [[ -n "$npu_ids" ]]; then
        npu_count=$(echo "$npu_ids" | wc -l)
        npu_list=$(echo "$npu_ids" | tr '\n' ',' | sed 's/,$//')
        ok "Found $npu_count NPU device(s) (npu-smi physical IDs: $npu_list)"
        if [[ $npu_count -eq 1 ]]; then
            info "Single-card server: ACL logical device ID is always 0 (regardless of physical ID)"
            info "Use -d 0 when running tests, e.g.:"
            info "  python examples/models/qwen3/qwen3_32b_decode_scope1.py -p a2a3 -d 0"
        else
            info "Multi-card server: ACL logical device IDs are 0..$(( npu_count - 1 )), mapped by ascending physical ID"
            info "Use -d 0 for the first card, e.g.:"
            info "  python examples/models/qwen3/qwen3_32b_decode_scope1.py -p a2a3 -d 0"
        fi
    else
        warn "npu-smi is available but no NPU devices detected."
        warn "Check hardware and CANN driver installation."
    fi
else
    warn "npu-smi command not found. Cannot detect NPU devices."
    warn "Make sure CANN is installed and PATH includes its bin/ directory."
fi
echo

# ============================================================================
# Step 4: Generate environment block
# ============================================================================
info "Step 4: Generating environment variable block..."

# Check if CANN provides set_env.sh (sets LD_LIBRARY_PATH etc.)
CANN_SET_ENV=""
if [[ -f "$CANN_PATH/set_env.sh" ]]; then
    CANN_SET_ENV="source $CANN_PATH/set_env.sh"
    ok "Will source $CANN_PATH/set_env.sh for LD_LIBRARY_PATH etc."
fi

if [[ -n "$CANN_SET_ENV" ]]; then
    ENV_BLOCK="$ENV_MARKER_BEGIN
export PTOAS_ROOT=$PTOAS_DIR
export PTO_ISA_ROOT=$PTO_ISA_DIR
export ASCEND_HOME_PATH=$CANN_PATH
$CANN_SET_ENV
export PATH=\$ASCEND_HOME_PATH/bin:\$PTOAS_ROOT:\$PATH
$ENV_MARKER_END"
else
    ENV_BLOCK="$ENV_MARKER_BEGIN
export PTOAS_ROOT=$PTOAS_DIR
export PTO_ISA_ROOT=$PTO_ISA_DIR
export ASCEND_HOME_PATH=$CANN_PATH
export PATH=\$ASCEND_HOME_PATH/bin:\$PTOAS_ROOT:\$PATH
$ENV_MARKER_END"
fi

echo -e "${CYAN}────────────────────────────────────────${NC}"
echo "$ENV_BLOCK"
echo -e "${CYAN}────────────────────────────────────────${NC}"
echo

# ============================================================================
# Step 5: Write to ~/.bashrc
# ============================================================================
if $DRY_RUN; then
    info "[DRY-RUN] Would insert the above block into $BASHRC (before early-return guard)."
    info "[DRY-RUN] No changes made."
    exit 0
fi

info "Step 5: Writing to $BASHRC..."

# Create ~/.bashrc if it doesn't exist
if [[ ! -f "$BASHRC" ]]; then
    touch "$BASHRC"
    info "Created $BASHRC (did not exist)."
fi

# Remove old block if present (idempotent)
if grep -qF "$ENV_MARKER_BEGIN" "$BASHRC"; then
    remove_env_block "$BASHRC"
    info "Removed previous pypto-lib environment block."
fi

# Back up
cp "$BASHRC" "$BASHRC.bak.$(date +%Y%m%d%H%M%S)"
ok "Backup created: $BASHRC.bak.*"

# Insert at the very top of ~/.bashrc (before any [ -z "\$PS1" ] && return)
{
    echo "$ENV_BLOCK"
    echo ""
    cat "$BASHRC"
} > "$BASHRC.tmp" && mv "$BASHRC.tmp" "$BASHRC"

ok "Environment block inserted at the top of $BASHRC"
echo

# ============================================================================
# Step 6: Verification
# ============================================================================
info "Step 6: Verifying..."

# Source the bashrc in a subshell to check
verify_output=$(bash -c "source $BASHRC 2>/dev/null && echo PTOAS_ROOT=\$PTOAS_ROOT && echo PTO_ISA_ROOT=\$PTO_ISA_ROOT && echo ASCEND_HOME_PATH=\$ASCEND_HOME_PATH" 2>/dev/null || true)

all_ok=true

check_var() {
    local name="$1"
    local expected="$2"
    local actual
    actual=$(echo "$verify_output" | grep "^${name}=" | cut -d= -f2-)
    if [[ "$actual" == "$expected" ]]; then
        ok "$name=$actual"
    else
        err "$name expected '$expected', got '$actual'"
        all_ok=false
    fi
}

check_var "PTOAS_ROOT"      "$PTOAS_DIR"
check_var "PTO_ISA_ROOT"    "$PTO_ISA_DIR"
check_var "ASCEND_HOME_PATH" "$CANN_PATH"

echo
if $all_ok; then
    echo -e "${GREEN}${BOLD}✓ All checks passed!${NC}"
    echo
    info "To apply in your current terminal, run:"
    echo -e "  ${BOLD}source ~/.bashrc${NC}"
    echo
    info "New terminals will pick up the variables automatically."
else
    err "Some checks failed. Please review $BASHRC manually."
    exit 1
fi
