#!/usr/bin/env bash
# system_health/maintenance.sh — Daily Lopen maintenance routine
#
# Runs automatically via launchd (install/lopen-maintenance.plist) or manually:
#   bash system_health/maintenance.sh
#
# Tasks performed:
#   1. Rotate logs older than 7 days
#   2. Vacuum SQLite databases (reclaim space)
#   3. Clear Python __pycache__ directories
#   4. Clear macOS system caches (user-level, safe to remove)
#   5. Report disk usage
#   6. Print health summary to console (and log file)
#
# Safe to run daily without disrupting a running Lopen instance.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_ROOT/logs"
STORAGE_DIR="$REPO_ROOT/storage"
MAINT_LOG="$LOG_DIR/maintenance.log"
TS="$(date '+%Y-%m-%d %H:%M:%S')"

mkdir -p "$LOG_DIR"

log() { echo "[$TS] $*" | tee -a "$MAINT_LOG"; }
ok()  { echo "    ✔  $*" | tee -a "$MAINT_LOG"; }
warn(){ echo "    ⚠  $*" | tee -a "$MAINT_LOG"; }
info(){ echo "    ℹ  $*" | tee -a "$MAINT_LOG"; }

log "=== Lopen daily maintenance started ==="

# ---------------------------------------------------------------------------
# 1. Rotate old log files (keep last 7 days)
# ---------------------------------------------------------------------------
log "[1/6] Rotating old logs..."
LOGS_REMOVED=0
if [ -d "$LOG_DIR" ]; then
    while IFS= read -r -d '' f; do
        rm -f "$f"
        LOGS_REMOVED=$((LOGS_REMOVED + 1))
    done < <(find "$LOG_DIR" -name "*.log" -mtime +7 -print0 2>/dev/null || true)
    while IFS= read -r -d '' f; do
        rm -f "$f"
        LOGS_REMOVED=$((LOGS_REMOVED + 1))
    done < <(find "$LOG_DIR" -name "*.log.gz" -mtime +30 -print0 2>/dev/null || true)
fi
ok "Logs rotated: $LOGS_REMOVED old file(s) removed"

# ---------------------------------------------------------------------------
# 2. Vacuum SQLite databases
# ---------------------------------------------------------------------------
log "[2/6] Vacuuming SQLite databases..."
DB_COUNT=0
if [ -d "$STORAGE_DIR" ]; then
    while IFS= read -r -d '' db; do
        sqlite3 "$db" "VACUUM;" 2>/dev/null && DB_COUNT=$((DB_COUNT + 1)) || warn "Vacuum failed: $db"
    done < <(find "$STORAGE_DIR" -name "*.db" -print0 2>/dev/null || true)
fi
ok "SQLite vacuum complete: $DB_COUNT database(s)"

# ---------------------------------------------------------------------------
# 3. Clear Python __pycache__ directories
# ---------------------------------------------------------------------------
log "[3/6] Clearing Python caches..."
CACHE_COUNT=0
while IFS= read -r -d '' d; do
    rm -rf "$d"
    CACHE_COUNT=$((CACHE_COUNT + 1))
done < <(find "$REPO_ROOT" -type d -name "__pycache__" -not -path "*/\.venv/*" -print0 2>/dev/null || true)
while IFS= read -r -d '' f; do
    rm -f "$f"
done < <(find "$REPO_ROOT" -name "*.pyc" -not -path "*/\.venv/*" -print0 2>/dev/null || true)
ok "Python caches cleared: $CACHE_COUNT __pycache__ dir(s)"

# ---------------------------------------------------------------------------
# 4. Clear macOS user-level caches (safe — does not affect other apps)
# ---------------------------------------------------------------------------
log "[4/6] Clearing macOS user caches..."
if [ "$(uname -s)" = "Darwin" ]; then
    # Clear Lopen-specific macOS cache entries only (not system-wide)
    LOPEN_CACHE_DIR="$HOME/Library/Caches/Lopen"
    if [ -d "$LOPEN_CACHE_DIR" ]; then
        find "$LOPEN_CACHE_DIR" -mtime +7 -delete 2>/dev/null || true
        ok "Cleared stale entries from $LOPEN_CACHE_DIR"
    fi

    # Clear Playwright browser cache (can grow large)
    PW_CACHE="$HOME/Library/Caches/ms-playwright"
    if [ -d "$PW_CACHE" ]; then
        BEFORE=$(du -sk "$PW_CACHE" 2>/dev/null | awk '{print $1}')
        find "$PW_CACHE" -name "*.log" -mtime +7 -delete 2>/dev/null || true
        AFTER=$(du -sk "$PW_CACHE" 2>/dev/null | awk '{print $1}')
        FREED=$(( (BEFORE - AFTER) / 1024 ))
        ok "Playwright cache cleaned (freed ~${FREED}MB)"
    fi
else
    info "Not macOS — skipping macOS cache cleanup"
fi

# ---------------------------------------------------------------------------
# 5. Disk usage report
# ---------------------------------------------------------------------------
log "[5/6] Disk usage check..."
FREE_GB=$(df -g / 2>/dev/null | awk 'NR==2 {print $4}' || echo "?")
REPO_SIZE=$(du -sh "$REPO_ROOT" 2>/dev/null | awk '{print $1}' || echo "?")
MODELS_SIZE=$(du -sh "$REPO_ROOT/models" 2>/dev/null | awk '{print $1}' || echo "?")
LOGS_SIZE=$(du -sh "$LOG_DIR" 2>/dev/null | awk '{print $1}' || echo "?")

info "Root disk free   : ${FREE_GB} GB"
info "Lopen repo       : ${REPO_SIZE}"
info "Models           : ${MODELS_SIZE}"
info "Logs             : ${LOGS_SIZE}"

if [ "$FREE_GB" != "?" ] && [ "$FREE_GB" -lt 5 ] 2>/dev/null; then
    warn "Disk space low (${FREE_GB} GB free) — consider deleting old model files"
fi

# ---------------------------------------------------------------------------
# 6. RAM usage snapshot (if psutil available via Python)
# ---------------------------------------------------------------------------
log "[6/6] System health snapshot..."

PYTHON_BIN="$REPO_ROOT/.venv/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

RAM_INFO=$("$PYTHON_BIN" - 2>/dev/null <<'PYEOF'
try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"RAM: {vm.used/1e9:.1f}/{vm.total/1e9:.1f} GB used ({vm.percent:.0f}%)")
    if vm.percent > 85:
        print("WARNING: RAM usage high — consider restarting Lopen services")
except ImportError:
    print("RAM: psutil not available")
PYEOF
) || RAM_INFO="RAM check skipped"

info "$RAM_INFO"

log "=== Maintenance complete ==="
echo ""
echo "Maintenance log: $MAINT_LOG"
