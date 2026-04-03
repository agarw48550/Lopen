#!/usr/bin/env bash
# health_check.sh — Run a comprehensive health check on all Lopen services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$REPO_ROOT/.venv/bin/activate"

[ -f "$VENV" ] && source "$VENV"
cd "$REPO_ROOT"

echo "==> Lopen Health Check"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

PASS=0
FAIL=0

check_endpoint() {
    local name="$1"
    local url="$2"
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 "$url" 2>/dev/null || echo "ERR")
    if [ "$code" = "200" ]; then
        echo "  [PASS] $name ($url)"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $name ($url) — HTTP $code"
        FAIL=$((FAIL + 1))
    fi
}

echo "--- API Health ---"
check_endpoint "Orchestrator /health" "http://localhost:8000/health"
check_endpoint "Dashboard /health"    "http://localhost:8080/health"

echo ""
echo "--- Disk Space ---"
python3 -c "
from system_health.disk_check import DiskCheck
d = DiskCheck(threshold_gb=5.0)
r = d.check()
print(f'  Free: {r[\"free_gb\"]} GB / Total: {r[\"total_gb\"]} GB')
if r['free_gb'] < 5:
    print('  [WARN] Low disk space!')
else:
    print('  [PASS] Disk space OK')
" 2>/dev/null || echo "  [SKIP] Disk check unavailable"

echo ""
echo "--- RAM Usage ---"
python3 -c "
from system_health.ram_watchdog import RamWatchdog
w = RamWatchdog()
r = w.check()
print(f'  Current: {r[\"current_gb\"]} GB')
print(f'  Warning: {r[\"warning_gb\"]} GB | Critical: {r[\"critical_gb\"]} GB')
" 2>/dev/null || echo "  [SKIP] RAM check unavailable"

echo ""
echo "--- Summary ---"
echo "  Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ] && echo "  Status: ALL SYSTEMS GO" || echo "  Status: DEGRADED"
