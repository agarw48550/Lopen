#!/usr/bin/env bash
# status.sh — Show Lopen service status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$(dirname "$SCRIPT_DIR")/.pids"

echo "==> Lopen service status"
echo ""

services=("orchestrator" "dashboard" "voice" "whatsapp")
for svc in "${services[@]}"; do
    pidfile="$PID_DIR/${svc}.pid"
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "  [RUNNING] $svc (PID $pid)"
        else
            echo "  [DEAD]    $svc (PID $pid — stale)"
        fi
    else
        echo "  [STOPPED] $svc"
    fi
done

echo ""
echo "--> Health endpoints:"
for url in "http://localhost:8000/health" "http://localhost:8080/health"; do
    code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "$url" 2>/dev/null || echo "ERR")
    if [ "$code" = "200" ]; then
        echo "  [OK]   $url"
    else
        echo "  [FAIL] $url (HTTP $code)"
    fi
done
