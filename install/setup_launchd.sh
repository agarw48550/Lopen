#!/usr/bin/env bash
# install/setup_launchd.sh — Install Lopen as macOS launch agents
#
# Usage (from your Lopen repo root):
#   bash install/setup_launchd.sh
#
# This script:
#   1. Detects your macOS username and Lopen installation path
#   2. Copies the plist templates to ~/Library/LaunchAgents/
#   3. Substitutes the placeholder paths with your actual paths
#   4. Creates the log directory ~/Library/Logs/Lopen/
#   5. Loads the agents via launchctl
#
# To uninstall:
#   launchctl unload ~/Library/LaunchAgents/com.lopen.agent.plist
#   launchctl unload ~/Library/LaunchAgents/com.lopen.maintenance.plist
#   rm ~/Library/LaunchAgents/com.lopen.*.plist

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LAUNCHD_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs/Lopen"
USERNAME="$(whoami)"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'
ok()   { echo -e "  ${GREEN}✔${RESET}  $1"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
fail() { echo -e "  ${RED}✖${RESET}  $1"; }
info() { echo -e "     $1"; }

echo ""
echo "  Lopen — macOS Launch Agent Setup"
echo "  ================================="
echo ""

# Check macOS
if [[ "$(uname -s)" != "Darwin" ]]; then
    fail "This script is macOS-only (launchctl)"
    exit 1
fi

# Check venv exists
VENV_PYTHON="$REPO_ROOT/.venv/bin/python3"
if [[ ! -f "$VENV_PYTHON" ]]; then
    fail ".venv not found at $REPO_ROOT/.venv — run 'bash install.sh' first"
    exit 1
fi

# Create directories
mkdir -p "$LAUNCHD_DIR" "$LOG_DIR"
ok "Created $LOG_DIR"

# Helper: substitute placeholders and copy plist
install_plist() {
    local src="$1"
    local label="$2"
    local dst="$LAUNCHD_DIR/${label}.plist"

    sed \
        -e "s|/Users/REPLACE_WITH_YOUR_USERNAME/Lopen/.venv/bin/python3|$VENV_PYTHON|g" \
        -e "s|/Users/REPLACE_WITH_YOUR_USERNAME/Lopen|$REPO_ROOT|g" \
        -e "s|/Users/REPLACE_WITH_YOUR_USERNAME|$HOME|g" \
        "$src" > "$dst"

    ok "Installed $dst"
    echo "$dst"
}

# Install orchestrator agent
ORCH_PLIST=$(install_plist "$SCRIPT_DIR/lopen.plist" "com.lopen.agent")
# Install maintenance agent
MAINT_PLIST=$(install_plist "$SCRIPT_DIR/lopen-maintenance.plist" "com.lopen.maintenance")

# Unload first (in case of reinstall)
launchctl unload "$ORCH_PLIST" 2>/dev/null || true
launchctl unload "$MAINT_PLIST" 2>/dev/null || true

# Load agents
launchctl load "$ORCH_PLIST"
ok "Loaded com.lopen.agent (orchestrator auto-start)"

launchctl load "$MAINT_PLIST"
ok "Loaded com.lopen.maintenance (daily 03:00 AM cleanup)"

echo ""
echo "  Setup complete!"
echo ""
info "Orchestrator will start automatically at next login."
info "To start now:     launchctl start com.lopen.agent"
info "To stop:          launchctl stop com.lopen.agent"
info "Logs:             tail -f $LOG_DIR/orchestrator.log"
info "Maintenance log:  tail -f $LOG_DIR/maintenance.log"
info ""
info "To uninstall:"
info "  launchctl unload $ORCH_PLIST"
info "  launchctl unload $MAINT_PLIST"
echo ""
