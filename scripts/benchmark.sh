#!/usr/bin/env bash
# benchmark.sh — Lopen inference speed and response quality benchmark
#
# Sends a series of prompts to the running Lopen orchestrator and measures
# response latency. Use this to verify that your LLM setup meets the
# performance targets for a 2017 Intel MacBook Pro.
#
# Usage:
#   bash scripts/benchmark.sh              # standard benchmark
#   bash scripts/benchmark.sh --verbose    # show full responses

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$REPO_ROOT/.venv/bin/activate"

[ -f "$VENV" ] && source "$VENV"

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

VERBOSE=false
for arg in "$@"; do [[ "$arg" == "--verbose" ]] && VERBOSE=true; done

ORCH_URL="${LOPEN_ORCHESTRATOR_URL:-http://localhost:8000}"

# ---------------------------------------------------------------------------
# Check orchestrator is running
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${CYAN}  Lopen Inference Benchmark${RESET}"
echo -e "${DIM}  ────────────────────────────────────────${RESET}"
echo ""

HEALTH=$(curl -sf "${ORCH_URL}/health" 2>/dev/null || echo "")
if [[ -z "$HEALTH" ]]; then
    echo -e "${RED}  ✖  Orchestrator not reachable at ${ORCH_URL}${RESET}"
    echo -e "${DIM}     Start it with: bash scripts/start.sh${RESET}"
    exit 1
fi

LLM_MODE=$(curl -sf "${ORCH_URL}/status" 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('llm_mode','?'))" 2>/dev/null || echo "?")
echo -e "  Orchestrator: ${GREEN}online${RESET}"
echo -e "  LLM mode:     ${CYAN}${LLM_MODE}${RESET}"
echo ""

# ---------------------------------------------------------------------------
# Benchmark prompts
# ---------------------------------------------------------------------------
declare -a PROMPTS=(
    "What is 2 + 2?"
    "Name the capital of France."
    "Summarise the water cycle in one sentence."
    "Write a Python function that returns the factorial of n."
    "List 3 best practices for writing secure code."
    "Explain what a large language model is in simple terms."
)

TOTAL=0
PASSED=0
FAILED=0
MIN_TIME=999
MAX_TIME=0

echo -e "${BOLD}  Running ${#PROMPTS[@]} benchmark prompts…${RESET}"
echo -e "${DIM}  ────────────────────────────────────────${RESET}"

for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$i]}"
    IDX=$((i+1))

    START_NS=$(python3 -c "import time; print(int(time.time()*1000))")

    RESULT=$(curl -sf -X POST "${ORCH_URL}/chat" \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"${PROMPT}\", \"session_id\": \"benchmark-$$\"}" \
        2>/dev/null || echo "{\"response\": \"ERROR\", \"error\": true}")

    END_NS=$(python3 -c "import time; print(int(time.time()*1000))")
    ELAPSED_MS=$(( END_NS - START_NS ))
    ELAPSED_S=$(echo "scale=2; $ELAPSED_MS / 1000" | bc)

    RESPONSE=$(echo "$RESULT" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('response','ERROR')[:80])" 2>/dev/null || echo "ERROR")

    # Timing colour
    if (( ELAPSED_MS < 3000 )); then
        TIME_COLOR=$GREEN
        RATING="🚀"
    elif (( ELAPSED_MS < 8000 )); then
        TIME_COLOR=$YELLOW
        RATING="✅"
    elif (( ELAPSED_MS < 15000 )); then
        TIME_COLOR=$YELLOW
        RATING="⚠️"
    else
        TIME_COLOR=$RED
        RATING="🐢"
    fi

    if [[ "$RESPONSE" != "ERROR" ]]; then
        PASSED=$((PASSED+1))
        STATUS="${GREEN}✔${RESET}"
    else
        FAILED=$((FAILED+1))
        STATUS="${RED}✖${RESET}"
    fi

    TOTAL=$((TOTAL + ELAPSED_MS))
    (( ELAPSED_MS < MIN_TIME )) && MIN_TIME=$ELAPSED_MS
    (( ELAPSED_MS > MAX_TIME )) && MAX_TIME=$ELAPSED_MS

    printf "  [%d/%d] %s  ${TIME_COLOR}%ss${RESET}  %s\n" \
        "$IDX" "${#PROMPTS[@]}" "$STATUS" "$ELAPSED_S" "$RATING"
    echo -e "  ${DIM}Prompt: ${PROMPT:0:55}${RESET}"

    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "  ${DIM}Response: ${RESPONSE:0:100}${RESET}"
    fi
    echo ""
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if [[ "${#PROMPTS[@]}" -gt 0 ]]; then
    AVG_MS=$(( TOTAL / ${#PROMPTS[@]} ))
    AVG_S=$(echo "scale=2; $AVG_MS / 1000" | bc)
    MIN_S=$(echo "scale=2; $MIN_TIME / 1000" | bc)
    MAX_S=$(echo "scale=2; $MAX_TIME / 1000" | bc)
fi

echo -e "${DIM}  ────────────────────────────────────────${RESET}"
echo -e "${BOLD}  Results:${RESET}"
echo -e "  Passed:   ${GREEN}${PASSED}${RESET} / ${#PROMPTS[@]}"
echo -e "  Failed:   ${FAILED}"
echo -e "  Avg time: ${AVG_S}s  (min: ${MIN_S}s, max: ${MAX_S}s)"

if (( AVG_MS < 3000 )); then
    echo -e "  Rating:   ${GREEN}🚀 Excellent — LLM is fast on this machine${RESET}"
elif (( AVG_MS < 8000 )); then
    echo -e "  Rating:   ${YELLOW}✅ Good — acceptable for a 2017 Intel Mac${RESET}"
elif (( AVG_MS < 15000 )); then
    echo -e "  Rating:   ${YELLOW}⚠️  Slow — consider a smaller GGUF model (Q4_K_M or Q3_K_M)${RESET}"
else
    echo -e "  Rating:   ${RED}🐢 Too slow — check model size and try: engine: llama_cpp${RESET}"
fi

echo ""
echo -e "${DIM}  Tips to improve speed:${RESET}"
echo -e "${DIM}  • Use Phi-3-mini Q4_K_M (~2.2GB) for best speed/quality balance${RESET}"
echo -e "${DIM}  • Set llm.memory_conservative: true in config/settings.yaml${RESET}"
echo -e "${DIM}  • Close browser tabs and other heavy apps before benchmarking${RESET}"
echo -e "${DIM}  • Run: bash scripts/diagnose.sh to check system health${RESET}"
echo ""
