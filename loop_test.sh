#!/usr/bin/env bash
# ============================================================================
# loop_test.sh — Iterative training diagnostic loop for ADAN Trading Bot
# ============================================================================
# Runs a brief training, reads rejection_reasons, counts actual trades,
# and iterates parameter tweaks until trades > 0.
#
# Usage:
#   chmod +x loop_test.sh
#   ./loop_test.sh
#
# Requirements:
#   - Python 3.8+ with ADAN installed (pip install -e .)
#   - config/config.yaml in the repo root
#   - Data in data/processed/indicators/train/BTCUSDT/
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="/tmp/adan_loop_test.log"
MAX_ITERATIONS=5
TRAINING_STEPS=2000

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW} ADAN Loop Test — Diagnostic Trainer${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

for iter in $(seq 1 $MAX_ITERATIONS); do
    echo -e "${YELLOW}--- Iteration $iter / $MAX_ITERATIONS ---${NC}"
    echo ""

    # 1. Run short training
    echo "[1/4] Running training ($TRAINING_STEPS steps)..."
    python3 scripts/train_simple_ppo.py --steps $TRAINING_STEPS > "$LOG_FILE" 2>&1 || true

    # 2. Count events
    TRADE_OPEN=$(grep -c "TRADE_OPEN" "$LOG_FILE" 2>/dev/null || echo 0)
    AGENT_CLOSE=$(grep -c "AGENT_CLOSE" "$LOG_FILE" 2>/dev/null || echo 0)
    HOLD_MIN=$(grep -c "HOLD_MIN" "$LOG_FILE" 2>/dev/null || echo 0)
    WAIT_BLOCK=$(grep -c "WAIT_BLOCK" "$LOG_FILE" 2>/dev/null || echo 0)
    EV_GATE=$(grep -c "EV_GATE" "$LOG_FILE" 2>/dev/null || echo 0)
    RISK_GATE=$(grep -c "RISK_GATE" "$LOG_FILE" 2>/dev/null || echo 0)
    SIZE_GATE=$(grep -c "SIZE_GATE" "$LOG_FILE" 2>/dev/null || echo 0)
    MAX_DURATION=$(grep -c "MAX_DURATION" "$LOG_FILE" 2>/dev/null || echo 0)
    REWARD_ANTIHACK=$(grep -c "REWARD_ANTIHACK" "$LOG_FILE" 2>/dev/null || echo 0)
    ACTION_DIFF=$(grep -c "ACTION_DIFF" "$LOG_FILE" 2>/dev/null || echo 0)
    EPISODE_REJECTIONS=$(grep -c "EPISODE_REJECTIONS" "$LOG_FILE" 2>/dev/null || echo 0)

    echo ""
    echo -e "[2/4] ${GREEN}Event Counts:${NC}"
    echo "  TRADE_OPEN:        $TRADE_OPEN"
    echo "  AGENT_CLOSE:       $AGENT_CLOSE"
    echo "  MAX_DURATION:      $MAX_DURATION"
    echo "  HOLD_MIN:          $HOLD_MIN"
    echo "  WAIT_BLOCK:        $WAIT_BLOCK"
    echo "  EV_GATE:           $EV_GATE"
    echo "  RISK_GATE:         $RISK_GATE"
    echo "  SIZE_GATE:         $SIZE_GATE"
    echo "  REWARD_ANTIHACK:   $REWARD_ANTIHACK"
    echo "  ACTION_DIFF:       $ACTION_DIFF"
    echo "  EPISODE_REJECTIONS: $EPISODE_REJECTIONS"
    echo ""

    # 3. Extract rejection_reasons from last EPISODE_REJECTIONS line
    echo -e "[3/4] ${GREEN}Last Rejection Reasons:${NC}"
    LAST_REJECTION=$(grep "EPISODE_REJECTIONS" "$LOG_FILE" 2>/dev/null | tail -1 || echo "none")
    echo "  $LAST_REJECTION"
    echo ""

    # 4. Check success criteria: at least 1 TRADE_OPEN AND at least 1 AGENT_CLOSE
    TOTAL_TRADES=$((TRADE_OPEN + AGENT_CLOSE + MAX_DURATION))
    TOTAL_CLOSES=$((AGENT_CLOSE + MAX_DURATION))

    echo -e "[4/4] ${GREEN}Summary:${NC}"
    echo "  Total opens:  $TRADE_OPEN"
    echo "  Total closes: $TOTAL_CLOSES (agent=$AGENT_CLOSE, max_dur=$MAX_DURATION)"
    echo ""

    if [ "$TRADE_OPEN" -gt 0 ] && [ "$TOTAL_CLOSES" -gt 0 ]; then
        echo -e "${GREEN}✅ SUCCESS: Agent opens AND closes positions!${NC}"
        echo "  Opens=$TRADE_OPEN, Closes=$TOTAL_CLOSES"
        echo ""

        # Show sample REWARD_ANTIHACK lines
        echo -e "${GREEN}Sample REWARD_ANTIHACK entries:${NC}"
        grep "REWARD_ANTIHACK" "$LOG_FILE" 2>/dev/null | tail -5 || echo "  (none found)"
        echo ""

        # Show ep_rew_mean
        echo -e "${GREEN}Training reward stats:${NC}"
        grep "ep_rew_mean" "$LOG_FILE" 2>/dev/null | tail -3 || echo "  (not available)"
        echo ""

        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN} LOOP TEST PASSED after $iter iteration(s)${NC}"
        echo -e "${GREEN}========================================${NC}"
        exit 0
    fi

    if [ "$TRADE_OPEN" -eq 0 ]; then
        echo -e "${RED}❌ No TRADE_OPEN events. Agent is not trading at all.${NC}"
        echo "  Possible fix: lower action_thresholds or check data length."
    elif [ "$TOTAL_CLOSES" -eq 0 ]; then
        echo -e "${RED}❌ No closes (AGENT_CLOSE=0, MAX_DURATION=0).${NC}"
        echo "  Agent opens but never closes positions."
        echo "  Possible causes: episodes too short, SELL threshold too high,"
        echo "  or anti-spam-hold blocking all SELL signals."
    fi
    echo ""

    if [ "$iter" -lt "$MAX_ITERATIONS" ]; then
        echo -e "${YELLOW}Retrying with same config (next iteration)...${NC}"
        echo ""
    fi
done

echo -e "${RED}========================================${NC}"
echo -e "${RED} LOOP TEST FAILED after $MAX_ITERATIONS iterations${NC}"
echo -e "${RED}========================================${NC}"
echo ""
echo "Diagnostics:"
echo "  - Check $LOG_FILE for full training output"
echo "  - Examine rejection_reasons above for dominant gate"
echo "  - Consider: reducing warmup_steps, increasing data size,"
echo "    lowering action_thresholds, or widening cooldown windows"
exit 1
