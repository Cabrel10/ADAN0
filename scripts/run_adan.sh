#!/usr/bin/env bash
# ============================================================================
# ADAN -- Autonomous Digital Asset Navigator
# Interactive Launcher
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env if present
if [ -f .env ]; then
    set -a; source .env; set +a
fi

export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

# Colors (disabled if not a terminal)
if [ -t 1 ]; then
    BOLD="\033[1m"; DIM="\033[2m"; RESET="\033[0m"
    GREEN="\033[32m"; YELLOW="\033[33m"; CYAN="\033[36m"; RED="\033[31m"
else
    BOLD=""; DIM=""; RESET=""; GREEN=""; YELLOW=""; CYAN=""; RED=""
fi

banner() {
    echo ""
    echo -e "${BOLD}============================================================${RESET}"
    echo -e "${BOLD}  ADAN -- Autonomous Digital Asset Navigator${RESET}"
    echo -e "${DIM}  Cross-Attention Hierarchical RL | Capital Tier Supremacy${RESET}"
    echo -e "${BOLD}============================================================${RESET}"
    echo ""
}

prompt_symbols() {
    echo -e "${CYAN}Available profiles: BTCUSDT ETHUSDT XRPUSDT SOLUSDT BNBUSDT DOGEUSDT${RESET}"
    read -rp "Symbols (space-separated, default: BTCUSDT): " SYMBOLS
    SYMBOLS="${SYMBOLS:-BTCUSDT}"
}

prompt_candles() {
    read -rp "Number of 5m candles (default: 5000): " CANDLES
    CANDLES="${CANDLES:-5000}"
}

# ── Menu actions ──────────────────────────────────────────────────────────

do_generate() {
    echo ""
    echo -e "${BOLD}[1] Generate Dataset${RESET}"
    echo ""
    prompt_symbols
    prompt_candles
    read -rp "Data split [train/test/val] (default: train): " SPLIT
    SPLIT="${SPLIT:-train}"
    read -rp "Use synthetic data instead of real Binance? [y/N] (default: N = real data): " USE_SYNTH

    SYNTH_FLAG=""
    if [[ "${USE_SYNTH,,}" == "y" ]]; then
        SYNTH_FLAG="--synthetic"
        echo -e "${YELLOW}Mode: synthetic data${RESET}"
    else
        echo -e "${GREEN}Mode: real Binance data (no API key needed)${RESET}"
    fi

    echo ""
    echo -e "${GREEN}Generating ${CANDLES} candles for: ${SYMBOLS}${RESET}"
    python scripts/generate_colab_dataset.py \
        --symbols $SYMBOLS \
        --candles "$CANDLES" \
        --split "$SPLIT" \
        $SYNTH_FLAG
    echo ""
    echo -e "${GREEN}Done. Data saved to data/processed/indicators/${SPLIT}/${RESET}"
}

do_train() {
    echo ""
    echo -e "${BOLD}[2] Train Model (Ray Tune PBT -- multi-worker)${RESET}"
    echo ""
    read -rp "Training steps (default: 200000): " STEPS
    STEPS="${STEPS:-200000}"
    read -rp "Number of parallel trials (default: 4): " SAMPLES
    SAMPLES="${SAMPLES:-4}"
    read -rp "CPUs to use (default: 8): " CPUS
    CPUS="${CPUS:-8}"

    if [ ! -d "data/processed/indicators/train" ]; then
        echo -e "${RED}ERROR: No training data found. Run option [1] first.${RESET}"
        return 1
    fi

    echo ""
    echo -e "${GREEN}PBT training: ${STEPS} steps, ${SAMPLES} trials, ${CPUS} CPUs${RESET}"
    python scripts/train_parallel_agents.py \
        --config config/config.yaml \
        --steps "$STEPS" \
        --num-samples "$SAMPLES" \
        --num-cpus "$CPUS" \
        --no-subproc
    echo ""
    echo -e "${GREEN}Done.${RESET}"
}

do_train_pbt() {
    echo ""
    echo -e "${BOLD}[2b] Train Model (Ray Tune PBT -- multi-worker)${RESET}"
    echo ""
    read -rp "Training steps (default: 200000): " STEPS
    STEPS="${STEPS:-200000}"
    read -rp "Profiles [scalper intraday swing position] (default: all): " PROFILES
    PROFILES="${PROFILES:-scalper intraday swing position}"

    if [ ! -d "data/processed/indicators/train" ]; then
        echo -e "${RED}ERROR: No training data found. Run option [1] first.${RESET}"
        return 1
    fi

    echo ""
    echo -e "${GREEN}PBT training: ${STEPS} steps, profiles: ${PROFILES}${RESET}"
    python scripts/train_parallel_agents.py \
        --config config/config.yaml \
        --steps "$STEPS" \
        --profiles $PROFILES \
        --no-subproc
    echo ""
    echo -e "${GREEN}Done.${RESET}"
}

do_backtest() {
    echo ""
    echo -e "${BOLD}[3] Backtest${RESET}"
    echo ""
    read -rp "Max steps (default: 2000): " STEPS
    STEPS="${STEPS:-2000}"

    MODEL="models/rl_agents/ppo_adan_simple.zip"
    if [ ! -f "$MODEL" ]; then
        echo -e "${RED}ERROR: No model found at ${MODEL}. Run training first.${RESET}"
        return 1
    fi

    echo ""
    echo -e "${GREEN}Running backtest (${STEPS} steps)...${RESET}"
    python scripts/backtest_engine.py --model "$MODEL" --steps "$STEPS"
    echo ""
    echo -e "${GREEN}Done. Report saved to results/backtest/${RESET}"
}

do_paper_trading() {
    echo ""
    echo -e "${BOLD}[4] Paper Trading${RESET}"
    echo ""
    read -rp "Duration in minutes (default: 60): " DURATION
    DURATION="${DURATION:-60}"

    read -rp "Mode: [L]ive (Binance testnet) or [O]ffline (local data)? (default: O): " MODE
    MODE="${MODE:-O}"

    OFFLINE_FLAG=""
    if [[ "${MODE,,}" == "o" || "${MODE,,}" == "offline" ]]; then
        OFFLINE_FLAG="--offline"
    else
        if [ -z "${BINANCE_API_KEY:-}" ]; then
            echo -e "${YELLOW}WARNING: BINANCE_API_KEY not set. Paper trading needs API keys.${RESET}"
            echo -e "${YELLOW}Create a .env file from .env.example and fill in your keys.${RESET}"
            read -rp "Continue anyway? [y/N]: " CONT
            if [[ "${CONT,,}" != "y" ]]; then return 1; fi
        fi
    fi

    MODEL="models/rl_agents/ppo_adan_simple.zip"
    if [ ! -f "$MODEL" ]; then
        echo -e "${RED}ERROR: No model found at ${MODEL}. Run training first.${RESET}"
        return 1
    fi

    echo ""
    echo -e "${GREEN}Starting paper trading for ${DURATION} minutes...${RESET}"
    python scripts/paper_trading_monitor.py --duration "$DURATION" $OFFLINE_FLAG
}

# ── Main loop ─────────────────────────────────────────────────────────────

banner

while true; do
    echo -e "${BOLD}What do you want to do?${RESET}"
    echo ""
    echo "  [1] Generate Dataset   (synthetic or live, any symbol)"
    echo "  [2] Train Model        (simple PPO, single GPU/CPU)"
    echo "  [3] Run Backtest       (on trained model)"
    echo "  [4] Paper Trading      (live or offline replay)"
    echo "  [q] Quit"
    echo ""
    read -rp "Choice: " CHOICE

    case "$CHOICE" in
        1) do_generate ;;
        2) do_train ;;
        3) do_backtest ;;
        4) do_paper_trading ;;
        q|Q) echo "Exiting."; exit 0 ;;
        *) echo -e "${RED}Invalid choice.${RESET}" ;;
    esac
    echo ""
done
