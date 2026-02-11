#!/usr/bin/env bash
#
# play.sh — Play Pong against the trained PPO AI
#
# Usage:
#   ./play.sh                       # default model + speed
#   ./play.sh --speed 0.7           # slower ball
#   ./play.sh --weights models/ppo_final.zip   # specific model
#   ./play.sh --help                # show all options
#
# Controls:
#   UP / DOWN arrows  — Move your paddle (right side)
#   ESC               — Quit
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Check uv is available ────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# ── Auto-detect best model if none specified ──────────────────────
# Priority: models_v2 (full 5-phase) > models > models_v5 > models_v4 > models_v3
DEFAULT_WEIGHTS=""
for candidate in models_v2/ppo_final.zip models/ppo_final.zip models_v5/ppo_final.zip models_v4/ppo_final.zip models_v3/ppo_final.zip; do
    if [ -f "$candidate" ]; then
        DEFAULT_WEIGHTS="$candidate"
        break
    fi
done

if [ -z "$DEFAULT_WEIGHTS" ]; then
    echo "❌ No trained model found."
    echo "   Train one first:  uv run python scripts/train_ppo_curriculum.py"
    exit 1
fi

# ── Check if --weights was passed by user ─────────────────────────
WEIGHTS_PASSED=false
for arg in "$@"; do
    if [ "$arg" = "--weights" ]; then
        WEIGHTS_PASSED=true
        break
    fi
done

# ── Build final arguments ─────────────────────────────────────────
if [ "$WEIGHTS_PASSED" = false ]; then
    EXTRA_ARGS="--weights $DEFAULT_WEIGHTS"
    echo "🏓 Using model: $DEFAULT_WEIGHTS"
else
    EXTRA_ARGS=""
fi

# ── Launch ────────────────────────────────────────────────────────
exec uv run python scripts/play_against_model.py $EXTRA_ARGS "$@"
