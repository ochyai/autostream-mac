#!/bin/bash
# ============================================================
# autostream — Sonnet experiments → Opus review rotation
# ダブルクリックで起動 or SSH: nohup bash autostream.command &
# ============================================================
export PATH="$HOME/.local/node/bin:$PATH"
cd "$(dirname "$0")"

# Load API key from .env (not committed to git)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

SONNET_MINUTES=30
SONNET_TIMEOUT=$((SONNET_MINUTES * 60))

# ── Preflight checks ────────────────────────────────────────
if ! claude --version >/dev/null 2>&1; then
    echo "ERROR: claude CLI が見つかりません"
    echo "  → npm install -g @anthropic-ai/claude-code"
    exit 1
fi

# Quick connectivity test
CONN_CHECK=$(claude -p "echo ok" --allowedTools "" --model haiku 2>&1 | head -5)
if echo "$CONN_CHECK" | grep -qi "error\|invalid\|unauthorized"; then
    echo "ERROR: Claude API接続エラー"
    echo "$CONN_CHECK"
    exit 1
fi
echo "  Claude CLI: connected"

BRANCH=$(git branch --show-current 2>/dev/null || echo "autostream/mar16")
echo "============================================================"
echo "  autostream — Sonnet/Opus rotation"
echo "  Branch: $BRANCH"
echo "  Sonnet: ${SONNET_MINUTES}min experiments → Opus: review & plan"
echo "  Stop: Ctrl+C or kill $$"
echo "============================================================"
echo ""

ROUND=0

while true; do
    ROUND=$((ROUND + 1))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Round $ROUND — Phase 1: Sonnet experiments (${SONNET_MINUTES}min)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    claude -p "Read program.md thoroughly. You are on branch $BRANCH.

results.tsv has all experiment history. The CoreML models are in coreml_models/. The Python venv is at .venv/bin/python.

CRITICAL NEW FINDING: Channel-slimmed UNet models are available!
- slim_30 [96,192,384]: 18.2ms/54.8FPS — BEST so far
- slim_50 [160,320,640]: 21.5ms/46.6FPS
- slim_80 [256,512,1024]: 28.9ms/34.6FPS
All pass quality. Read slim_results.json for details.

To use a slim model, change pipeline.py UNet loading:
  unet_path = os.path.join(COREML_DIR, 'unet_sdxs_512_slim_30.mlpackage')

Your job: optimize pipeline.py using the slim_30 model as the default UNet.
Then apply all pipeline-level optimizations on top (pre/post processing,
buffer management, compute units, async operations).

Quality gates in benchmark.py are mandatory. If quality_pass is false, revert.

Start NOW. Edit pipeline.py, benchmark, iterate. NEVER STOP." \
        --allowedTools 'Edit,Read,Write,Bash,Glob,Grep' \
        --model sonnet 2>&1 | tee -a autostream.log &
    SONNET_PID=$!

    sleep $SONNET_TIMEOUT && kill $SONNET_PID 2>/dev/null &
    TIMER_PID=$!

    wait $SONNET_PID 2>/dev/null
    kill $TIMER_PID 2>/dev/null
    wait $TIMER_PID 2>/dev/null

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Round $ROUND — Phase 2: Opus review & strategy"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    claude -p "You are the Opus reviewer. Read:
1. results.tsv — experiment history
2. pipeline.py — current state
3. program.md — rules
4. run.log — latest benchmark
5. slim_results.json — channel-slimmed models

Analyze, audit quality, plan next round strategy.
Run benchmark yourself: .venv/bin/python benchmark.py > run.log 2>&1
Commit any changes.
Output brief summary." \
        --allowedTools 'Edit,Read,Write,Bash,Glob,Grep' \
        --model opus 2>&1 | tee -a autostream.log

    echo ""
    echo "  Round $ROUND complete."
    sleep 5
done
