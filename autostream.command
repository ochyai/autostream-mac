#!/bin/bash
# ============================================================
# autostream — Sonnet experiments → Opus review rotation
# ダブルクリックで起動
# ============================================================
export PATH="$HOME/.local/node/bin:$PATH"
cd "$(dirname "$0")"

SONNET_MINUTES=30
SONNET_TIMEOUT=$((SONNET_MINUTES * 60))

# ── Preflight checks ────────────────────────────────────────
if ! claude --version >/dev/null 2>&1; then
    echo "ERROR: claude CLI が見つかりません"
    echo "  → npm install -g @anthropic-ai/claude-code"
    read -p "Press Enter to close..."
    exit 1
fi

LOGIN_CHECK=$(claude -p "echo ok" --allowedTools "" --model haiku 2>&1 | head -5)
if echo "$LOGIN_CHECK" | grep -qi "not logged in\|login\|unauthorized\|authenticate"; then
    echo "ERROR: Claude CLI にログインしていません"
    echo "  → export PATH=\$HOME/.local/node/bin:\$PATH && claude /login"
    read -p "Press Enter to close..."
    exit 1
fi

BRANCH=$(git branch --show-current 2>/dev/null || echo "autostream/mar16")
echo "============================================================"
echo "  autostream — Sonnet/Opus rotation"
echo "  Branch: $BRANCH"
echo "  Sonnet: ${SONNET_MINUTES}min experiments → Opus: review & plan"
echo "  Stop: Ctrl+C or close this window"
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

    # macOS has no `timeout` — use background + sleep + kill
    claude -p "Read program.md thoroughly. You are on branch $BRANCH.

results.tsv has all experiment history. The CoreML models are in coreml_models/. The Python venv is at .venv/bin/python.

IMPORTANT: benchmark.py now includes QUALITY CHECKS. After speed measurement, it checks:
- input_variance >= 2.0 (outputs must differ for different inputs)
- unique_colors >= 100 (no solid-color outputs)
- spatial_stddev >= 10.0 (output must have spatial structure)
If quality_pass is false, the experiment MUST be treated as a crash/qfail.

Check quality with: grep 'quality_pass:\|QUALITY_FAIL' run.log

Start the experiment loop NOW. Edit pipeline.py, run '.venv/bin/python benchmark.py > run.log 2>&1', check BOTH speed AND quality results, keep or discard. NEVER STOP. Run experiments until this session ends." \
        --allowedTools 'Edit,Read,Write,Bash,Glob,Grep' \
        --model sonnet 2>&1 | tee -a autostream.log &
    SONNET_PID=$!

    # Wait for timeout or natural exit
    sleep $SONNET_TIMEOUT && kill $SONNET_PID 2>/dev/null &
    TIMER_PID=$!

    # Wait for Sonnet to finish (either by timeout kill or natural completion)
    wait $SONNET_PID 2>/dev/null
    kill $TIMER_PID 2>/dev/null
    wait $TIMER_PID 2>/dev/null

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Round $ROUND — Phase 2: Opus review & strategy"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    claude -p "You are the Opus reviewer for the autostream optimization project. Your job is to analyze what Sonnet did and plan the next round.

Read these files:
1. results.tsv — full experiment history
2. pipeline.py — current pipeline state
3. program.md — project rules and constraints
4. run.log — latest benchmark output

Then do the following:

## Analysis
- How many experiments were run this round?
- What is the current best avg_ms and FPS?
- Which experiments were kept vs discarded vs quality-failed?
- Are there patterns in what works and what doesn't?

## Quality Audit
- Run the benchmark once yourself to verify: .venv/bin/python benchmark.py > run.log 2>&1
- Check quality_pass. If quality is failing, fix pipeline.py to restore quality.
- If the pipeline has drifted into degenerate territory (ignoring input, etc.), reset to a known-good state.

## Strategy for Next Round
- What optimization axes should Sonnet explore next?
- Are there diminishing returns? Should we change approach?
- Write a brief strategy note as the LAST line of results.tsv (as a comment starting with #).

## Cleanup
- Make sure results.tsv is clean and accurate
- Commit any changes you made

When done, output a brief summary of findings and next-round strategy." \
        --allowedTools 'Edit,Read,Write,Bash,Glob,Grep' \
        --model opus 2>&1 | tee -a autostream.log

    echo ""
    echo "  Round $ROUND complete. Starting next round..."
    sleep 5
done
