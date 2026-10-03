#!/bin/sh
ROOT="${MOE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Wait for the moe-cache server to link, then benchmark it and gate on output quality.
# Runs unattended: the safe config first, and the rest only if generation is not corrupted.
ROOT=$ROOT
EXE="$ROOT/src/moecache/build-mc/bin/llama-server.exe"
LOG="$ROOT/bench/build_mc2.log"

while [ ! -f "$EXE" ]; do
    if grep -qE "BUILD FAILED|error C[0-9]{4}|LNK1[0-9]{3}" "$LOG" 2>/dev/null; then
        echo "BUILD FAILED"
        grep -E "BUILD FAILED|error C[0-9]{4}|LNK1[0-9]{3}" "$LOG" | head -5
        exit 1
    fi
    sleep 60
done

echo "=== linked ==="
ls -la "$EXE"
# the batch may still be finishing; give the link a moment to settle
sleep 20

cd "$ROOT/bench" || exit 1

echo "=== config 1: windows-safe budgets ==="
python mc.py N_mc_safe48 2>&1 | tail -22

echo "=== quality gate ==="
if python qcheck.py logs/out_N_mc_safe48.txt; then
    echo "QUALITY OK - continuing with the rest"
    python mc.py N_mc_c48_m6g,N_mc_c48_p12g,N_mc_early 2>&1 | tail -40
    echo "=== quality gate, all configs ==="
    python qcheck.py logs/out_N_mc_c48_m6g.txt logs/out_N_mc_c48_p12g.txt logs/out_N_mc_early.txt
else
    echo "QUALITY FAILED on the safe config - not running the rest"
fi
