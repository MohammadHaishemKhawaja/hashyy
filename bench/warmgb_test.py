#!/usr/bin/env python3
"""Does preloading further down the ranking into the OS page cache help?

LLAMA_MOE_STREAM_WARM_GB reads the next N GiB of the saved ranking with ordinary
buffered reads, so those slabs land on the standby list. Unlike the L2 arena this
allocates nothing - it fills the page cache with the right slabs instead of letting
demand misses fill it at random - so it should not pay the arena's zero-sum penalty.
Cost is startup time: roughly N GiB / 2.85 GB/s.

Needs bench/expert.heat to exist already (run warm_test.py first).
Every generation measured, cold cache, so the first-request effect is visible.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _root import ROOT
import json, os, subprocess, sys

ROOT = ROOT  # bench/_root.py: $MOE_ROOT, or the parent of bench/
HEAT = os.path.join(ROOT, "bench", "expert.heat")
BEST = "--moe-stream --moe-stream-cache 6 --moe-stream-io-threads 6 --poll 0"
COMMON = "LLAMA_MOE_STREAM_KEEP_MMAP=1;LLAMA_MOE_STREAM_DEBUG=1;LLAMA_MOE_STREAM_HEAT=" + HEAT

if not os.path.exists(HEAT):
    print("no heat file at", HEAT, "- run warm_test.py first")
    sys.exit(1)
print("heat file: %.1f KiB" % (os.path.getsize(HEAT)/1024.0))

RUNS = [
    ("W_slots_only", ""),                              # device slots only (already measured)
    ("W_gb4",        "LLAMA_MOE_STREAM_WARM_GB=4"),
    ("W_gb8",        "LLAMA_MOE_STREAM_WARM_GB=8"),
]

if len(sys.argv) > 1:
    want = set(sys.argv[1].split(","))
    RUNS = [r for r in RUNS if r[0] in want]

for name, extra in RUNS:
    env = COMMON + (";" + extra if extra else "")
    cmd = [sys.executable, os.path.join(ROOT, "bench", "bench.py"),
           "--name", name, "--args", BEST, "--env", env,
           "--warm", "0", "--meas", "4", "--cold"]
    print("\n>>>", name, extra or "(slots only)", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode:
        sys.stdout.write(r.stderr[-1500:])
    sys.stdout.flush()

print("\n===== page-cache warming sweep (every generation measured, cold) =====")
seen = {}
for line in open(os.path.join(ROOT, "bench", "results.jsonl")):
    d = json.loads(line); seen[d["name"]] = d
for name, extra in RUNS:
    d = seen.get(name)
    if d:
        print("%-14s median %5.2f   per-gen %s" % (name, d["median"],
              " ".join("%.2f" % v for v in d["vals"])))
print("\nstartup cost is in the server logs: 'warmed N device slots + X GiB ... in Y s'")
