#!/usr/bin/env python3
"""Does the persisted heat map shorten warm-up?

Run 1 starts with no heat file, so it is a normal cold start and writes the ranking
on exit. Run 2 starts from that ranking. Both runs are cold-cache (dropcache) and
measure every generation (--warm 0), because the point is the first few answers.

The ranking comes from run 1's prompts, which are the same 8 prompts run 2 measures.
That overlap is the honest-reporting hazard the warming idea carries: to quote this
as a user-facing number, warm on one prompt set and measure on another. Here the
question is only whether the mechanism works at all.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _root import ROOT
import json, os, subprocess, sys

ROOT = ROOT  # bench/_root.py: $MOE_ROOT, or the parent of bench/
HEAT = os.path.join(ROOT, "bench", "expert.heat")
BEST = "--moe-stream --moe-stream-cache 6 --moe-stream-io-threads 6 --poll 0"
COMMON = "LLAMA_MOE_STREAM_KEEP_MMAP=1;LLAMA_MOE_STREAM_DEBUG=1"

if "--keep" not in sys.argv and os.path.exists(HEAT):
    os.remove(HEAT)
    print("removed existing heat file")

RUNS = [
    ("R_warm_build", ""),                       # no heat file yet -> plain cold start
    ("R_warm_use",   ""),                       # heat file now exists -> preloaded
]

for name, extra in RUNS:
    env = COMMON + ";LLAMA_MOE_STREAM_HEAT=" + HEAT
    if extra:
        env += ";" + extra
    cmd = [sys.executable, os.path.join(ROOT, "bench", "bench.py"),
           "--name", name, "--args", BEST, "--env", env,
           "--warm", "0", "--meas", "4", "--cold"]
    print("\n>>>", name, flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode:
        sys.stdout.write(r.stderr[-1500:])
    sys.stdout.flush()
    if os.path.exists(HEAT):
        print("   heat file: %.1f KiB" % (os.path.getsize(HEAT)/1024.0), flush=True)

print("\n======== warm-up effect (every generation measured) ========")
seen = {}
for line in open(os.path.join(ROOT, "bench", "results.jsonl")):
    d = json.loads(line); seen[d["name"]] = d
for name, _ in RUNS:
    d = seen.get(name)
    if d:
        print("%-14s median %5.2f   per-gen %s" % (name, d["median"],
              " ".join("%.2f" % v for v in d["vals"])))
a, b = seen.get("R_warm_build"), seen.get("R_warm_use")
if a and b:
    print("\nfirst generation: %.2f -> %.2f tok/s" % (a["vals"][0], b["vals"][0]))
