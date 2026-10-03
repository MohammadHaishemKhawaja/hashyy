#!/usr/bin/env python3
"""Run a list of configs back to back with clean teardown, then print a summary table."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _root import ROOT
import json, os, subprocess, sys, time

ROOT = ROOT  # bench/_root.py: $MOE_ROOT, or the parent of bench/
BASE = "--moe-stream --moe-stream-cache 6"
COMMON = "LLAMA_MOE_STREAM_KEEP_MMAP=1;LLAMA_MOE_STREAM_DEBUG=1"

# name, extra args, extra env
BEST = "--moe-stream --moe-stream-cache 6 --moe-stream-io-threads 6 --poll 0"
CONFIGS = [
    ("S_ctl",       BEST, ""),
    ("S_l2pin12",   BEST, "LLAMA_MOE_STREAM_L2_GB=12"),
    ("S_l2page12",  BEST, "LLAMA_MOE_STREAM_L2_GB=12;LLAMA_MOE_STREAM_L2_PIN=0"),
]

if len(sys.argv) > 1:
    want = set(sys.argv[1].split(","))
    CONFIGS = [c for c in CONFIGS if c[0] in want]

WARM = os.environ.get("WARM", "3")
MEAS = os.environ.get("MEAS", "4")
NPRED = os.environ.get("NPRED", "128")

for name, args, env in CONFIGS:
    cmd = [sys.executable, os.path.join(ROOT, "bench", "bench.py"),
           "--name", name, "--args", args, "--env", COMMON + ";" + env,
           "--warm", WARM, "--meas", MEAS, "--npred", NPRED]
    if os.environ.get("COLD", "1") != "0":
        cmd.append("--cold")
    print("\n>>>", name, flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stdout.write(r.stderr[-2000:])
    sys.stdout.flush()

print("\n================ SUMMARY ================")
seen = {}
with open(os.path.join(ROOT, "bench", "results.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        seen[d["name"]] = d
for name, _, _ in CONFIGS:
    d = seen.get(name)
    if d:
        print("%-16s %6.2f tok/s   (%s)" % (name, d["median"],
              " ".join("%.2f" % v for v in d["vals"])))
