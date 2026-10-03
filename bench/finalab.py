#!/usr/bin/env python3
"""Headline A/B: the streaming path as inherited vs the best configuration found.
Both cold-started and interleaved, so neither inherits the other's page cache."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _root import ROOT
import json, os, subprocess, sys

ROOT = ROOT  # bench/_root.py: $MOE_ROOT, or the parent of bench/
COMMON = "LLAMA_MOE_STREAM_KEEP_MMAP=1;LLAMA_MOE_STREAM_DEBUG=1"
INHERITED_ENV = ("LLAMA_MOE_STREAM_SERIAL_IO=1;LLAMA_MOE_STREAM_SPLIT_WEIGHTS=0;"
                 "LLAMA_MOE_STREAM_PINNED=0")

CONFIGS = [
    ("M_inherited",  "--moe-stream --moe-stream-cache 6", INHERITED_ENV),
    ("M_best",       "--moe-stream --moe-stream-cache 6 --moe-stream-io-threads 6 --poll 0", ""),
    ("M_inherited2", "--moe-stream --moe-stream-cache 6", INHERITED_ENV),
    ("M_best2",      "--moe-stream --moe-stream-cache 6 --moe-stream-io-threads 6 --poll 0", ""),
]

for name, args, env in CONFIGS:
    cmd = [sys.executable, os.path.join(ROOT, "bench", "bench.py"),
           "--name", name, "--args", args, "--env", COMMON + (";"+env if env else ""),
           "--warm", "6", "--meas", "4", "--cold"]
    print("\n>>>", name, flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode: sys.stdout.write(r.stderr[-1200:])
    sys.stdout.flush()

print("\n========= HEADLINE A/B (cold, interleaved, 6 warmup gens) =========")
seen = {}
for line in open(os.path.join(ROOT, "bench", "results.jsonl")):
    d = json.loads(line); seen[d["name"]] = d
for name, _, _ in CONFIGS:
    d = seen.get(name)
    if d: print("%-14s %6.2f tok/s   (%s)" % (name, d["median"], " ".join("%.2f"%v for v in d["vals"])))
