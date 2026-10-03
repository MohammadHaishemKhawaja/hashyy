#!/usr/bin/env python3
"""Measure the case the server is actually used in: one conversation, one topic.

bench/sweep.py rotates eight unrelated prompts with cache_prompt off, which is the worst
case for an expert cache - every prompt drags a different expert set through it. A real
session stays on a topic, so report both numbers and never just the better one.
"""
import os, statistics, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diskprobe import post
import sweep

TURNS = [
 "Explain how a B-tree index speeds up a database lookup.",
 "How does that change for a clustered index?",
 "What happens to the tree when I insert rows in sorted order?",
 "Write a short Python function that does a binary search over a sorted list.",
 "Now modify it to return the insertion point when the value is missing.",
 "What is the time complexity, and why?",
 "Compare it to a hash index for range queries.",
 "Summarise the tradeoffs in five bullet points.",
]

if __name__ == "__main__":
    env = {}
    for kv in (sys.argv[1] if len(sys.argv) > 1 else "").split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1); env[k.strip()] = v.strip()
    args = (sys.argv[2] if len(sys.argv) > 2 else
            "--moe-stream --moe-stream-cache 6 --moe-stream-io-threads 4 --poll 0").split()
    name = sys.argv[3] if len(sys.argv) > 3 else "chat"

    e = dict(os.environ); e["CUDA_VISIBLE_DEVICES"]="0"
    e["LLAMA_MOE_STREAM_KEEP_MMAP"]="1"; e["LLAMA_MOE_STREAM_DEBUG"]="1"; e.update(env)
    sweep.kill()
    log = open(os.path.join(sweep.LOGD, "srv_%s.log" % name), "wb")
    p = subprocess.Popen([sweep.EXE, "-m", sweep.MODEL] + sweep.BASE + args,
                         stdout=log, stderr=subprocess.STDOUT, env=e)
    try:
        load = sweep.wait_ready(p)
        vals = []
        for i, t in enumerate(TURNS):
            r = post(sweep.URL, "/completion", {"prompt": t, "n_predict": 160, "temperature": 0,
                     "cache_prompt": False, "samplers": ["top_k","temperature"], "top_k": 1})
            v = r["timings"]["predicted_per_second"]; vals.append(v)
            sweep.say("  %-10s turn %d: %6.2f tok/s" % (name, i+1, v))
        sweep.say("%-12s load %3.0fs  turns[%s]  median(last 5) %.2f tok/s"
                  % (name, load, " ".join("%.1f"%v for v in vals), statistics.median(vals[-5:])))
    finally:
        try: p.kill()
        except Exception: pass
        log.close(); sweep.kill()
