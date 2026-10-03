#!/usr/bin/env python3
"""Start run-full177b.bat exactly as the user would, measure decode, then stop it.

This checks the launcher itself, not the benchmark harness - the flags live in the
.bat, so a mistake there would not show up in bench.py runs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _root import ROOT
import json, os, subprocess, sys, time, urllib.request

ROOT = ROOT  # bench/_root.py: $MOE_ROOT, or the parent of bench/
BAT  = os.path.join(ROOT, "run-full177b.bat")
URL  = "http://127.0.0.1:8080"
LOG  = os.path.join(ROOT, "bench", "launcher_check.log")

PROMPTS = [
    "Write a Python function that merges two sorted lists.",
    "What is the capital of Australia? Answer, then explain briefly.",
    "Explain the difference between TCP and UDP.",
    "Summarise what a B-tree is and when you would use one.",
]

def post(path, payload, timeout=900):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

lf = open(LOG, "wb")
p = subprocess.Popen(["cmd.exe", "/c", BAT], stdout=lf, stderr=subprocess.STDOUT,
                     cwd=ROOT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
print("launched run-full177b.bat (pid %d)" % p.pid, flush=True)

t0 = time.time()
ready = False
while time.time() - t0 < 600:
    try:
        with urllib.request.urlopen(URL + "/health", timeout=3) as r:
            if r.status == 200:
                ready = True
                break
    except Exception:
        time.sleep(3)
print("ready in %.0f s" % (time.time() - t0) if ready else "NEVER BECAME READY", flush=True)

vals, texts = [], []
if ready:
    for i, pr in enumerate(PROMPTS):
        r = post("/completion", {"prompt": pr, "n_predict": 128, "temperature": 0,
                                 "cache_prompt": False, "samplers": ["top_k", "temperature"],
                                 "top_k": 1})
        t = r["timings"]
        vals.append(t["predicted_per_second"])
        texts.append(r.get("content", ""))
        print("  gen%d: %6.2f tok/s" % (i, t["predicted_per_second"]), flush=True)

try:
    import signal
    p.send_signal(signal.CTRL_BREAK_EVENT)
    p.wait(timeout=20)
except Exception:
    pass
subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
lf.close()

if vals:
    import statistics
    print("\nrun-full177b.bat: median %.2f tok/s  (warm steady %.2f)"
          % (statistics.median(vals), max(vals)))
    out = os.path.join(ROOT, "bench", "logs", "out_launcher.txt")
    with open(out, "w", encoding="utf8") as f:
        for i, t in enumerate(texts):
            f.write("--- gen%d | %s%s%s%s" % (i, PROMPTS[i], chr(10), t, chr(10)))
    print("saved generations to", out)
else:
    print("no measurements taken")
    sys.exit(1)
