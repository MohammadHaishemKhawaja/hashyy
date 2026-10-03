#!/usr/bin/env python3
"""Measure how many bytes per generated token actually leave the SSD.

The moe-stream debug line buckets slab reads by latency, which cannot separate a
slow page-cache memcpy from a fast NVMe read. This reads the OS physical-disk and
memory counters instead, so the split is unambiguous:

    disk MiB/tok = (PhysicalDisk read bytes during the generation) / tokens generated

Run it against an already-running llama-server (default port 8099).
"""
import json, subprocess, sys, time, urllib.request, argparse, threading, os

RPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diskprobe_out.txt")
def say(*a):
    m = " ".join(str(x) for x in a)
    print(m, flush=True)
    with open(RPT, "a") as f:
        f.write(m+chr(10)); f.flush()

def post(url, path, payload, timeout=1200):
    req = urllib.request.Request(url+path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

PS = r"""
$ErrorActionPreference='Stop'
$rd = New-Object Diagnostics.PerformanceCounter('PhysicalDisk','Disk Read Bytes/sec','_Total')
$rc = New-Object Diagnostics.PerformanceCounter('PhysicalDisk','Disk Reads/sec','_Total')
$av = New-Object Diagnostics.PerformanceCounter('Memory','Available MBytes')
$sb = New-Object Diagnostics.PerformanceCounter('Memory','Standby Cache Normal Priority Bytes')
$null=$rd.NextValue(); $null=$rc.NextValue(); $null=$av.NextValue(); $null=$sb.NextValue()
while ($true) {
  Start-Sleep -Milliseconds 500
  $o = '{0:F0} {1:F1} {2:F0} {3:F0}' -f $rd.NextValue(), $rc.NextValue(), $av.NextValue(), $sb.NextValue()
  Write-Output $o
  [Console]::Out.Flush()
}
"""

class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.rows = []
        self.on = False
        self.p = None
    def run(self):
        self.p = subprocess.Popen(["powershell","-NoProfile","-Command",PS],
                                  stdout=subprocess.PIPE, text=True,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        for line in self.p.stdout:
            parts = line.split()
            if len(parts) != 4: continue
            if self.on:
                self.rows.append((time.time(), float(parts[0]), float(parts[1]),
                                  float(parts[2]), float(parts[3])))

PROMPTS = [
    "Write a Python function that merges two sorted lists.",
    "Explain why the sky appears blue, in three sentences.",
    "List the first eight prime numbers and their sum.",
    "Summarise what a B-tree is and when you would use one.",
    "What is the capital of Australia? Answer, then explain briefly.",
    "Describe the difference between TCP and UDP.",
    "Write a short SQL query to find duplicate email addresses.",
    "Explain gradient descent to someone who knows basic calculus.",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--meas", type=int, default=4)
    ap.add_argument("--npred", type=int, default=128)
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}"

    s = Sampler(); s.start(); time.sleep(2)

    for i in range(a.warm):
        r = post(url, "/completion", {"prompt": PROMPTS[i%len(PROMPTS)], "n_predict": a.npred,
                                      "temperature":0, "cache_prompt": False,
                                      "samplers":["top_k","temperature"], "top_k":1})
        say("  warm%d %.2f tok/s" % (i, r["timings"]["predicted_per_second"]))

    tot_tok = 0; tps = []
    s.rows.clear(); s.on = True
    t0 = time.time()
    for i in range(a.meas):
        pr = PROMPTS[(a.warm+i) % len(PROMPTS)]
        r = post(url, "/completion", {"prompt": pr, "n_predict": a.npred,
                                      "temperature":0, "cache_prompt": False,
                                      "samplers":["top_k","temperature"], "top_k":1})
        t = r["timings"]
        tot_tok += t["predicted_n"]; tps.append(t["predicted_per_second"])
        say("  meas%d %.2f tok/s (%d tok)" % (i, t["predicted_per_second"], t["predicted_n"]))
    s.on = False
    wall = time.time() - t0

    if not s.rows:
        print("no samples"); sys.exit(1)
    dt = 0.5
    rbytes = sum(r[1] for r in s.rows)*dt
    rops   = sum(r[2] for r in s.rows)*dt
    avail  = sum(r[3] for r in s.rows)/len(s.rows)
    stand  = sum(r[4] for r in s.rows)/len(s.rows)/2**30

    say("")
    say("wall %.1f s over %d samples, %d tokens, median %.2f tok/s" %
          (wall, len(s.rows), tot_tok, sorted(tps)[len(tps)//2]))
    say("DISK read      : %8.2f MiB/tok   (%.2f GiB total, %.0f MB/s avg)" %
          (rbytes/2**20/tot_tok, rbytes/2**30, rbytes/wall/1e6))
    say("DISK read ops  : %8.1f /tok      (avg %.0f KiB per op)" %
          (rops/tot_tok, rbytes/max(1,rops)/1024))
    say("mem available  : %8.0f MiB      standby %.2f GiB" % (avail, stand))
