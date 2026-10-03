#!/usr/bin/env python3
"""Config sweep: one server per config, fixed prompts, tok/s + real disk traffic.

Writes every line to bench/sweep_out.txt as it goes, so a kill or a detach never
loses the run. Pass --cold to evict the standby list before each server.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _root import ROOT
import json, os, subprocess, sys, time, urllib.request, argparse, statistics, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dropcache
from diskprobe import Sampler, PROMPTS, post

ROOT  = ROOT  # bench/_root.py: $MOE_ROOT, or the parent of bench/
EXE   = os.path.join(ROOT, r"src\lcpp\build-ms\bin\llama-server.exe")
MODEL = os.path.join(ROOT, r"models\flashnext\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf")
PORT  = 8099
URL   = f"http://127.0.0.1:{PORT}"
OUT   = os.path.join(ROOT, r"bench\sweep_out.txt")
LOGD  = os.path.join(ROOT, r"bench\logs")

def say(m):
    print(m, flush=True)
    with open(OUT, "a") as f:
        f.write(m + chr(10)); f.flush()

HOT = os.path.join(ROOT, "models", "flashnext", "experts.hot")

def expand(v):
    """${ROOT} and ${HOT} in a config, so cfg_*.json travels between machines."""
    return v.replace("${ROOT}", ROOT).replace("${HOT}", HOT) if isinstance(v, str) else v

def kill():
    subprocess.run(["taskkill","/F","/IM","llama-server.exe"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(180):
        r = subprocess.run(["tasklist","/FI","IMAGENAME eq llama-server.exe"],
                           capture_output=True, text=True)
        if "llama-server.exe" not in r.stdout: return
        time.sleep(1)
    raise RuntimeError("server will not die")

def wait_ready(p, timeout=900):
    t0 = time.time()
    while time.time()-t0 < timeout:
        if p.poll() is not None:
            raise RuntimeError("server exited rc=%s" % p.returncode)
        try:
            with urllib.request.urlopen(URL+"/health", timeout=3) as r:
                if r.status == 200: return time.time()-t0
        except Exception:
            time.sleep(2)
    raise RuntimeError("never ready")

BASE = ["-ngl","99","-fa","on","-ctk","q8_0","-ctv","q8_0","-ub","256","-c","4096",
        "--jinja","--no-warmup","--host","127.0.0.1","--port",str(PORT)]

def run(name, extra_args, extra_env, warm, meas, npred, cold):
    # A measurement must never write the expert ranking: the server checkpoints it every
    # ~256 tokens, so a config that sets this would fold the prompts being measured into the
    # ranking that bench/mkhot.py then builds the hot file from. That is how the first hot
    # file got contaminated.
    # configs are written with ${ROOT} / ${HOT} so they are not tied to one machine
    extra_env = {k: expand(v) for k, v in extra_env.items()}
    extra_args = [expand(a) for a in extra_args]
    if any("LLAMA_MOE_STREAM_HEAT" in k for k in extra_env):
        raise RuntimeError("config %s sets LLAMA_MOE_STREAM_HEAT; a benchmark may not write "
                           "the ranking it is later scored against" % name)
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"]="0"
    env["LLAMA_MOE_STREAM_KEEP_MMAP"]="1"; env["LLAMA_MOE_STREAM_DEBUG"]="1"
    env.update(extra_env)
    kill()
    if cold: dropcache.drop()
    os.makedirs(LOGD, exist_ok=True)
    log = open(os.path.join(LOGD, "srv_%s.log" % name), "wb")
    base = list(BASE)
    if "-c" in extra_args:          # a per-config -c must replace the default, not duplicate it
        i = base.index("-c"); del base[i:i+2]
    args = [EXE, "-m", MODEL] + base + extra_args
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        load = wait_ready(p)
        s = Sampler(); s.start(); time.sleep(1.5)
        for i in range(warm):
            post(URL,"/completion",{"prompt":PROMPTS[i%len(PROMPTS)],"n_predict":npred,
                 "temperature":0,"cache_prompt":False,"samplers":["top_k","temperature"],"top_k":1})
        tps=[]; tok=0
        s.rows.clear(); s.on=True; t0=time.time()
        for i in range(meas):
            r = post(URL,"/completion",{"prompt":PROMPTS[(warm+i)%len(PROMPTS)],"n_predict":npred,
                 "temperature":0,"cache_prompt":False,"samplers":["top_k","temperature"],"top_k":1})
            t=r["timings"]; tps.append(t["predicted_per_second"]); tok+=t["predicted_n"]
        s.on=False; wall=time.time()-t0
        mem = subprocess.run(["powershell","-NoProfile","-Command",
              "$p=Get-Process llama-server -EA SilentlyContinue;"
              r"$c=Get-Counter '\Memory\Standby Cache Normal Priority Bytes','\Memory\Committed Bytes','\Memory\Free & Zero Page List Bytes','\Memory\Modified Page List Bytes' -EA SilentlyContinue;"
              "$v=($c.CounterSamples|%{$_.CookedValue}) -join ' ';"
              "'{0} {1} {2}' -f $p.WorkingSet64,$p.PrivateMemorySize64,$v"],
              capture_output=True, text=True).stdout.split()
        rb = sum(x[1] for x in s.rows)*0.5; ro = sum(x[2] for x in s.rows)*0.5
        sb = sum(x[4] for x in s.rows)/max(1,len(s.rows))/2**30
        med = statistics.median(tps)
        g = lambda i: (float(mem[i])/2**30 if len(mem) > i else 0.0)
        say("%-16s load %3.0fs  med %6.2f tok/s  [%s]  disk %6.1f MiB/tok  %5.0f ops/tok  %4.0f MB/s  "
            "standby %.1f  srvWS %.1f  srvPriv %.1f  commit %.1f  free %.2f GiB"
            % (name, load, med, " ".join("%.1f"%v for v in tps), rb/2**20/tok, ro/tok, rb/wall/1e6,
               sb, g(0), g(1), g(3), g(4)))
        return med
    finally:
        try: p.kill()
        except Exception: pass
        log.close(); kill(); time.sleep(3)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--meas", type=int, default=3)
    ap.add_argument("--npred", type=int, default=128)
    ap.add_argument("--cfg", required=True, help="path to a json list of {name,args,env}")
    a = ap.parse_args()
    cfgs = json.load(open(a.cfg))
    say("=== sweep %s  cold=%s warm=%d meas=%d ===" % (time.strftime("%H:%M:%S"), a.cold, a.warm, a.meas))
    for c in cfgs:
        try:
            run(c["name"], c.get("args","").split(), c.get("env",{}), a.warm, a.meas, a.npred, a.cold)
        except Exception as e:
            say("%-16s FAILED: %r" % (c["name"], e))
    say("=== done ===")
