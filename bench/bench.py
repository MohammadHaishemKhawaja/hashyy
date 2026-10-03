#!/usr/bin/env python3
"""Reproducible tok/s harness for the streamed 177B. One server process per config,
fixed prompt set, fixed n_predict; warmup gens discarded, median of the rest reported."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _root import ROOT
import json, os, subprocess, sys, time, urllib.request, urllib.error, argparse, statistics, signal
import dropcache

ROOT = ROOT  # bench/_root.py: $MOE_ROOT, or the parent of bench/
EXE  = os.environ.get("LLAMA_SERVER_EXE",
                      os.path.join(ROOT, r"src\lcpp\build-ms\bin\llama-server.exe"))
MODEL= os.path.join(ROOT, r"models\flashnext\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf")
PORT = 8099
URL  = f"http://127.0.0.1:{PORT}"

# fixed, varied prompts: routing diversity must be identical across configs
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

def post(path, payload, timeout=900):
    req = urllib.request.Request(URL+path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def kill_stragglers():
    # a server that survives teardown holds GB of RAM and silently skews the next run
    subprocess.run(["taskkill","/F","/IM","llama-server.exe"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        r = subprocess.run(["tasklist","/FI","IMAGENAME eq llama-server.exe"],
                           capture_output=True, text=True)
        if "llama-server.exe" not in r.stdout:
            return
        time.sleep(1)
    raise RuntimeError("could not clear llama-server processes")

def wait_ready(proc, timeout=600):
    t0=time.time()
    while time.time()-t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early, rc={proc.returncode}")
        try:
            with urllib.request.urlopen(URL+"/health", timeout=3) as r:
                if r.status==200: return time.time()-t0
        except Exception:
            time.sleep(2)
    raise RuntimeError("server never became ready")

def gen(prompt, npred):
    r = post("/completion", {"prompt": prompt, "n_predict": npred,
                             "temperature": 0, "cache_prompt": False,
                             "samplers": ["top_k","temperature"], "top_k": 1})
    t = r["timings"]
    return t["predicted_per_second"], t["predicted_n"], t.get("prompt_per_second",0), r.get("content","")

def run(name, extra_args, extra_env, warm, meas, npred, logdir, cold=False):
    env = dict(os.environ); env.update(extra_env); env["CUDA_VISIBLE_DEVICES"]="0"
    args = [EXE, "-m", MODEL, "-ngl","99","-fa","on","-ctk","q8_0","-ctv","q8_0",
            "-ub","256","-c","4096","--jinja","--host","127.0.0.1","--port",str(PORT),
            "--no-warmup"] + extra_args
    os.makedirs(logdir, exist_ok=True)
    kill_stragglers()
    if cold:
        # the standby list keeps GGUF pages across server restarts, so without this a
        # config measured later in a batch looks faster than one measured earlier
        dropcache.drop()
    logp = os.path.join(logdir, f"srv_{name}.log")
    print(f"\n=== {name} ===")
    print("  env :", " ".join(f"{k}={v}" for k,v in extra_env.items()) or "(none)")
    print("  args:", " ".join(extra_args))
    with open(logp,"wb") as lf:
        p = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env,
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        try:
            load = wait_ready(p)
            print(f"  ready in {load:.0f}s")
            vals=[]; texts=[]
            for i in range(warm+meas):
                pr = PROMPTS[i % len(PROMPTS)]
                tps, n, pps, txt = gen(pr, npred)
                texts.append(txt)
                tag = "warm" if i<warm else "MEAS"
                print(f"  [{tag}] gen{i}: {tps:6.2f} tok/s  ({n} tok, pp {pps:.1f})")
                if i>=warm: vals.append(tps)
            with open(os.path.join(logdir, f"out_{name}.txt"), "w", encoding="utf8") as tf:
                for i,t in enumerate(texts):
                    tf.write("--- gen%d | %s%s%s%s" % (i, PROMPTS[i%len(PROMPTS)], chr(10), t, chr(10)))
            med = statistics.median(vals)
            print(f"  --> {name}: median {med:.2f} tok/s  min {min(vals):.2f} max {max(vals):.2f}")
            return med, vals
        finally:
            try: p.send_signal(signal.CTRL_BREAK_EVENT)
            except Exception: pass
            try: p.wait(timeout=20)
            except Exception: pass
            kill_stragglers()
            time.sleep(5)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--args", default="")
    ap.add_argument("--env", default="")
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--meas", type=int, default=4)
    ap.add_argument("--npred", type=int, default=128)
    ap.add_argument("--logdir", default=os.path.join(ROOT,"bench","logs"))
    ap.add_argument("--cold", action="store_true", help="evict the file cache before starting the server")
    a = ap.parse_args()
    ee = {}
    for kv in a.env.split(";"):
        if "=" in kv: k,v = kv.split("=",1); ee[k.strip()]=v.strip()
    med, vals = run(a.name, a.args.split(), ee, a.warm, a.meas, a.npred, a.logdir, a.cold)
    with open(os.path.join(ROOT,"bench","results.jsonl"),"a") as f:
        f.write(json.dumps({"name":a.name,"args":a.args,"env":a.env,"cold":a.cold,
                            "median":med,"vals":vals,"npred":a.npred,
                            "ts":time.strftime("%Y-%m-%d %H:%M:%S")})+"\n")
