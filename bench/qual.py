#!/usr/bin/env python3
"""Quality gate. Any change to the expert path can deliver the wrong bytes into a cache slot
and still run fast, so speed is never reported without this."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _root import ROOT
import json, os, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diskprobe import post

ROOT  = ROOT  # bench/_root.py: $MOE_ROOT, or the parent of bench/
EXE   = os.path.join(ROOT, r"src\lcpp\build-ms\bin\llama-server.exe")
MODEL = os.path.join(ROOT, r"models\flashnext\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf")
PORT  = 8099; URL = f"http://127.0.0.1:{PORT}"

CHECKS = [
    ("What is the capital of Australia? Answer in one word.", ["canberra"]),
    ("What is 17 * 23? Answer with just the number.", ["391"]),
    ("Name the largest planet in our solar system. One word.", ["jupiter"]),
    ("Who wrote the play Hamlet? Answer with the surname only.", ["shakespeare"]),
    ("Complete: the chemical symbol for gold is", ["au"]),
    ("In Python, what does len([1,2,3]) return? Just the number.", ["3"]),
]

def kill():
    subprocess.run(["taskkill","/F","/IM","llama-server.exe"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)

if __name__ == "__main__":
    extra_env = {}
    for kv in (sys.argv[1] if len(sys.argv) > 1 else "").split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1); extra_env[k.strip()] = v.strip()
    args_extra = (sys.argv[2] if len(sys.argv) > 2 else
                  "--moe-stream --moe-stream-cache 6 --moe-stream-io-threads 4 --poll 0").split()
    kill()
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"]="0"
    env["LLAMA_MOE_STREAM_KEEP_MMAP"]="1"; env.update(extra_env)
    args=[EXE,"-m",MODEL,"-ngl","99","-fa","on","-ctk","q8_0","-ctv","q8_0","-ub","256",
          "-c","4096","--jinja","--no-warmup","--host","127.0.0.1","--port",str(PORT)]+args_extra
    log=open(os.path.join(ROOT,r"bench\logs\srv_qual.log"),"wb")
    p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env)
    t0=time.time()
    while True:
        if p.poll() is not None: raise SystemExit("server died")
        try:
            with urllib.request.urlopen(URL+"/health",timeout=3) as r:
                if r.status==200: break
        except Exception: time.sleep(2)
        if time.time()-t0>900: raise SystemExit("never ready")
    ok=0
    for q,keys in CHECKS:
        r=post(URL,"/completion",{"prompt":q,"n_predict":48,"temperature":0,
               "cache_prompt":False,"samplers":["top_k","temperature"],"top_k":1})
        txt=r["content"].strip().replace("\n"," ")
        good=any(k in txt.lower() for k in keys)
        ok+=good
        print(("PASS " if good else "FAIL ")+q)
        print("      -> "+txt[:160], flush=True)
    r=post(URL,"/completion",{"prompt":"Write a short paragraph about why the ocean is salty.",
           "n_predict":90,"temperature":0,"cache_prompt":False,"samplers":["top_k","temperature"],"top_k":1})
    print("\nprose sample:\n"+r["content"].strip()[:400])
    print("\n%d/%d factual checks passed" % (ok,len(CHECKS)))
    p.kill(); log.close(); kill()
