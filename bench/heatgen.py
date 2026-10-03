#!/usr/bin/env python3
"""Build a deep expert-frequency ranking (bench/expert_train.heat).

Uses a prompt set DISJOINT from the one bench/sweep.py measures with, so a warm
built from this file is never warmed on the prompts it is then scored on.
"""
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
HEAT  = os.path.join(ROOT, r"bench\expert_train.heat")

TRAIN = [
 "Describe the water cycle and name each phase.",
 "Write a Rust function that reverses a linked list in place.",
 "Who composed the Brandenburg Concertos and roughly when?",
 "Explain what a Kalman filter estimates and why it is recursive.",
 "Translate into French: the weather will improve by the weekend.",
 "Outline the causes of the 1929 stock market crash.",
 "Write a bash one-liner that finds the ten largest files under a directory.",
 "What distinguishes mitosis from meiosis?",
 "Derive the quadratic formula by completing the square.",
 "Summarise the plot of Moby-Dick in one paragraph.",
 "Explain how a DNS recursive resolver answers a query.",
 "List five common logical fallacies with a one-line example each.",
 "Write a regular expression that matches an ISO-8601 date.",
 "How does a lithium-ion cell store and release charge?",
 "Compare REST and GraphQL for a mobile client.",
 "Explain the Monty Hall problem and the correct strategy.",
 "Write a haiku about a lighthouse in winter.",
 "What does the Pauli exclusion principle state?",
 "Give a worked example of Bayes' theorem with real numbers.",
 "Explain memory-mapped I/O to a systems programmer.",
 "Describe how vaccines train the adaptive immune system.",
 "Write a SQL schema for a library lending system.",
 "What were the main provisions of the Treaty of Westphalia?",
 "Explain tail recursion and when a compiler can eliminate it.",
]

def kill():
    subprocess.run(["taskkill","/F","/IM","llama-server.exe"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)

if __name__ == "__main__":
    npred = int(sys.argv[1]) if len(sys.argv) > 1 else 192
    kill()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"]="0"; env["LLAMA_MOE_STREAM_KEEP_MMAP"]="1"
    env["LLAMA_MOE_STREAM_DEBUG"]="1"; env["LLAMA_MOE_STREAM_HEAT"]=HEAT
    args=[EXE,"-m",MODEL,"--moe-stream","--moe-stream-cache","6","--moe-stream-io-threads","10",
          "--poll","0","-ngl","99","-fa","on","-ctk","q8_0","-ctv","q8_0","-ub","256","-c","4096",
          "--jinja","--no-warmup","--host","127.0.0.1","--port",str(PORT)]
    log=open(os.path.join(ROOT,r"bench\logs\srv_heatgen.log"),"wb")
    p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env)
    t0=time.time()
    while True:
        if p.poll() is not None: raise SystemExit("server died")
        try:
            with urllib.request.urlopen(URL+"/health",timeout=3) as r:
                if r.status==200: break
        except Exception: time.sleep(2)
        if time.time()-t0>900: raise SystemExit("never ready")
    for i,pr in enumerate(TRAIN):
        r=post(URL,"/completion",{"prompt":pr,"n_predict":npred,"temperature":0,
               "cache_prompt":False,"samplers":["top_k","temperature"],"top_k":1})
        print("%2d/%d %.2f tok/s" % (i+1,len(TRAIN),r["timings"]["predicted_per_second"]), flush=True)
    # the heat file is checkpointed every ~256 tokens; one more flush then a clean kill
    p.kill(); log.close(); kill()
    if os.path.exists(HEAT):
        import collections
        s=open(HEAT).read().split()
        ver=int(s[1]); i=2; tot=0; lay=0
        step = 3 if ver>=2 else 2
        while i<len(s) and s[i]=="L":
            n=int(s[i+2]); i+=3+step*n; tot+=n; lay+=1
        print("heat v%d: %d layers, %d ranked experts = %.2f GiB deep" % (ver,lay,tot,tot*1.886/1024))
    else:
        print("NO HEAT FILE WRITTEN")
