import subprocess,time,json,urllib.request,os,sys,signal
D=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llama")
M=os.environ.get("MTP_MODEL") or sys.argv[1]  # path to the dense 27B gguf
PORT=8099
BASE=[os.path.join(D,"llama-server.exe"),"-m",M,"-ngl","99","-sm","tensor","-ts","70/30",
      "-fa","on","-ctk","q8_0","-ctv","q8_0","-c","4096","--port",str(PORT),"--host","127.0.0.1"]
P=("Write a complete Python implementation of a red-black tree with insert, delete, search "
   "and inorder traversal, with docstrings and unit tests.")
def req(path,payload=None,t=600):
    u=f"http://127.0.0.1:{PORT}{path}"
    d=json.dumps(payload).encode() if payload is not None else None
    r=urllib.request.Request(u,data=d,headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(r,timeout=t))
def run(label,extra):
    p=subprocess.Popen(BASE+extra,cwd=D,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        for _ in range(240):
            time.sleep(1)
            try:
                if req("/health",t=3).get("status")=="ok": break
            except Exception: pass
        else: print(label,"SERVER TIMEOUT"); return
        for n in (1,2):  # warm, then measured
            r=req("/completion",{"prompt":P,"n_predict":300,"temperature":0,"seed":1},t=600)
        t=r["timings"]
        extras={k:v for k,v in t.items() if "draft" in k.lower() or "accept" in k.lower()}
        print(f"{label:34s} {t['predicted_per_second']:6.2f} tok/s  "
              f"(pp {t['prompt_per_second']:.0f})  {extras}")
    finally:
        p.terminate()
        try: p.wait(30)
        except Exception: p.kill()
        time.sleep(4)
run("baseline (no spec)",[])
run("MTP n-max=3",["--spec-type","draft-mtp"])
run("MTP n-max=5",["--spec-type","draft-mtp","--spec-draft-n-max","5"])
