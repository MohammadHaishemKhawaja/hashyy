#!/usr/bin/env python
"""Preflight-guarded llama-server launcher.

Refuses configs that would exhaust RAM, and kills the server if free RAM
drops below a floor while running. Use this instead of calling llama-server
directly for any model larger than system RAM.
"""
import os, sys, glob, time, subprocess, threading, argparse, ctypes

LLAMA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "llama", "llama-server.exe")
HEADROOM_GB  = 8.0   # OS + browser + working set we must never eat into
FLOOR_GB     = 2.0   # default; override with --floor-gb (mmap runs can go lower safely)
FLOOR_STRIKES= 3     # consecutive samples under floor before killing

# Flags that force the whole model into real RAM. Fatal if model > RAM.
FORBIDDEN_WHEN_OVERSIZED = {
    "--load-mode": {"dio", "none", "mlock", "mmap+mlock"},
    "--lazy-mode": {"off"},
}
FORBIDDEN_ALWAYS_WHEN_OVERSIZED = {"--no-mmap", "--mlock"}

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

def mem():
    m = MEMORYSTATUSEX(); m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPhys/2**30, m.ullAvailPhys/2**30

def model_bytes(path):
    """Sum all shards of a split GGUF."""
    d, base = os.path.dirname(path), os.path.basename(path)
    if "-00001-of-" in base:
        pat = base.replace("-00001-of-", "-*-of-")
        return sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, pat)))
    return os.path.getsize(path)

def preflight(model, extra):
    total, avail = mem()
    size = model_bytes(model)/2**30
    usable = avail - HEADROOM_GB
    oversized = size > usable
    print(f"  model      : {size:.1f} GB")
    print(f"  RAM        : {total:.1f} GB total, {avail:.1f} GB available")
    print(f"  usable     : {usable:.1f} GB (after {HEADROOM_GB:.0f} GB headroom)")
    print(f"  oversized  : {'YES -> mmap is MANDATORY' if oversized else 'no'}")
    if not oversized:
        return True
    toks = list(extra)
    for bad in FORBIDDEN_ALWAYS_WHEN_OVERSIZED:
        if bad in toks:
            print(f"\nREFUSED: {bad} with a model larger than RAM would allocate "
                  f"{size:.0f} GB into {usable:.0f} GB."); return False
    for flag, badvals in FORBIDDEN_WHEN_OVERSIZED.items():
        if flag in toks:
            v = toks[toks.index(flag)+1] if toks.index(flag)+1 < len(toks) else ""
            if v in badvals:
                print(f"\nREFUSED: {flag} {v} disables mmap/lazy loading. The model "
                      f"({size:.0f} GB) does not fit in {usable:.0f} GB of usable RAM.")
                print(f"         Use: {flag} " +
                      ("mmap" if flag=="--load-mode" else "on")); return False
    if "--load-mode" not in toks: extra += ["--load-mode","mmap"]
    if "--lazy-mode" not in toks: extra += ["--lazy-mode","on"]
    print("  enforced   : --load-mode mmap --lazy-mode on")
    return True

def watchdog(proc, stop):
    strikes = 0
    while not stop.is_set() and proc.poll() is None:
        _, avail = mem()
        if avail < FLOOR_GB:
            strikes += 1
            print(f"  [watchdog] free RAM {avail:.1f} GB < {FLOOR_GB} GB "
                  f"({strikes}/{FLOOR_STRIKES})", flush=True)
            if strikes >= FLOOR_STRIKES:
                print("  [watchdog] KILLING server to protect the system", flush=True)
                proc.kill(); return
        else:
            strikes = 0
        time.sleep(2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m","--model",required=True)
    ap.add_argument("--floor-gb",type=float,default=FLOOR_GB,
                    help="kill server if available RAM stays under this (mmap runs may use ~0.8)")
    ap.add_argument("--strikes",type=int,default=FLOOR_STRIKES)
    args, extra = ap.parse_known_args()
    globals()["FLOOR_GB"]=args.floor_gb
    globals()["FLOOR_STRIKES"]=args.strikes
    print("preflight:")
    if not preflight(args.model, extra):
        print("\nNot launching."); sys.exit(2)
    cmd = [LLAMA,"-m",args.model]+extra
    print("\nlaunching:", " ".join(cmd[2:]), "\n")
    p = subprocess.Popen(cmd)
    stop = threading.Event()
    t = threading.Thread(target=watchdog,args=(p,stop),daemon=True); t.start()
    try: p.wait()
    except KeyboardInterrupt: p.terminate()
    finally: stop.set()

if __name__ == "__main__":
    main()
