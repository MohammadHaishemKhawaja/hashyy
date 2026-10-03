#!/usr/bin/env python3
"""Does one shared Windows file HANDLE serialize concurrent positional reads?

llama-moe-stream.cpp opens ONE handle per GGUF split with FILE_FLAG_RANDOM_ACCESS
(no FILE_FLAG_OVERLAPPED) and has every I/O worker ReadFile through it with an
OVERLAPPED offset. On a *synchronous* handle the I/O manager takes the file object
lock for the duration of each read, so those workers queue behind one another no
matter how many there are. This measures that directly:

   shared  = N threads, 1 handle   (what the code does today)
   perthr  = N threads, N handles  (the proposed fix)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _root import ROOT
import ctypes, ctypes.wintypes as wt, os, sys, threading, time, random, argparse

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
GENERIC_READ=0x80000000; FILE_SHARE=0x3; OPEN_EXISTING=3
FLAG_RANDOM=0x10000000; FLAG_NOBUF=0x20000000; FLAG_OVERLAPPED=0x40000000
INVALID=wt.HANDLE(-1).value

class OVERLAPPED(ctypes.Structure):
    _fields_=[("Internal",ctypes.c_void_p),("InternalHigh",ctypes.c_void_p),
              ("Offset",wt.DWORD),("OffsetHigh",wt.DWORD),("hEvent",wt.HANDLE)]

k32.CreateFileW.restype=wt.HANDLE
k32.CreateFileW.argtypes=[wt.LPCWSTR,wt.DWORD,wt.DWORD,ctypes.c_void_p,wt.DWORD,wt.DWORD,wt.HANDLE]
k32.ReadFile.argtypes=[wt.HANDLE,ctypes.c_void_p,wt.DWORD,ctypes.POINTER(wt.DWORD),ctypes.POINTER(OVERLAPPED)]
k32.GetOverlappedResult.argtypes=[wt.HANDLE,ctypes.POINTER(OVERLAPPED),ctypes.POINTER(wt.DWORD),wt.BOOL]
k32.CreateEventW.restype=wt.HANDLE

def openf(path, flags):
    h = k32.CreateFileW(path, GENERIC_READ, FILE_SHARE, None, OPEN_EXISTING, flags, None)
    if h == INVALID: raise ctypes.WinError(ctypes.get_last_error())
    return h

def bench(path, size, nthr, mode, seconds, flags):
    fsz = os.path.getsize(path)
    stop = time.time()+seconds
    counts=[0]*nthr
    align = 4096
    handles = [openf(path, flags)] if mode=="shared" else [openf(path, flags) for _ in range(nthr)]
    def work(i):
        h = handles[0] if mode=="shared" else handles[i]
        buf = ctypes.create_string_buffer(size+align)
        addr = (ctypes.addressof(buf)+align-1)//align*align
        rnd = random.Random(1234+i)
        got = wt.DWORD()
        while time.time() < stop:
            off = rnd.randrange(0, (fsz-size)//align)*align
            ov = OVERLAPPED(); ov.Offset = off & 0xffffffff; ov.OffsetHigh = off>>32
            if not k32.ReadFile(h, ctypes.c_void_p(addr), size, ctypes.byref(got), ctypes.byref(ov)):
                err = ctypes.get_last_error()
                if err != 997: raise ctypes.WinError(err)
                k32.GetOverlappedResult(h, ctypes.byref(ov), ctypes.byref(got), True)
            counts[i]+=1
    t0=time.time()
    ts=[threading.Thread(target=work,args=(i,)) for i in range(nthr)]
    for t in ts: t.start()
    for t in ts: t.join()
    dt=time.time()-t0
    n=sum(counts)
    for h in handles: k32.CloseHandle(wt.HANDLE(h))
    return n*size/dt/1e9, n/dt

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--path", default=os.path.join(ROOT, "models", "flashnext", "Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00003.gguf"))
    ap.add_argument("--secs", type=float, default=4.0)
    a=ap.parse_args()
    print("file: %s (%.1f GiB)" % (os.path.basename(a.path), os.path.getsize(a.path)/2**30))
    for label, flags in (("NOBUF ", FLAG_NOBUF|FLAG_RANDOM), ("buffrd", FLAG_RANDOM)):
        print("\n%s  size      shared-handle        per-thread-handles" % label)
        for size in (458752, 917504):
            row=["  %s %4dKiB " % (label, size//1024)]
            for nthr in (1,4,6,12):
                gs,ops = bench(a.path, size, nthr, "shared", a.secs, flags)
                gp,opp = bench(a.path, size, nthr, "perthr", a.secs, flags)
                row.append("t%-2d %.2f / %.2f GB/s " % (nthr, gs, gp))
            print(" ".join(row))
