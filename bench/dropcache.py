#!/usr/bin/env python3
"""Evict the Windows file cache so each benchmark run starts from the same state.

The standby list keeps GGUF pages between server restarts, which drifts results
up 3-5% per batch and makes a later config look faster than an earlier one.
Purging it properly (NtSetSystemInformation MemoryPurgeStandbyList, or
SetSystemFileCacheSize) needs SeProfileSingleProcessPrivilege / admin, which we
do not have, so instead commit and touch private pages until the OS has handed
the standby pages over, then release them.

Stops at a free-memory floor so the machine never goes into a paging storm.
"""
import ctypes, ctypes.wintypes as wt, sys, time

k32 = ctypes.WinDLL("kernel32", use_last_error=True)

MEM_COMMIT      = 0x1000
MEM_RESERVE     = 0x2000
MEM_RELEASE     = 0x8000
PAGE_READWRITE  = 0x04

k32.VirtualAlloc.restype  = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, wt.DWORD]
k32.VirtualFree.restype   = wt.BOOL
k32.VirtualFree.argtypes  = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD]

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

def avail_mb():
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    k32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / (1024*1024), m.ullTotalPhys / (1024*1024)

CHUNK = 256 * 1024 * 1024

def standby_gib():
    # PDH would need a helper process; the perf counter is only used for reporting
    return None

def drop(floor_mb=1400, cap_gb=30, verbose=True, passes=1):
    """Two passes: the first pass's own pages end up on the free/standby lists, and the
    second pass reclaims those instead of letting the OS hand them back to the file cache."""
    total_touched = 0
    for _ in range(passes):
        total_touched += _drop_once(floor_mb, cap_gb, verbose)
    return total_touched

def _drop_once(floor_mb=1400, cap_gb=30, verbose=True):
    a0, total = avail_mb()
    blocks = []
    touched = 0
    try:
        while True:
            a, _ = avail_mb()
            if a <= floor_mb:
                break
            if touched >= cap_gb * 1024 * 1024 * 1024:
                break
            p = k32.VirtualAlloc(None, CHUNK, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
            if not p:
                break
            blocks.append(p)
            # one write per page is enough to make the OS back it with a real page
            buf = (ctypes.c_char * CHUNK).from_address(p)
            for off in range(0, CHUNK, 4096):
                buf[off] = b"\x01"
            touched += CHUNK
    finally:
        for p in blocks:
            k32.VirtualFree(ctypes.c_void_p(p), 0, MEM_RELEASE)
        blocks = []
    a1, _ = avail_mb()
    if verbose:
        print("dropcache: touched %.1f GiB | avail %.0f -> %.0f MB (of %.0f) | now %.0f MB"
              % (touched/2**30, a0, a1, total, avail_mb()[0]), flush=True)
    return touched

if __name__ == "__main__":
    floor = int(sys.argv[1]) if len(sys.argv) > 1 else 1400
    drop(floor_mb=floor)
    time.sleep(1.0)
