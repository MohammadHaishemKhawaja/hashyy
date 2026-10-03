#!/usr/bin/env python3
"""Achievable host memcpy bandwidth, 1..10 threads, at expert-slab size.

Every L1 miss that the page cache serves costs one 512-900 KiB memcpy into pinned staging
and then a DMA read of the same bytes, so this number and the PCIe number together bound
the stall. The DIMMs are a DDR4-3600 kit running at 2400.
"""
import ctypes, threading, time, sys

SZ = 626*1024
def bench(nthr, secs=2.0):
    bufs = [(ctypes.create_string_buffer(SZ), ctypes.create_string_buffer(SZ)) for _ in range(nthr)]
    stop = time.time()+secs
    cnt = [0]*nthr
    memmove = ctypes.memmove
    def w(i):
        src, dst = bufs[i]
        a = ctypes.addressof(src); b = ctypes.addressof(dst)
        n = 0
        while time.time() < stop:
            for _ in range(64):
                memmove(b, a, SZ)
            n += 64
        cnt[i] = n
    t0=time.time()
    ts=[threading.Thread(target=w,args=(i,)) for i in range(nthr)]
    for t in ts: t.start()
    for t in ts: t.join()
    dt=time.time()-t0
    return sum(cnt)*SZ/dt/1e9

if __name__ == "__main__":
    print("memcpy %d KiB buffers (read+write, so DRAM traffic is 2x the figure)" % (SZ//1024))
    for n in (1,2,4,6,10):
        print("  threads %2d : %6.2f GB/s copied  (%.1f GB/s of DRAM traffic)" % (n, bench(n), 2*bench(n,1.0)))
