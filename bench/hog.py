#!/usr/bin/env python3
"""Hold N GiB of touched private memory, to measure how tok/s responds to the size
of the OS page cache (which is the expert stream's only L2)."""
import sys, time
n = float(sys.argv[1]); secs = float(sys.argv[2]) if len(sys.argv) > 2 else 3600
chunks = []
MB = 1 << 20
for _ in range(int(n*1024)):
    b = bytearray(MB)
    b[::4096] = b"x" * (MB//4096)   # touch every page so it is resident
    chunks.append(b)
print("holding %.1f GiB" % n, flush=True)
time.sleep(secs)
