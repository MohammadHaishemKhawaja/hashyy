#!/usr/bin/env python3
"""Two-level residency model over a recorded routing trace.

L1 is the per-layer VRAM slot cache; L2 is one global host-RAM tier shared by all
layers (that is what both the OS page cache and the explicit arena are). Reports
where each expert access is served from, which is what sets the stall time.

usage: hier.py trace [l1_slots] [l2_experts] [skip_frac]
"""
import sys, collections
from policy import load

class LRU:
    def __init__(self, cap):
        self.cap = cap
        self.d = collections.OrderedDict()
    def get(self, k):
        if k in self.d:
            self.d.move_to_end(k)
            return True
        return False
    def put(self, k):
        self.d[k] = 1
        self.d.move_to_end(k)
        if len(self.d) > self.cap:
            self.d.popitem(last=False)

def run(tr, l1_slots, l2_experts, skip=0.3):
    layers = sorted(tr)
    ntok = min(len(tr[l]) for l in layers)   # the last token can be cut short mid-layer
    start = int(ntok*skip)

    l1 = {l: LRU(l1_slots) for l in layers}
    l2 = LRU(l2_experts)           # keyed (layer, expert), global

    n = h1 = h2 = d = 0
    for t in range(ntok):
        for l in layers:
            for e in tr[l][t]:
                scoring = t >= start
                if scoring: n += 1
                if l1[l].get(e):
                    if scoring: h1 += 1
                    continue
                # L1 miss: the slab has to come from L2 or disk, then enters L1
                if l2.get((l, e)):
                    if scoring: h2 += 1
                else:
                    if scoring: d += 1
                    l2.put((l, e))
                l1[l].put(e)
    return n, h1, h2, d

MIB = 1.815  # bytes per expert, all three weights

def report(tag, n, h1, h2, d, ntok):
    per = lambda x: x/max(1, ntok)
    disk_mib = per(d)*MIB
    ram_mib  = per(h2)*MIB
    # 2.8 GB/s disk, ~6 GB/s for a host->device DMA out of RAM, overlapped
    t_disk = disk_mib/2800.0*1000
    t_ram  = ram_mib/6000.0*1000
    stall  = max(t_disk, t_ram) + 0.35*min(t_disk, t_ram)
    print("  %-22s L1 %4.1f%%  L2 %4.1f%%  disk %4.1f%%  | %6.1f MiB/tok disk"
          "  -> stall ~%5.1f ms  => ~%4.1f tok/s" %
          (tag, 100*h1/n, 100*h2/n, 100*d/n, disk_mib, stall, 1000.0/(stall+20)))

if __name__ == "__main__":
    tr = load(sys.argv[1])
    layers = sorted(tr); ntok = min(len(tr[l]) for l in layers)
    scored = ntok - int(ntok*0.3)
    print("trace: %d layers, %d tokens (scoring last %d)" % (len(layers), ntok, scored))
    print("L1 slots x L2 experts (per layer in brackets):")
    for l1s in (67, 80, 100):
        for l2gb in (0, 8, 12, 16, 20, 26):
            l2e = int(l2gb*1024/MIB)
            n, h1, h2, d = run(tr, l1s, l2e)
            report("L1=%d L2=%dGiB[%d]" % (l1s, l2gb, l2e//len(layers)), n, h1, h2, d, scored)
        print()
