#!/usr/bin/env python3
"""Per-layer slot allocation over a recorded routing trace.

Every layer gets the same number of L1 slots today (--moe-stream-cache / n_layer).
Layers do not route alike: some concentrate on a few experts, some spread. This
computes the exact LRU hit curve of every layer with one Mattson stack-distance
pass, then spends a fixed slot budget where the marginal hit is largest.

usage: alloc.py trace [total_slots_per_layer] [skip_frac]
"""
import sys, collections, bisect
from policy import load

class Fenwick:
    def __init__(self, n):
        self.n = n; self.t = [0]*(n+1)
    def add(self, i, v):
        i += 1
        while i <= self.n:
            self.t[i] += v; i += i & -i
    def pre(self, i):            # sum of [0, i)
        s = 0
        while i > 0:
            s += self.t[i]; i -= i & -i
        return s

def stack_distances(seq_flat, n_slots_max):
    """exact LRU reuse distances; returns hits[k] = extra hits gained by slot k (1-indexed)"""
    last = {}                      # expert -> timestamp of last access
    order = []                     # timestamps in access order (we mark/unmark in a BIT)
    bit = Fenwick(len(seq_flat)+1)
    hist = collections.Counter()   # stack distance -> count
    cold = 0
    for t, e in enumerate(seq_flat):
        if e in last:
            p = last[e]
            # distinct items accessed after p = number of live marks strictly after p
            d = bit.pre(t) - bit.pre(p+1)
            hist[d] += 1
            bit.add(p, -1)
        else:
            cold += 1
        bit.add(t, 1)
        last[e] = t
    return hist, cold, len(seq_flat)

def hit_curve(hist, cold, total, cap_max):
    """hits at cache size c = sum of hist[d] for d < c"""
    out = [0]*(cap_max+1)
    run = 0
    for c in range(cap_max+1):
        if c > 0:
            run += hist.get(c-1, 0)
        out[c] = run
    return out

if __name__ == "__main__":
    tr = load(sys.argv[1])
    per_layer = int(sys.argv[2]) if len(sys.argv) > 2 else 67
    skip = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3

    layers = sorted(tr)
    ntok = min(len(tr[l]) for l in layers)
    start = int(ntok*skip)
    CAP = 420

    print("trace: %d layers, %d tokens (scoring from %d)" % (len(layers), ntok, start))

    curves = {}
    totals = {}
    for l in layers:
        flat = [e for t in range(ntok) for e in tr[l][t]]
        hist, cold, tot = stack_distances(flat, CAP)
        # restrict scoring to the steady part: approximate by scaling, the warm-up
        # only adds cold misses which no cache size removes
        curves[l] = hit_curve(hist, cold, tot, CAP)
        totals[l] = tot

    T = sum(totals.values())
    base = sum(curves[l][per_layer] for l in layers)
    print("uniform %d slots/layer: hit %.2f%%  (%d of %d)" % (per_layer, 100*base/T, base, T))

    # greedy: hand out slots one at a time to whichever layer gains most
    budget = per_layer*len(layers)
    alloc  = {l: 1 for l in layers}
    spent  = len(layers)
    import heapq
    h = []
    for l in layers:
        g = curves[l][2] - curves[l][1]
        heapq.heappush(h, (-g, l))
    while spent < budget:
        g, l = heapq.heappop(h)
        alloc[l] += 1
        spent += 1
        c = alloc[l]
        if c < CAP:
            heapq.heappush(h, (-(curves[l][c+1]-curves[l][c]), l))
    got = sum(curves[l][alloc[l]] for l in layers)
    print("greedy  %d slots total : hit %.2f%%  (%d of %d)  -> misses %.1f%% of uniform" %
          (budget, 100*got/T, got, T, 100*(T-got)/(T-base)))
    a = sorted(alloc.items())
    print("alloc per layer:", " ".join("%d:%d" % (l, n) for l, n in a))
    print("min %d max %d" % (min(alloc.values()), max(alloc.values())))

    # per-layer hit at uniform, to see the spread
    print("\nlayer  uniform_hit%  greedy_slots  greedy_hit%")
    for l in layers:
        print("  %2d     %6.2f        %4d        %6.2f" %
              (l, 100*curves[l][per_layer]/totals[l], alloc[l], 100*curves[l][alloc[l]]/totals[l]))
