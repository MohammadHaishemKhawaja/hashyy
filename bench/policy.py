#!/usr/bin/env python3
"""Score cache policies against a recorded routing trace.

The trace is a stream of records: int32 layer, int32 n_ids, int32 ids[n_ids].
Each record is one layer of one token, so a 48-layer model emits 48 per token.

Answers the question the live benchmark cannot: how much of the L1 miss rate is
the policy's fault, and how much is just capacity. BELADY is the floor no online
policy can beat.
"""
import sys, struct, collections

def load(path):
    buf = open(path, "rb").read()
    out = collections.defaultdict(list)   # layer -> [tuple(ids), ...]
    off, n = 0, len(buf)
    while off + 8 <= n:
        layer, n_ids = struct.unpack_from("<ii", buf, off)
        off += 8
        if n_ids < 0 or off + 4*n_ids > n:
            break
        ids = struct.unpack_from("<%di" % n_ids, buf, off)
        off += 4*n_ids
        out[layer].append(ids)
    return out

def sim_lru(seq, cap):
    from collections import OrderedDict
    c, hit, tot = OrderedDict(), 0, 0
    for ids in seq:
        for e in ids:
            tot += 1
            if e in c:
                hit += 1; c.move_to_end(e)
            else:
                c[e] = 1
                if len(c) > cap: c.popitem(last=False)
    return hit, tot

def sim_lfu_decay(seq, cap, decay_every=64, lru_tiebreak=True):
    """what the shipped policy does: decaying route hotness, LRU tiebreak, always admit"""
    heat = collections.Counter()
    last = {}
    res  = set()
    hit = tot = 0
    for t, ids in enumerate(seq):
        if decay_every and t and t % decay_every == 0:
            for k in list(heat): heat[k] >>= 1
        for e in ids:
            tot += 1
            heat[e] += 1
            last[e] = tot
            if e in res:
                hit += 1
                continue
            if len(res) >= cap:
                victim = min(res, key=lambda x: (heat[x], last.get(x, 0)) if lru_tiebreak else heat[x])
                res.discard(victim)
            res.add(e)
    return hit, tot

def sim_lfu_admit(seq, cap, bypass=10, decay_every=64):
    """same, but an expert colder than the coldest resident is not admitted:
    it borrows one of `bypass` scratch slots instead, so it cannot evict a warm entry"""
    heat = collections.Counter()
    last = {}
    res  = set()
    retain = cap - bypass
    hit = tot = 0
    for t, ids in enumerate(seq):
        if decay_every and t and t % decay_every == 0:
            for k in list(heat): heat[k] >>= 1
        for e in ids:
            tot += 1
            heat[e] += 1
            last[e] = tot
            if e in res:
                hit += 1
                continue
            if len(res) < retain:
                res.add(e); continue
            victim = min(res, key=lambda x: (heat[x], last.get(x, 0)))
            if heat[e] > heat[victim]:
                res.discard(victim); res.add(e)
            # else: computed from a bypass slot, nothing cached
    return hit, tot

def sim_belady(seq, cap):
    """offline optimum: evict whatever is needed furthest in the future"""
    flat = []
    for ids in seq:
        flat.extend(ids)
    nxt = collections.defaultdict(collections.deque)
    for i, e in enumerate(flat):
        nxt[e].append(i)
    res = set()
    hit = 0
    for i, e in enumerate(flat):
        nxt[e].popleft()
        if e in res:
            hit += 1
            continue
        if len(res) >= cap:
            def when(x):
                return nxt[x][0] if nxt[x] else float("inf")
            res.discard(max(res, key=when))
        res.add(e)
    return hit, len(flat)

if __name__ == "__main__":
    path = sys.argv[1]
    cap  = int(sys.argv[2]) if len(sys.argv) > 2 else 67
    byl  = len(sys.argv) > 3 and sys.argv[3] == "belady"
    tr = load(path)
    layers = sorted(tr)
    ntok = len(tr[layers[0]]) if layers else 0
    print("layers %d, records/layer %d (~%d tokens), cap %d" % (len(layers), ntok, ntok, cap))

    sims = [("LRU", sim_lru), ("LFU-decay (shipped)", sim_lfu_decay), ("LFU+admit", sim_lfu_admit)]
    if byl:
        sims.append(("BELADY (optimum)", sim_belady))
    for name, fn in sims:
        H = T = 0
        for l in layers:
            h, t = fn(tr[l], cap)
            H += h; T += t
        print("  %-22s %5.1f%%  (misses %.1f/tok)" % (name, 100.0*H/max(1,T),
              (T-H)/max(1,ntok)))

def sim_arc(seq, cap):
    """ARC: adaptive split between a recency list T1 and a frequency list T2, with
    ghost lists B1/B2 driving the split. Scan resistant without tuning."""
    from collections import OrderedDict
    T1, T2 = OrderedDict(), OrderedDict()
    B1, B2 = OrderedDict(), OrderedDict()
    p = 0
    hit = tot = 0

    def replace(e):
        nonlocal p
        if T1 and (len(T1) > p or (e in B2 and len(T1) == p)):
            k, _ = T1.popitem(last=False); B1[k] = 1
        else:
            if T2:
                k, _ = T2.popitem(last=False); B2[k] = 1
            elif T1:
                k, _ = T1.popitem(last=False); B1[k] = 1

    for ids in seq:
        for e in ids:
            tot += 1
            if e in T1:
                hit += 1; del T1[e]; T2[e] = 1; continue
            if e in T2:
                hit += 1; T2.move_to_end(e); continue
            if e in B1:
                p = min(cap, p + max(1, len(B2)//max(1, len(B1))))
                replace(e); del B1[e]; T2[e] = 1; continue
            if e in B2:
                p = max(0, p - max(1, len(B1)//max(1, len(B2))))
                replace(e); del B2[e]; T2[e] = 1; continue
            if len(T1) + len(B1) >= cap:
                if len(T1) < cap and B1:
                    B1.popitem(last=False); replace(e)
                elif T1:
                    T1.popitem(last=False)
            elif len(T1)+len(T2)+len(B1)+len(B2) >= cap:
                if len(T1)+len(T2)+len(B1)+len(B2) >= 2*cap and B2:
                    B2.popitem(last=False)
                replace(e)
            T1[e] = 1
    return hit, tot

def sim_2q(seq, cap, kin=0.25, kout=0.5):
    """2Q: new entries go to a small FIFO A1in; only a second reference inside the
    A1out ghost window promotes to the main LRU."""
    from collections import OrderedDict
    nin  = max(1, int(cap*kin))
    nout = max(1, int(cap*kout))
    A1in, A1out, Am = OrderedDict(), OrderedDict(), OrderedDict()
    hit = tot = 0
    for ids in seq:
        for e in ids:
            tot += 1
            if e in Am:
                hit += 1; Am.move_to_end(e); continue
            if e in A1in:
                hit += 1; continue
            if e in A1out:
                del A1out[e]
                if len(Am) >= cap - len(A1in):
                    if Am: Am.popitem(last=False)
                Am[e] = 1; continue
            if len(A1in) >= nin:
                k, _ = A1in.popitem(last=False)
                A1out[k] = 1
                if len(A1out) > nout: A1out.popitem(last=False)
            while len(A1in) + len(Am) >= cap and Am:
                Am.popitem(last=False)
            A1in[e] = 1
    return hit, tot
