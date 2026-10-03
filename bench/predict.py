#!/usr/bin/env python3
"""How predictable is a layer's routing from what came before it?

If layer L+1's expert set can be guessed from layer L's, its slabs can be fetched
while layer L computes, which is the only way to overlap I/O with compute - the
router for L+1 does not exist until L is done.

Three predictors, all trained online (counts updated as the trace is replayed, so
no look-ahead):
  prev-token  same layer, same expert set as the previous token
  cooc        co-occurrence: score e' by sum over e in layer L of count[L][e][e']
  marginal    just the globally hottest experts of layer L+1

Reports recall of the true top-10 and how many of the K guesses were wasted.
"""
import sys, collections
from policy import load

def run(tr, K=10):
    layers = sorted(tr)
    ntok = min(len(tr[l]) for l in layers)
    nL = len(layers)

    cooc = collections.defaultdict(collections.Counter)  # (L, e) -> Counter(e')
    marg = collections.defaultdict(collections.Counter)  # L -> Counter(e)
    prev = {}

    stat = {k: [0, 0, 0] for k in ("prev-token", "cooc", "marginal")}  # hit, true, guessed

    warm = ntok//3
    for t in range(ntok):
        for i, l in enumerate(layers):
            truth = set(tr[l][t])

            if t >= warm:
                # prev token, same layer
                g = prev.get(l, set())
                stat["prev-token"][0] += len(truth & g)
                stat["prev-token"][1] += len(truth)
                stat["prev-token"][2] += len(g)

                # marginal
                g = set(e for e, _ in marg[l].most_common(K))
                stat["marginal"][0] += len(truth & g)
                stat["marginal"][1] += len(truth)
                stat["marginal"][2] += len(g)

                # co-occurrence from the layer before it, same token
                if i > 0:
                    src = tr[layers[i-1]][t]
                    sc = collections.Counter()
                    for e in src:
                        sc.update(cooc[(layers[i-1], e)])
                    g = set(e for e, _ in sc.most_common(K))
                    stat["cooc"][0] += len(truth & g)
                    stat["cooc"][1] += len(truth)
                    stat["cooc"][2] += len(g)

            # update the models
            for e in truth:
                marg[l][e] += 1
            if i > 0:
                for a in tr[layers[i-1]][t]:
                    for b in truth:
                        cooc[(layers[i-1], a)][b] += 1
            prev[l] = truth

    print("predicting the top-%d experts of a layer (scored over the last %d tokens)" % (K, ntok-warm))
    for k, (h, tru, g) in stat.items():
        if g == 0:
            continue
        print("  %-11s recall %5.1f%%   precision %5.1f%%   wasted guesses %.1f per layer-token"
              % (k, 100.0*h/max(1, tru), 100.0*h/max(1, g), (g-h)/max(1, (ntok-warm)*len(layers))))

if __name__ == "__main__":
    tr = load(sys.argv[1])
    for K in (10, 16, 24):
        run(tr, K)
        print()
