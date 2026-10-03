#!/usr/bin/env python3
"""Quality gate for a benchmark run's saved generations.

The moe-cache fork has an open Windows bug (their issue #104) where Flash-Next
emits repeated "/" instead of text when the expert cache budgets are wrong. That
failure is fast, so timing alone would rank it as a win. This checks the output is
actually language before any number from the run is trusted.

usage: qcheck.py logs/out_<name>.txt [...]
"""
import re, sys, os, collections

# the model is asked "What is the capital of Australia?" in the fixed prompt set
CANARY_PROMPT = "capital of Australia"
CANARY_WANT   = "canberra"

def check(path):
    txt = open(path, encoding="utf8", errors="replace").read()
    gens = txt.split("--- gen")
    problems = []
    n = 0
    for g in gens[1:]:
        n += 1
        body = g.split(chr(10), 1)[1] if chr(10) in g else ""
        stripped = body.strip()
        if not stripped:
            problems.append("gen%d: empty" % n)
            continue
        # a single character repeated is the #104 signature
        counts = collections.Counter(c for c in stripped if not c.isspace())
        if counts:
            top, hits = counts.most_common(1)[0]
            frac = hits / sum(counts.values())
            if frac > 0.5:
                problems.append("gen%d: %.0f%% of output is '%s' - looks corrupted" % (n, 100*frac, top))
                continue
        # some fraction of it should be ordinary words
        words = re.findall(r"[A-Za-z]{3,}", stripped)
        if len(words) < 10:
            problems.append("gen%d: only %d word-like tokens" % (n, len(words)))

    low = txt.lower()
    canary = None
    if CANARY_PROMPT.lower() in low:
        # look only at the generation that followed that prompt
        after = low.split(CANARY_PROMPT.lower(), 1)[1]
        after = after.split("--- gen", 1)[0]
        canary = CANARY_WANT in after
        if canary is False:
            problems.append("canary: 'Canberra' missing from the Australia answer")

    tag = "FAIL" if problems else "ok"
    print("%-44s %-5s gens=%d canary=%s" % (os.path.basename(path), tag, n,
          {True: "pass", False: "FAIL", None: "n/a"}[canary]))
    for p in problems[:6]:
        print("    " + p)
    return not problems

if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        import glob
        paths = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "logs", "out_*.txt")))
    ok = all(check(p) for p in paths)
    sys.exit(0 if ok else 1)
