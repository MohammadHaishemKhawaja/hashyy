#!/usr/bin/env python3
"""Compare two runs at identical cache states.

Routing is deterministic, so the (L1 hit, miss/tok, cold) triple identifies the
same point of the same workload in every run. Matching on it removes the warming
drift that makes raw averages across runs meaningless.
"""
import re, sys, os, collections

PAT = re.compile(
    r"L1 hit\s+([\d.]+)%\s+\|\s+miss\s+([\d.]+)/tok\s+\(\s*(\d+) cold\)"
    r"\s+\|\s+remap\s+([\d.]+)\s+\|\s+stall\s+([\d.]+)\s+\|\s+read\s+([\d.]+)"
    r"\s+\|\s+up\s+([\d.]+)\s+ms/tok\s+\|\s+src\s+([\d.]+)/\s*([\d.]+)/\s*([\d.]+)")

def parse(path):
    out = {}
    for line in open(path, encoding="utf8", errors="replace"):
        m = PAT.search(line)
        if m:
            key = (m.group(1), m.group(2), m.group(3))
            out[key] = dict(stall=float(m.group(5)), read=float(m.group(6)),
                            up=float(m.group(7)), fast=float(m.group(8)),
                            med=float(m.group(9)), slow=float(m.group(10)))
    return out

if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    A, B = parse(a), parse(b)
    common = [k for k in A if k in B]
    if not common:
        print("no matching cache states"); sys.exit(1)
    common.sort(key=lambda k: -float(k[0]))
    print("%-22s | %-28s | %-28s" % ("state (L1hit/miss/cold)", os.path.basename(a), os.path.basename(b)))
    print("%-22s | %8s %7s %8s | %8s %7s %8s" % ("", "stall", "slow", "read", "stall", "slow", "read"))
    ta = tb = 0.0
    for k in common:
        x, y = A[k], B[k]
        ta += x["stall"]; tb += y["stall"]
        print("%-22s | %8.1f %7.1f %8.1f | %8.1f %7.1f %8.1f" % (
            "%s/%s/%s" % k, x["stall"], x["slow"], x["read"], y["stall"], y["slow"], y["read"]))
    print("-"*84)
    print("mean stall over %d matched states: %.1f vs %.1f ms/tok  (%+.1f%%)" % (
        len(common), ta/len(common), tb/len(common), 100.0*(tb-ta)/ta))
