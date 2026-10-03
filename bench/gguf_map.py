#!/usr/bin/env python3
"""Minimal GGUF reader: tensor name, type, shape, absolute file offset, size.

Enough to locate the routed-expert tensors in the splits so their slabs can be
sampled directly (compressibility, read timing) without going through llama.cpp.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _root import ROOT
import struct, sys, os, glob

GGUF_MAGIC = 0x46554747
# ggml_type -> (name, block_size, type_size)
TYPES = {
 0:("f32",1,4), 1:("f16",1,2), 2:("q4_0",32,18), 3:("q4_1",32,20), 6:("q5_0",32,22),
 7:("q5_1",32,24), 8:("q8_0",32,34), 9:("q8_1",32,36), 10:("q2_K",256,84),
 11:("q3_K",256,110), 12:("q4_K",256,144), 13:("q5_K",256,176), 14:("q6_K",256,210),
 15:("q8_K",256,292), 16:("iq2_xxs",256,66), 17:("iq2_xs",256,74), 18:("iq3_xxs",256,98),
 19:("iq1_s",256,50), 20:("iq4_nl",32,18), 21:("iq3_s",256,110), 22:("iq2_s",256,82),
 23:("iq4_xs",256,136), 24:("i8",1,1), 25:("i16",1,2), 26:("i32",1,4), 27:("i64",1,8),
 28:("f64",1,8), 29:("iq1_m",256,56), 30:("bf16",1,2), 34:("tq1_0",256,54), 35:("tq2_0",256,66),
 39:("mxfp4",32,17),
}

def rd(f, fmt):
    n = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(n))

def rstr(f):
    (n,) = rd(f, "<Q")
    return f.read(n).decode("utf-8", "replace")

def skip_val(f, t):
    # 0..7 scalars, 8 string, 9 array, 10 u64, 11 i64, 12 f64
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t == 8:
        rstr(f)
    elif t == 9:
        (et,) = rd(f, "<I"); (n,) = rd(f, "<Q")
        if et == 8:
            for _ in range(n): rstr(f)
        elif et == 9:
            raise ValueError("nested array")
        else:
            f.seek(sz[et]*n, 1)
    else:
        f.seek(sz[t], 1)

def read_kv(f, want):
    """read the metadata block, returning the values of the keys in `want`"""
    (magic, ver) = rd(f, "<II")
    assert magic == GGUF_MAGIC, hex(magic)
    (n_tensors,) = rd(f, "<Q"); (n_kv,) = rd(f, "<Q")
    got = {}
    for _ in range(n_kv):
        k = rstr(f); (t,) = rd(f, "<I")
        if k in want and t in (4,5,10,11):      # u32/i32/u64/i64
            fmt = {4:"<I",5:"<i",10:"<Q",11:"<q"}[t]
            (got[k],) = rd(f, fmt)
        elif k in want and t == 6:
            (got[k],) = rd(f, "<f")
        elif k in want and t == 8:
            got[k] = rstr(f)
        else:
            skip_val(f, t)
    return n_tensors, got

def tensors(path):
    out = []
    with open(path, "rb") as f:
        n_tensors, kv = read_kv(f, {"general.alignment"})
        align = kv.get("general.alignment", 32)
        for _ in range(n_tensors):
            name = rstr(f)
            (nd,) = rd(f, "<I")
            ne = rd(f, "<%dQ" % nd)
            (tt,) = rd(f, "<I")
            (off,) = rd(f, "<Q")
            out.append([name, tt, list(ne), off])
        data_start = f.tell()
        if data_start % align:
            data_start += align - data_start % align
    res = []
    for name, tt, ne, off in out:
        tn, bs, ts = TYPES.get(tt, ("?%d" % tt, 1, 1))
        n = 1
        for d in ne: n *= d
        nbytes = n // bs * ts
        res.append(dict(name=name, type=tn, ne=ne, offs=data_start+off, nbytes=nbytes, path=path))
    return res

if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, "models", "flashnext", "Qwen3.8-Flash-Next-UD-IQ3_XXS-*.gguf")
    allt = []
    for p in sorted(glob.glob(pat)):
        try:
            ts = tensors(p)
        except Exception as e:
            print("skip", os.path.basename(p), e); continue
        allt += ts
        print("%s: %d tensors, %.2f GiB" % (os.path.basename(p), len(ts),
              sum(t["nbytes"] for t in ts)/2**30))
    exps = [t for t in allt if "_exps" in t["name"]]
    oth  = [t for t in allt if "_exps" not in t["name"]]
    print("\nrouted experts: %d tensors, %.2f GiB" % (len(exps), sum(t["nbytes"] for t in exps)/2**30))
    seen = set()
    for t in exps:
        k = (t["name"].split(".")[-2], t["type"], tuple(t["ne"]))
        if k in seen: continue
        seen.add(k)
        print("   %-28s %-8s ne=%s  %.1f KiB total, %.1f KiB/expert" %
              (t["name"], t["type"], t["ne"], t["nbytes"]/1024, t["nbytes"]/1024/t["ne"][-1]))
    big = sorted(oth, key=lambda t: -t["nbytes"])[:6]
    print("\nlargest non-expert tensors:")
    for t in big:
        print("   %-34s %-8s ne=%s  %.2f GiB" % (t["name"], t["type"], t["ne"], t["nbytes"]/2**30))
    import json
    with open(os.path.join(os.path.dirname(__file__), "gguf_map.json"), "w") as f:
        json.dump(allt, f)
    print("\nwrote gguf_map.json")
