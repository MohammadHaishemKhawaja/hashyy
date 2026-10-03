#!/usr/bin/env python3
"""Build the page-locked hot-expert file.

Every L1 miss served by the OS page cache costs three passes over DRAM: the cache manager's
memcpy into pinned staging, and then the DMA reading those bytes back out. A file mapping
opened READ-WRITE can be handed to cudaHostRegister on this box (the read-only registration
that ggml normally asks for is unsupported under WDDM), and the GPU can then DMA straight out
of it - one pass, no copy, and the pages are locked so they can never fall back to disk.

This writes the top N GiB of experts, ranked by the undecayed counts in a v2 heat file, into
one contiguous file plus an index that llama-moe-stream mmaps and locks at startup.

usage: mkhot.py <heat> <out> <gib> [gguf-glob]
The index is keyed on the slab's own source location (split index + byte offset in that
split), not on (layer, weight position). The streaming layer registers a layer's weights in
whatever order the model builder creates them - qwen4exp creates ffn_down_exps BEFORE
gate/up - so anything positional silently delivers the wrong tensor, which runs at full
speed and emits "!!!!".

index format (text):
    moe_hot 2 <n_entries>
    <file_idx> <src_offset> <nbytes> <hot_offset>
"""
import sys, os, json, struct, collections

def load_heat(path):
    t = open(path).read().split()
    assert t[0] == "moe_heat", t[0]
    ver = int(t[1]); i = 2
    step = 3 if ver >= 2 else 2
    out = {}
    while i < len(t) and t[i] == "L":
        il = int(t[i+1]); n = int(t[i+2]); i += 3
        rows = []
        for _ in range(n):
            e = int(t[i]); h = int(t[i+1])
            tot = int(t[i+2]) if step == 3 else h
            rows.append((e, tot)); i += step
        out[il] = rows
    return out

def main():
    heat = sys.argv[1]; out = sys.argv[2]; gib = float(sys.argv[3])
    # Experts the VRAM cache holds anyway are wasted space here: the hot file is the tier
    # BEHIND L1, so skip the top `skip` of every layer's ranking (= --moe-stream-cache slots).
    skip = int(sys.argv[4]) if len(sys.argv) > 4 else 67
    mapjson = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gguf_map.json")
    if not os.path.exists(mapjson):
        # derived from the local GGUFs, so it is not in git - just rebuild it
        print("no gguf_map.json, generating it from the model files...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "gguf_map.py")], check=True)
    tens = json.load(open(mapjson))
    # (layer, weight) -> tensor record, in the order the streaming layer registers them
    WORDER = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
    per = collections.defaultdict(dict)
    for t in tens:
        if "_exps" not in t["name"]:
            continue
        parts = t["name"].split(".")
        per[int(parts[1])][parts[2]] = t

    # split index as the loader sees it: 00001-of-00003 -> 0, and so on
    def fidx(path):
        b = os.path.basename(path)
        i = b.rfind("-of-")
        return int(b[i-5:i]) - 1

    ranked = load_heat(heat)
    # one global ranking by undecayed use count - the static top-N by frequency is what the
    # offline residency model scores, and it beat every per-layer split
    allrows = [(tot, il, e) for il, rows in ranked.items() for e, tot in rows[skip:]]
    allrows.sort(reverse=True)

    budget = int(gib * (1 << 30))
    align  = 4096
    entries = []
    off = 0
    for tot, il, e in allrows:
        ws = [per[il][w] for w in WORDER if w in per[il]]
        if not ws:
            continue
        nb = [w["nbytes"] // w["ne"][-1] for w in ws]
        need = sum((n + align - 1)//align*align for n in nb)
        if off + need > budget:
            break
        slabs = []
        for w, n in zip(ws, nb):
            slabs.append((fidx(w["path"]), w["offs"] + e*n, n, off))
            off += (n + align - 1)//align*align
        entries.append((il, e, nb, ws, slabs))

    print("selected %d experts, %.2f GiB (of %d ranked, top %d/layer left to the VRAM cache)"
          % (len(entries), off/2**30, len(allrows), skip))
    fhs = {}
    def fh(path):
        if path not in fhs:
            fhs[path] = open(path, "rb")
        return fhs[path]

    pad = b"\0" * align
    with open(out, "wb", buffering=1 << 22) as o:
        for k, (il, e, nb, ws, slabs) in enumerate(entries):
            for w, n in zip(ws, nb):
                f = fh(w["path"])
                f.seek(w["offs"] + e*n)
                b = f.read(n)
                assert len(b) == n
                o.write(b)
                r = (align - n % align) % align
                if r:
                    o.write(pad[:r])
            if k % 500 == 0:
                print("  %d/%d  %.2f GiB" % (k, len(entries), o.tell()/2**30), flush=True)
    with open(out + ".idx", "w") as ix:
        n_slabs = sum(len(x[4]) for x in entries)
        ix.write("moe_hot 2 %d" % n_slabs + chr(10))
        for il, e, nb, ws, slabs in entries:
            for fi, src, n, ho in slabs:
                ix.write("%d %d %d %d" % (fi, src, n, ho) + chr(10))
    print("wrote %s (%.2f GiB) and .idx" % (out, os.path.getsize(out)/2**30))

if __name__ == "__main__":
    main()
